"""
Jax implementation
This file implements our KSDAgg test in the function ksdagg().
For details, see our paper:
KSD Aggregated Goodness-of-fit Test
Antonin Schrab, Benjamin Guedj, Arthur Gretton
"""

import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from jax.flatten_util import ravel_pytree
from functools import partial


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 9, 10, 11, 12))
def ksdagg(
    X,
    score_X,
    alpha=0.05,
    number_bandwidths=10,
    weights_type="uniform", 
    approx_type="wild bootstrap",
    kernel="imq",
    B1=2000, 
    B2=2000, 
    B3=50,
    seed=42,
    return_dictionary=False,
    bandwidths=None,
):
    """
    Goodness-of-fit KSDAgg test. 
    
    Given data and its score under the model, 
    return 0 if the test fails to reject the null (i.e. fits the data), 
    or return 1 if the test rejects the null (i.e. does not fit the data).
    
    Parameters
    ----------
    X : array_like
        The shape of X must be of the form (m, d) where m is the number
        of samples and d is the dimension.
    score_X: array_like
        The shape of score_X must be the same as the shape of X.
    alpha: scalar
        The value of alpha must be between 0 and 1.
    number_bandwidths: int
        The number of bandwidths to include in the collection.
    weights_type: str
        Must be "uniform", or "centred", or "increasing", or "decreasing".
    approx_type: str
        Must be "wild bootstrap" or "parametric".
        Note that the required type of B1 and B2 depends on approx_type.
    kernel: str
        The value of kernel must be "imq".
    B1: int or array_like
        If approx_type is "wild bootstrap", then B1 should be an integer which corresponds 
        to the number of wild bootstrap samples to approximate the quantiles.
        If approx_type is "parametric", then B1 should be an array of shape (B1int, number_bandwidths)
        consisting of B1int KSD values computed under parametric bootstrap for number_bandwidths bandwidths.
        It is used to approximate the quantiles.
        Note that number_bandwidths is overwritten to be the second dimension of B1.
    B2: int or array_like
        If approx_type is "wild bootstrap", then B2 should be an integer which corresponds 
        to the number of wild bootstrap samples to approximate the level correction.
        If approx_type is "parametric, then B2 should be an array of shape (B2int, number_bandwidths)
        consisting of B2int KSD values computed under parametric bootstrap for number_bandwidths bandwidths.
        It is used to approximate the level correction.
        Note that number_bandwidths is overwritten to be the second dimension of B2.
    B3: int
        Number of steps of bissection method to perform to estimate the level correction.
    seed: int 
        Random seed used for the randomness of the Rademacher variables when approx_type is "wild bootstrap.
    return_dictionary: bool
        If true, a dictionary is returned containing for each single test: the test output, the kernel,
        the bandwidth, the KSD value, the KSD quantile value, the p-value and the p-value threshold value.
    bandwidths: array_like or None
        If bandwidths is None, the collection of bandwidths is computed automatically.
        If bandwidths is array_like of one dimension, the provided collection is used instead.
        Note that number_bandwidths is overwritten by the length of bandwidths.
        
    Returns
    -------
    output : int
        0 if the aggregated KSDAgg test fails to reject the null (i.e. fits the data)
        1 if the aggregated KSDAgg test rejects the null (i.e. does not fit the data)
    dictionary: dict
        Returned only if return_dictionary is True.
        Dictionary containing the overall output of the KSDAgg test, and for each single test: 
        the test output, the kernel, the bandwidth, the KSD value, the KSD quantile value, 
        the p-value and the p-value threshold value.
    
    Examples
    --------
    >>> perturbation = 0.5
    >>> rs = np.random.RandomState(0)
    >>> X = rs.gamma(5 + perturbation, 5, (500, 1))
    >>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
    >>> score_X = score_gamma(X, 5, 5)
    >>> output = ksdagg(X, score_X)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = ksdagg(X, score_X, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from ksdagg.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'KSDAgg test reject': True,
     'Single test 1': {'Bandwidth': 1.0,
      'KSD': 5.788900671177544e-05,
      'KSD quantile': 0.0009193826699629426,
      'Kernel IMQ': True,
      'Reject': False,
      'p-value': 0.41079461574554443,
      'p-value threshold': 0.01699146442115307},
      ...
    }
    """
    # Assertions
    m = X.shape[0]
    assert m >= 2 and X.shape == score_X.shape
    if approx_type == "wild bootstrap":
        assert B1 > 0 and B2 > 0 and type(B1) == type(B2) == int
    elif approx_type == "parametric":
        assert isinstance(B1, jnp.ndarray)
        B1_parametric = B1
        B2_parametric = B2
        assert B1_parametric.shape[0] == B2_parametric.shape[0]
        B1 = B1_parametric.shape[1]
        B2 = B2_parametric.shape[1]
    else:
        raise ValueError("approx_type must be either 'wild bootstrap' or 'parametric'.")
    assert 0 < alpha  and alpha < 1
    assert kernel in ("imq", )
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert weights_type in ("uniform", "decreasing", "increasing", "centred")
    assert B3 > 0 and type(B3) == int
    
    if type(bandwidths) == jnp.ndarray:
        assert bandwidths.ndim == 1
        number_bandwidths = len(bandwidths)
        if approx_type == "parametric":
            assert B1_parametric.shape[0] == B2_parametric.shape[0] == len(bandwidths)
    else:
        # Collection of bandwidths
        max_samples = 500
        distances = jax_distances(X, X, max_samples)     
        lambda_min = 1
        lambda_max = jnp.maximum(jnp.max(distances), 2)
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = jnp.array([power ** i * lambda_min / X.shape[1] for i in range(number_bandwidths)])
    
    # Weights 
    weights = create_weights(number_bandwidths, weights_type)

    # Step 1: compute all simulated KSD estimates efficiently
    if approx_type == "wild bootstrap":
        M = jnp.zeros((number_bandwidths, B1 + B2 + 1))
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        R = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(B1 + B2 + 1, m))  # (B1+B2+1, m) Rademacher
        R = R.at[B1].set(jnp.ones(m))
        R = R.transpose()  # (m, B1+B2+1)
        # IMQ kernel
        beta_imq = 0.5
        p = X.shape[1]
        norms = jnp.sum(X ** 2, -1)
        dists = -2 * X @ X.T + jnp.expand_dims(norms, 1) + jnp.expand_dims(norms, 0)
        diffs = jnp.expand_dims(jnp.sum(X * score_X, -1), 1) - (X @ score_X.T)
        diffs = diffs + diffs.T
        for i in range(number_bandwidths):
            bandwidth = bandwidths[i]
            g = 1 / bandwidth ** 2
            res = 1 + g * dists
            kxy = res ** (-beta_imq)
            dkxy = 2 * beta_imq * g * (res) ** (-beta_imq - 1) * diffs
            d2kxy = 2 * (
                beta_imq * g * (res) ** (-beta_imq - 1) * p
                - 2
                * beta_imq
                * (beta_imq + 1)
                * g ** 2
                * dists
                * res ** (-beta_imq - 2)
            )
            H = score_X @ score_X.T * kxy + dkxy + d2kxy
            H = H.at[jnp.diag_indices(H.shape[0])].set(0)
            M = M.at[i].set(jnp.sum(R * (H @ R), 0) / (m * (m - 1))) # (B1+B2+1, ) wild bootstrap KSD estimates
        KSD_original = M[:, B1]
        M1_sorted = jnp.sort(M[:, :B1 + 1])  # (number_bandwidths, B1+1)
        M2 = M[:, B1 + 1:]  # (number_bandwidths, B2)
    elif approx_type == "parametric":
        # IMQ kernel
        beta_imq = 0.5
        p = X.shape[1]
        norms = jnp.sum(X ** 2, -1)
        dists = -2 * X @ X.T + jnp.expand_dims(norms, 1) + jnp.expand_dims(norms, 0)
        diffs = jnp.expand_dims(jnp.sum(X * score_X, -1), 1) - (X @ score_X.T)
        diffs = diffs + diffs.T
        KSD_original = jnp.zeros((number_bandwidths, ))
        for i in range(number_bandwidths):
            bandwidth = bandwidths[i]
            g = 1 / bandwidth ** 2
            res = 1 + g * dists
            kxy = res ** (-beta_imq)
            dkxy = 2 * beta_imq * g * (res) ** (-beta_imq - 1) * diffs
            d2kxy = 2 * (
                beta_imq * g * (res) ** (-beta_imq - 1) * p
                - 2
                * beta_imq
                * (beta_imq + 1)
                * g ** 2
                * dists
                * res ** (-beta_imq - 2)
            )
            H = score_X @ score_X.T * kxy + dkxy + d2kxy
            H = H.at[jnp.diag_indices(H.shape[0])].set(0)
            r = jnp.ones(m)
            KSD_original = KSD_original.at[i].set(r @ H @ r  / (m * (m - 1)))
        M1 = jnp.concatenate((B1_parametric, KSD_original.reshape(-1, 1)), axis=1)  # (number_bandwidths, B1+1)
        M1_sorted = jnp.sort(M1)  # (number_bandwidths, B1+1)
        M2 = B2_parametric  # (number_bandwidths, B2)
        
    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = jnp.zeros((number_bandwidths, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0.
    u_max = jnp.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for i in range(number_bandwidths):
            quantiles = quantiles.at[i].set(
                M1_sorted[
                    i, 
                    (jnp.ceil((B1 + 1) * (1 - u * weights[i]))).astype(int) - 1
                ]
            )
        P_u = jnp.sum(jnp.max(M2 - quantiles, 0) > 0) / B2
        u_min, u_max = jax.lax.cond(P_u <= alpha, lambda: (u, u_max), lambda: (u_min, u))
    u = u_min
    for i in range(number_bandwidths):
        quantiles = quantiles.at[i].set(
            M1_sorted[
                i, 
                (jnp.ceil((B1 + 1) * (1 - u * weights[i]))).astype(int) - 1
            ]
        )
        
    # Step 3: output test result
    p_vals = jnp.mean((M1_sorted - KSD_original.reshape(-1, 1) >= 0), -1)
    thresholds = u * weights
    # reject if p_val <= threshold
    reject_p_vals = p_vals <= thresholds

    ksd_vals = KSD_original
    quantiles = quantiles.reshape(-1)
    # reject if ksd_val > quantile
    reject_ksd_vals = ksd_vals > quantiles

    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary["KSDAgg test reject"] = False
    for i in range(number_bandwidths):
        index = "Single test " + str(i + 1)
        reject_dictionary[index] = {}
        reject_dictionary[index]["Reject"] = reject_p_vals[i]
        reject_dictionary[index]["Kernel IMQ"] = True
        reject_dictionary[index]["Bandwidth"] = bandwidths[i]
        reject_dictionary[index]["KSD"] = ksd_vals[i]
        reject_dictionary[index]["KSD quantile"] = quantiles[i]
        reject_dictionary[index]["p-value"] = p_vals[i]
        reject_dictionary[index]["p-value threshold"] = thresholds[i]
        # Aggregated test rejects if one single test rejects
        reject_dictionary["KSDAgg test reject"] = jnp.any(ravel_pytree(
            (reject_dictionary["KSDAgg test reject"],
            reject_p_vals[i])
        )[0])

    if return_dictionary:
        return (reject_dictionary["KSDAgg test reject"]).astype(int), reject_dictionary
    else:
        return (reject_dictionary["KSDAgg test reject"]).astype(int)


def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of MMD Aggregated Two-Sample Test (Schrab et al., 2021).
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = jnp.array(
            [
                1 / N,
            ]
            * N
        )
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = jnp.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = jnp.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = jnp.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = jnp.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights


def jax_distances(X, Y, max_samples):
    def dist(x, y):
        z = x - y
        return jnp.sqrt(jnp.sum(jnp.square(z)))
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    output = output[jnp.triu_indices(output.shape[0])]
    return output

def human_readable_dict(dictionary):
    """
    Transform all jax arrays of one element into scalars.
    """
    meta_keys = dictionary.keys()
    for meta_key in meta_keys:
        if isinstance(dictionary[meta_key], jnp.ndarray):
            dictionary[meta_key] = dictionary[meta_key].item()
        elif isinstance(dictionary[meta_key], dict):
            for key in dictionary[meta_key].keys():
                if isinstance(dictionary[meta_key][key], jnp.ndarray):
                    dictionary[meta_key][key] = dictionary[meta_key][key].item()