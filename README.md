# Code for KSDAgg: KSD Aggregated Goodness-of-fit Test

This GitHub repository contains the code for the reproducible experiments presented in our paper 
[KSD Aggregated Goodness-of-fit Test](https://arxiv.org/pdf/2202.00824.pdf):
- Gamma distribution experiment,
- Gaussian-Bernoulli Restricted Boltzmann Machine experiment,
- MNIST Normalizing Flow experiment.

We provide the code to run the experiments to generate Figures 1-4 and Table 1 from our paper, 
those can be found in [figures](figures). 

Our aggregated test [KSDAgg](https://arxiv.org/pdf/2202.00824.pdf#page=4) is implemented in [ksdagg.py](ksdagg.py), we provide code below explaining how to use KSDAgg in practice.

Our implementation uses two quantile estimation methods (the wild bootstrap and the parametric bootstrap) with the IMQ (inverse multiquadric) kernel.
The KSDAgg test aggregates over a collection of bandwidths, and uses one of the four types of weights proposed in [MMD Aggregated Two-Sample Test](https://arxiv.org/pdf/2110.15073.pdf#page=22).

## Requirements
- `python 3.9`

## Installation

In a chosen directory, clone the repository and change to its directory by executing 
```
git clone git@github.com:antoninschrab/ksdagg-paper.git
cd ksdagg-paper
```
We then recommend creating and activating a virtual environment by either 
- using `venv`:
  ```
  python3 -m venv ksdagg-env
  source ksdagg-env/bin/activate
  # can be deactivated by running:
  # deactivate
  ```
- or using `conda`:
  ```
  conda create --name ksdagg-env python=3.9
  conda activate ksdagg-env
  # can be deactivated by running:
  # conda deactivate
  ```
The packages required for reproducibility of the experiments can then be installed in the virtual environment by running
```
python -m pip install -r requirements.txt
```
Note that, in order to run the `ksdagg` function from [ksdagg.py](ksdagg.py), only the `numpy` and `scipy` packages are required, those can be installed on their own by instead running
```
python -m pip install numpy scipy
```

## Generating or downloading the data 

The data for the Gaussian-Bernoulli Restricted Boltzmann Machine experiment
and for the MNIST Normalizing Flow experiment can
- be obtained by executing
  ```
  python generate_data_rbm.py
  python generate_data_nf.py
  ```
- or, as running the above scripts can be computationally expensive, we also provide the option to download their outputs directly
  ```
  python download_data.py
  ```
Those scripts generate samples and compute their associated scores under the model for the different settings considered in our [experiments](https://arxiv.org/pdf/2202.00824.pdf#page=7), the data is saved in the new directory `data`.

## Reproducing the experiments of the paper

First, for our three experiments, we compute KSD values to be used for the parametric bootstrap and save them in the directory [parametric](parametric). 
This can be done by running
```
python generate_parametric.py
```
For convenience, we directly provide the directory [parametric](parametric) obtained by running this script.

To run the three experiments, the following commands can be executed
```
python experiment_gamma.py 
python experiment_rbm.py 
python experiment_nf.py 
```
Those commands run all the tests necessary for our experiments, the results are saved in dedicated `.csv` and `.pkl` files in the directory [results](results) (which is already provided for ease of use).
Note that our experiments are comprised of 'embarrassingly parallel for loops', for which significant speed up can be obtained by using 
parallel computing libraries such as `joblib` or `dask`.

The actual figures of the paper can be obtained from the saved dataframes in [results](results) by using the command 
```
python figures.py  
```
The figures are saved in the directory [figures](figures) and correspond to the ones used in our [paper](https://arxiv.org/pdf/2202.00824.pdf).

## Example code for using KSDAgg

```python
# import modules
>>> import numpy as np 
>>> from ksdagg import ksdagg

# generate data
>>> perturbation = 0.5
>>> rs = np.random.RandomState(0)
>>> X = rs.gamma(5 + perturbation, 5, (500, 1))
>>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
>>> score_X = score_gamma(X, 5, 5)

# run KSDAgg test
>>> output = ksdagg(X, score_X)
>>> output
1

# run KSDAgg test with dictionary details
>>> output, dictionary = ksdagg(X, score_X, return_dictionary=True)
>>> output
1
>>> dictionary
{'KSDAgg aggregated test reject': True,
 'Single test 1': {'Reject': False,
  'Kernel': 'imq',
  'Bandwidth': 1.0,
  'KSD': 131.51170316873277,
  'KSD quantile': 300.758422752927,
  'p-value': 0.095952023988006,
  'p-value threshold': 0.005197401299350267},
  ...
}
```

## References

Our KSDAgg code is based our MMDAgg implementation which can be found at [https://github.com/antoninschrab/mmdagg-paper](https://github.com/antoninschrab/mmdagg-paper).

For the Gaussian-Bernoulli Restricted Boltzmann Machine experiment, we obtain the samples and scores in [generate_data_rbm.py](generate_data_rbm.py) by relying on [Wittawat Jitkrittum](https://github.com/wittawatj)'s implementation which can be found at [https://github.com/wittawatj/kernel-gof](https://github.com/wittawatj/kernel-gof) under the MIT License. The relevant files we use are in the directory [kgof](kgof).

For the MNIST Normalizing Flow experiment, we use in [generate_data_nf.py](generate_data_nf.py) a multiscale Normalizing Flow trained on the MNIST dataset as implemented by [Phillip Lippe](https://github.com/phlippe) in [Tutorial 11: Normalizing Flows for image modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html)
as part of the [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html)
under the MIT License.

## KSDAggInc

For a computationally efficient version of KSDAgg which can run in linear time, check out our paper [Efficient Aggregated Kernel Tests using Incomplete U-statistics](https://arxiv.org/pdf/2206.09194.pdf) and its implementation available on the [agginc-paper](https://github.com/antoninschrab/agginc-paper) repository.

## Author

[Antonin Schrab](https://antoninschrab.github.io)

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@inproceedings{
  schrab2022ksd,
  title={{KSD} Aggregated Goodness-of-fit Test},
  author={Antonin Schrab and Benjamin Guedj and Arthur Gretton},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=9-SZkJLkCcB}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md))
