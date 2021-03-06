# Code for KSDAgg: a KSD aggregated goodness-of-fit test

This GitHub repository contains the code for the reproducible experiments presented in our paper 
[KSD Aggregated Goodness-of-fit Test](https://arxiv.org/pdf/2202.00824.pdf):
- Gamma distribution experiment,
- Gaussian-Bernoulli Restricted Boltzmann Machine experiment,
- MNIST Normalizing Flow experiment.

We provide the code to run the experiments to generate Figures 1-4 and Table 1 from our paper, 
those can be found in [figures](figures). 

Our aggregated test [KSDAgg](https://arxiv.org/pdf/2202.00824.pdf#page=4) is implemented in [ksdagg.py](ksdagg.py).
We provide code for two quantile estimation methods: the wild bootstrap and the parametric bootstrap.
Our implementation uses the IMQ (inverse multiquadric) kernel with a collection of bandwidths consisting of 
the median bandwidth scaled by powers of 2, and with one of the four types of weights proposed in 
[MMD Aggregated Two-Sample Test](https://arxiv.org/pdf/2110.15073.pdf#page=22).
We also provide `custom` KSDAgg functions in [ksdagg.py](ksdagg.py) which allow for the use of any kernel collections and weights.

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
The required packages can then be installed in the virtual environment by running
```
python -m pip install -r requirements.txt
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
Note that our expeiments are comprised of 'embarrassingly parallel for loops', for which significant speed up can be obtained by using 
parallel computing libraries such as `joblib` or `dask`.

The actual figures of the paper can be obtained from the saved dataframes in [results](results) by using the command 
```
python figures.py  
```
The figures are saved in the directory [figures](figures) and correspond to the ones used in our [paper](https://arxiv.org/pdf/2202.00824.pdf).

## References

Our KSDAgg code is based our MMDAgg implementation which can be found at [https://github.com/antoninschrab/mmdagg-paper](https://github.com/antoninschrab/mmdagg-paper).

For the Gaussian-Bernoulli Restricted Boltzmann Machine experiment, we obtain the samples and scores in [generate_data_rbm.py](generate_data_rbm.py) by relying on [Wittawat Jitkrittum](https://github.com/wittawatj)'s implementation which can be found at [https://github.com/wittawatj/kernel-gof](https://github.com/wittawatj/kernel-gof) under the MIT License. The relevant files we use are in the directory [kgof](kgof).

For the MNIST Normalizing Flow experiment, we use in [generate_data_nf.py](generate_data_nf.py) a multiscale Normalizing Flow trained on the MNIST dataset as implemented by [Phillip Lippe](https://github.com/phlippe) in [Tutorial 11: Normalizing Flows for image modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html)
as part of the [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html)
under the MIT License.

## Author

[Antonin Schrab](https://antoninschrab.github.io)

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria, Lille - Nord Europe research centre and Inria London Programme

## Bibtex

```
@unpublished{schrab2022ksd,
    title={{KSD} Aggregated Goodness-of-fit Test},
    author={Antonin Schrab and Benjamin Guedj and Arthur Gretton},
    year={2022},
    note = "Submitted.",
    abstract = {We investigate properties of goodness-of-fit tests based on the Kernel Stein Discrepancy (KSD). We introduce a strategy to construct a test, called KSDAgg, which aggregates multiple tests with different kernels. KSDAgg avoids splitting the data to perform kernel selection (which leads to a loss in test power), and rather maximises the test power over a collection of kernels. We provide theoretical guarantees on the power of KSDAgg: we show it achieves the smallest uniform separation rate of the collection, up to a logarithmic term. KSDAgg can be computed exactly in practice as it relies either on a parametric bootstrap or on a wild bootstrap to estimate the quantiles and the level corrections. In particular, for the crucial choice of bandwidth of a fixed kernel, it avoids resorting to arbitrary heuristics (such as median or standard deviation) or to data splitting. We find on both synthetic and real-world data that KSDAgg outperforms other state-of-the-art adaptive KSD-based goodness-of-fit testing procedures.},
    url = {https://arxiv.org/abs/2202.00824},
    url_PDF = {https://arxiv.org/pdf/2202.00824.pdf},
    url_Code = {https://github.com/antoninschrab/ksdagg-paper},
    eprint={2202.00824},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md))
