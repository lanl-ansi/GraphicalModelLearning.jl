# GraphicalModelLearning.jl

Dev:
[![CI](https://github.com/lanl-ansi/GraphicalModelLearning.jl/workflows/CI/badge.svg)](https://github.com/lanl-ansi/GraphicalModelLearning.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lanl-ansi/GraphicalModelLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lanl-ansi/GraphicalModelLearning.jl)
[![Documentation](https://github.com/lanl-ansi/GraphicalModelLearning.jl/workflows/Documentation/badge.svg)](https://lanl-ansi.github.io/GraphicalModelLearning.jl/stable/)



Algorithms for Learning Graphical Models


## Installation
The Julia package manager can be used to install GraphicalModelLearning as follows, 
```
] add GraphicalModelLearning
```

## Quick Start
Try the following commands in julia,

```
using GraphicalModelLearning

model = FactorGraph([0.0 0.1 0.2; 0.1 0.0 0.3; 0.2 0.3 0.0])
samples = sample(model, 100000)
learned = learn(samples)

err = abs.(convert(Array{Float64,2}, model) - learned)
```

Note that the first invocation of `learn` will be slow as the dependent libraries are compiled.  Subsequent calls will be fast.


## Reference

If you find GraphicalModelLearning useful in your work, we kindly request that you cite the following publications:
```
@misc{1902.00600,
    author = {Marc Vuffray and Sidhant Misra and Andrey Y. Lokhov},
    title = {Efficient Learning of Discrete Graphical Models},
    year = {2019},
    eprint = {arXiv:1902.00600},
    url = {https://arxiv.org/abs/1902.00600}
}
```
```
@article{Lokhove1700791,
    author = {Lokhov, Andrey Y. and Vuffray, Marc and Misra, Sidhant and Chertkov, Michael},
    title = {Optimal structure and parameter learning of Ising models},
    journal = {Science Advances}
    volume = {4}, number = {3}, year = {2018},
    publisher = {American Association for the Advancement of Science},
    doi = {10.1126/sciadv.1700791},
    url = {http://advances.sciencemag.org/content/4/3/e1700791}
}
```
```
@incollection{NIPS2016_6375,
    author = {Vuffray, Marc and Misra, Sidhant and Lokhov, Andrey Y. and Chertkov, Michael},
    title = {Interaction Screening: Efficient and Sample-Optimal Learning of Ising Models},
    booktitle = {Advances in Neural Information Processing Systems 29},
    year = {2016}, pages = {2595--2603},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/6375-interaction-screening-efficient-and-sample-optimal-learning-of-ising-models.pdf}
}
```


## License

This code is provided under a BSD license as part of the Optimization, Inference and Learning for Advanced Networks project, C18014.
