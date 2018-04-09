# GraphicalModelLearning.jl
Algorithms for Learning Graphical Models

## Installation
In julia run, 

`Pkg.clone("git@github.com:lanl-ansi/GraphicalModelLearning.jl.git")`


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

If you find GraphicalModelLearning useful in your work, we kindly request that you cite the following publication:
```
@article{Lokhove1700791,
    author = {Lokhov, Andrey Y. and Vuffray, Marc and Misra, Sidhant and Chertkov, Michael},
    title = {Optimal structure and parameter learning of Ising models},
    journal = {Science Advances}
    volume = {4}, number = {3}, year = {2018},
    publisher = {American Association for the Advancement of Science},
    doi = {10.1126/sciadv.1700791},
    URL = {http://advances.sciencemag.org/content/4/3/e1700791}
}
```


## License

This code is provided under a BSD license as part of the Optimization, Inference and Learning for Advanced Networks project, C18014.
