# GraphicalModelLearning.jl
Algorithms for Learning Graphical Models

### Installation
In julia run, 

`Pkg.clone("git@github.com:lanl-ansi/GraphicalModelLearning.jl.git")` or 
`Pkg.clone("https://github.com/lanl-ansi/GraphicalModelLearning.jl.git")`.

### Quick Start
Try the following commands in julia,

```
using GraphicalModelLearning

model = FactorGraph([0.0 0.1 0.2; 0.1 0.0 0.3; 0.2 0.3 0.0])
samples = sample(model, 100000)
learned = learn(samples)

err = abs.(convert(Array{Float64,2}, model) - learned)
```

Note that the first invocation of `learn` will be slow as the dependent libraries are compiled.  Subsequent calls will be fast.
