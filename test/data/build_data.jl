#!/usr/bin/env julia

# updates ground truth data for testing

using GraphicalModelLearning

## remove old files
#`rm -rf *.csv`

using Random

include("../common.jl")

for (gm_name, model) in gms
    Random.seed!(0) # fix random number generator
    samples = sample(model, gibbs_test_samples)
    writecsv("$(gm_name)_samples.csv", samples)
end

for (gm_name, model) in gms
    samples = readdlm("$(gm_name)_samples.csv", ",")
    for (form_name, formulation) in formulations
        Random.seed!(0) # fix random number generator
        learned_gm = learn(samples, formulation)
        writecsv("$(gm_name)_$(form_name)_learned.csv", learned_gm)
    end
end
