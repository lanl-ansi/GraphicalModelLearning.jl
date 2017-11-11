#!/usr/bin/env julia

# updates ground truth data for testing

using GraphicalModelLearning

# remove old files
`rm -rf *.csv`

include("../common.jl")

for (name, model) in gms
    srand(0) # fix random number generator
    samples = gibbs_sampler(model, gibbs_test_samples)
    writecsv("$(name)_samples.csv", samples)
end

for (name, model) in gms
    samples = readcsv("$(name)_samples.csv")
    for formulation in formulations
        srand(0) # fix random number generator
        learned_gm = inverse_ising(samples, method=formulation)
        writecsv("$(name)_$(formulation)_learned.csv", learned_gm)
    end
end
