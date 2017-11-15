#!/usr/bin/env julia

# updates ground truth data for testing

using GraphicalModelLearning

# remove old files
`rm -rf *.csv`

include("../common.jl")

for (gm_name, model) in gms
    srand(0) # fix random number generator
    samples = sample(model, gibbs_test_samples)
    writecsv("$(gm_name)_samples.csv", convert(Array{Int,2}, samples))
end

for (gm_name, model) in gms
    samples = readcsv("$(gm_name)_samples.csv", Int)
    for (form_name, formulation) in formulations
        srand(0) # fix random number generator
        learned_gm = learn(samples, formulation)
        writecsv("$(gm_name)_$(form_name)_learned.csv", learned_gm)
    end
end
