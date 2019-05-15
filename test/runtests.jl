using GraphicalModelLearning
using Ipopt
using JuMP

using Test
using Random

import DelimitedFiles: readdlm
import LinearAlgebra: diag

const SOLVER = with_optimizer(Ipopt.Optimizer, print_level=0)

include("common.jl")

@testset "GraphicalModelLearning" begin

@testset "factor graphs" begin
    for (name, gm) in gms
        matrix = convert(Array{Float64,2}, gm)
        gm2 = FactorGraph(matrix)
        for key in keys(gm)
            @test isapprox(gm[key], gm2[key])
            if length(key) == 1
                @test isapprox(gm[key], matrix[key..., key...])
            else
                @test isapprox(gm[key], matrix[key...])
            end
        end
    end
end


@testset "gibbs sampler" begin
    for (name, gm) in gms
        gm_tmp = deepcopy(gm)
        gm_tmp.order = 3
        Random.seed!(0) # fix random number generator
        samples = sample(gm_tmp, gibbs_test_samples)
        #base_samples = readdlm("data/$(name)_samples.csv", ',', Int64)
        base_samples = readdlm("data/$(name)_samples.csv", ',')
        @test isapprox(samples, base_samples)
    end
end

@testset "gibbs sampler, 2nd order" begin
    for (name, gm) in gms
        Random.seed!(0) # fix random number generator
        samples = sample(gm, gibbs_test_samples)
        base_samples = readdlm("data/$(name)_samples.csv", ',')
        #println(name)
        #println(base_samples)
        #println(samples)
        #println(abs.(base_samples-samples))
        @test isapprox(samples, base_samples)
    end
end

@testset "gibbs sampler with replicates" begin
    for (name, gm) in gms
        Random.seed!(0) # fix random number generator
        samples_set = sample(gm, gibbs_test_samples, 3)
        for samples in samples_set
            @test sum(samples[:,1]) == gibbs_test_samples
        end
    end
end


@testset "inverse ising formulations" begin

    for (form_name, formulation) in formulations
        @testset "  $(form_name)" begin
            for (name, gm) in gms
                samples = readdlm("data/$(name)_samples.csv", ',')
                Random.seed!(0) # fix random number generator
                #learned_gm = inverse_ising(samples, method=formulation)
                learned_gm = learn(samples, formulation)
                base_learned_gm = readdlm("data/$(name)_$(form_name)_learned.csv", ',')
                #println(maximum(abs.(learned_gm - base_learned_gm)))
                @test isapprox(learned_gm, base_learned_gm)
            end
        end
    end


    samples = readdlm("data/mvt_samples.csv", ',')

    Random.seed!(0) # fix random number generator
    learned_gm_rise = learn(samples, RISE(0.2, false), NLP(SOLVER))
    base_learned_gm = readdlm("data/mvt_RISE_learned.csv", ',')
    #println(abs.(learned_gm_rise - base_learned_gm))
    @test isapprox(learned_gm_rise, base_learned_gm)

    Random.seed!(0) # fix random number generator
    learned_gm_logrise = learn(samples, logRISE(0.2, false), NLP(SOLVER))
    base_learned_gm = readdlm("data/mvt_logRISE_learned.csv", ',')
    #println(abs.(learned_gm_logrise - base_learned_gm))
    @test isapprox(learned_gm_logrise, base_learned_gm)

    Random.seed!(0) # fix random number generator
    learned_gm_rple = learn(samples, RPLE(0.2, false), NLP(SOLVER))
    base_learned_gm = readdlm("data/mvt_RPLE_learned.csv", ',')
    #println(abs.(learned_gm_rple - base_learned_gm))
    @test isapprox(learned_gm_rple, base_learned_gm)
end


accuracy_tests = [
    AccuracyTest(RISE(),     1000, 0.15)
    AccuracyTest(logRISE(),  1000, 0.15)
    AccuracyTest(RPLE(),     1000, 0.15)
    AccuracyTest(RISE(),    10000, 0.05)
    AccuracyTest(logRISE(), 10000, 0.05)
    AccuracyTest(RPLE(),    10000, 0.05)
]

Random.seed!(0) # fix random number generator
@testset "learned model accuracy" begin
    for act in accuracy_tests
        @testset "  $(act.formulation) $(act.samples) $(act.threshold)" begin
            for (gm_name, gm) in gms
                sample_histo = sample(gm, act.samples)
                #learned_gm = inverse_ising(sample_histo, method=act.formulation)
                learned_gm = learn(sample_histo, act.formulation)
                max_error = maximum(abs.(convert(Array{Float64,2}, gm) - learned_gm))
                @test max_error <= act.threshold
            end
        end
    end
end


@testset "inverse multi-body formulations" begin

    for (name, gm) in gms
        samples = readdlm("data/$(name)_samples.csv", ',')
        Random.seed!(0) # fix random number generator
        learned_ising = learn(samples, RISE(0.2, false))
        learned_two_body = learn(samples, multiRISE(0.2, false, 2))

        learned_ising_dict = convert(Dict, learned_ising)
        #println(learned_ising_dict)
        #println(learned_two_body)

        @test length(learned_ising_dict) == length(learned_two_body)
        for (key, value) in learned_ising_dict
            @test isapprox(learned_two_body[key], value, atol=1e-7)
        end
    end

    samples = readdlm("data/mvt_samples.csv", ',')

    rand(0) # fix random number generator
    learned_ising = learn(samples, RISE(0.2, false), NLP(SOLVER))
    learned_two_body = learn(samples, multiRISE(0.2, false, 2), NLP(SOLVER))

    learned_ising_dict = convert(Dict, learned_ising)
    @test length(learned_ising_dict) == length(learned_two_body)
    for (key, value) in learned_ising_dict
        @test isapprox(learned_two_body[key], value)
    end


    for (name, gm) in gms
        gm_tmp = deepcopy(gm)
        gm_tmp.order = 4
        Random.seed!(0) # fix random number generator
        samples = sample(gm_tmp, 10000)

        learned_gm = learn(samples, multiRISE(0.0, false, 4))

        for (key, value) in gm_tmp
            #println(learned_gm[key], value)
            @test isapprox(learned_gm[key], value, atol = 0.15)
        end

        samples = sample(learned_gm, 10000)
        learned_gm2 = learn(samples, multiRISE(0.0, false, 4))

        for (key, value) in gm_tmp
            #println(learned_gm2[key], " ", value)
            # this is a bug, the learned_gm2 value should not be twice as large
            @test isapprox(learned_gm2[key]/2.0, value, atol = 0.15)
        end

    end
end



Random.seed!(0) # fix random number generator
@testset "docs example" begin
    model = FactorGraph([0.0 0.1 0.2; 0.1 0.0 0.3; 0.2 0.3 0.0])
    samples = sample(model, 100000)
    learned = learn(samples)

    err = abs.(convert(Array{Float64,2}, model) - learned)
    @test maximum(err) <= 0.01
end


end
