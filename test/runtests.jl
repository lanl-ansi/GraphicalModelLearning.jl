using GraphicalModelLearning

using Base.Test

include("common.jl")


@testset "gibbs sampler" begin
    for (name, gm) in gms
        srand(0) # fix random number generator
        samples = gibbs_sampler(gm, gibbs_test_samples)
        base_samples = readcsv("data/$(name)_samples.csv")
        @test samples == base_samples
    end
end


@testset "inverse ising formulations" begin
    for formulation in formulations
        @testset "  $(formulation)" begin
            for (name, gm) in gms
                samples = readcsv("data/$(name)_samples.csv")
                srand(0) # fix random number generator
                learned_gm = inverse_ising(samples, method=formulation)
                base_learned_gm = readcsv("data/$(name)_$(formulation)_learned.csv")
                @test learned_gm == base_learned_gm
            end
        end
    end
end


type AccuracyTest
   formulation::Symbol
   samples::Int
   threshold::Float64
end

accuracy_tests = [
    AccuracyTest(:RISE,     1000, 0.15)
    AccuracyTest(:logRISE,  1000, 0.15)
    AccuracyTest(:RPLE,     1000, 0.15)
    AccuracyTest(:RISE,    10000, 0.05)
    AccuracyTest(:logRISE, 10000, 0.05)
    AccuracyTest(:RPLE,    10000, 0.05)
]

srand(0) # fix random number generator
@testset "learned model accuracy" begin
    for act in accuracy_tests
        test_name = eval("$(act.formulation) $(act.samples) $(act.threshold)")
        @testset "  $(act.formulation) $(act.samples) $(act.threshold)" begin
            for (gm_name, gm) in gms
                sample_histo = gibbs_sampler(gm, act.samples)
                learned_gm = inverse_ising(sample_histo, method=act.formulation)
                max_error = maximum(abs(gm - learned_gm))
                @test max_error <= act.threshold
            end
        end
    end
end
