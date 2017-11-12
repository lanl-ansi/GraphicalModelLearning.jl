using GraphicalModelLearning

using Base.Test

include("common.jl")


@testset "gibbs sampler" begin
    for (name, gm) in gms
        srand(0) # fix random number generator
        samples = sample(gm, gibbs_test_samples)
        base_samples = readcsv("data/$(name)_samples.csv")
        @test isapprox(samples, base_samples)
    end
end

#=
@testset "inverse ising formulations" begin
    for (form_name, formulation) in formulations
        @testset "  $(form_name)" begin
            for (name, gm) in gms
                samples = readcsv("data/$(name)_samples.csv")
                srand(0) # fix random number generator
                #learned_gm = inverse_ising(samples, method=formulation)
                learned_gm = learn(samples, formulation)
                base_learned_gm = readcsv("data/$(name)_$(form_name)_learned.csv")
                println(maximum(abs.(learned_gm - base_learned_gm)))
                @test isapprox(learned_gm, base_learned_gm)
            end
        end
    end
end
=#

type AccuracyTest
   formulation::GMLFormulation
   samples::Int
   threshold::Float64
end

accuracy_tests = [
    AccuracyTest(RISE(),     1000, 0.15)
    AccuracyTest(logRISE(),  1000, 0.15)
    AccuracyTest(RPLE(),     1000, 0.15)
    AccuracyTest(RISE(),    10000, 0.05)
    AccuracyTest(logRISE(), 10000, 0.05)
    AccuracyTest(RPLE(),    10000, 0.05)
]

srand(0) # fix random number generator
@testset "learned model accuracy" begin
    for act in accuracy_tests
        @testset "  $(act.formulation) $(act.samples) $(act.threshold)" begin
            for (gm_name, gm) in gms
                sample_histo = sample(gm, act.samples)
                #learned_gm = inverse_ising(sample_histo, method=act.formulation)
                learned_gm = learn(sample_histo, act.formulation)
                max_error = maximum(abs.(gm - learned_gm))
                @test max_error <= act.threshold
            end
        end
    end
end
