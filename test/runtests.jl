using GraphicalModelLearning
using Ipopt
using Base.Test
using Logging

# suppress info and warnings during testing
Logging.configure(level=ERROR)

include("common.jl")


@testset "gibbs sampler" begin
    for (name, gm) in gms
        srand(0) # fix random number generator
        samples = sample(gm, gibbs_test_samples)
        samples_tbl = convert(Array{Int,2}, samples)
        base_samples = readcsv("data/$(name)_samples.csv")
        @test isapprox(samples_tbl, base_samples)
    end
end



@testset "inverse ising formulations" begin

    for (form_name, formulation) in formulations
        @testset "  $(form_name)" begin
            for (name, gm) in gms
                samples = readcsv("data/$(name)_samples.csv", Int)
                srand(0) # fix random number generator
                #learned_gm = inverse_ising(samples, method=formulation)
                learned_gm = learn(samples, formulation)
                base_learned_gm = readcsv("data/$(name)_$(form_name)_learned.csv")
                #println(maximum(abs.(learned_gm - base_learned_gm)))
                @test isapprox(learned_gm, base_learned_gm)
            end
        end
    end


    samples = readcsv("data/mvt_samples.csv", Int)

    rand(0) # fix random number generator
    learned_gm_rise = learn(samples, RISE(0.2, false), NLP(IpoptSolver(print_level=0)))
    base_learned_gm = readcsv("data/mvt_RISE_learned.csv")
    #println(abs.(learned_gm_rise - base_learned_gm))
    @test isapprox(learned_gm_rise, base_learned_gm)

    srand(0) # fix random number generator
    learned_gm_logrise = learn(samples, logRISE(0.2, false), NLP(IpoptSolver(print_level=0)))
    base_learned_gm = readcsv("data/mvt_logRISE_learned.csv")
    #println(abs.(learned_gm_logrise - base_learned_gm))
    @test isapprox(learned_gm_logrise, base_learned_gm)

    srand(0) # fix random number generator
    learned_gm_rple = learn(samples, RPLE(0.2, false), NLP(IpoptSolver(print_level=0)))
    base_learned_gm = readcsv("data/mvt_RPLE_learned.csv")
    #println(abs.(learned_gm_rple - base_learned_gm))
    @test isapprox(learned_gm_rple, base_learned_gm)
end



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
                samples = sample(gm, act.samples)
                #learned_gm = inverse_ising(samples, method=act.formulation)
                learned_gm = learn(samples, act.formulation)
                max_error = maximum(abs.(gm - learned_gm))
                @test max_error <= act.threshold
            end
        end
    end
end



srand(0) # fix random number generator
@testset "docs example" begin
    model = [0.0 0.1 0.2; 0.1 0.0 0.3; 0.2 0.3 0.0]
    samples = sample(model, 100000)
    learned = learn(samples)

    err = abs.(model - learned)
    @test maximum(err) <= 0.01
end
