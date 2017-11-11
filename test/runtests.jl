using GraphicalModelLearning

using Base.Test

# fix random number seed for testing
srand(0)

GML = GraphicalModelLearning

gms = Dict(
    "a" => [
        0.0 0.1 0.2;
        0.1 0.0 0.3;
        0.2 0.3 0.0
    ],
    "b" => [
        0.3 0.1 0.2;
        0.1 0.2 0.3;
        0.2 0.3 0.1
    ],
    "c" => [
        0.0 0.1 0.2 0.3;
        0.1 0.0 0.2 0.3;
        0.2 0.2 0.0 0.3;
        0.3 0.3 0.3 0.0
    ]
)

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

@testset "learned model accuracy" begin
    for act in accuracy_tests
        test_name = eval("$(act.formulation) $(act.samples) $(act.threshold)")
        @testset "  $(act.formulation) $(act.samples) $(act.threshold)" begin
            for (gm_name, gm) in gms
                sample_histo = GML.gibbs_sampler(gm, act.samples)
                learned_gm = GML.inverse_ising(sample_histo, method=act.formulation)
                max_error = maximum(abs(gm - learned_gm))
                @test max_error <= act.threshold
            end
        end
    end
end
