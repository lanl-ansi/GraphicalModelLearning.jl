
# if any of these data are modified make sure to run data/build_data.jl

# gibbs_test_name = "$(name)_samples.csv"
gibbs_test_samples = 1000000

# formulation_test_name = "$(name)_$(formulation)_learned.csv"
#formulations = [:RISE, :logRISE, :RPLE]
formulations = Dict(
    "RISE" => RISE(), 
    "logRISE" => logRISE(), 
    "RPLE" => RPLE()
)

gms = Dict(
    "a" => FactorGraph([
        0.0 0.1 0.2;
        0.1 0.0 0.3;
        0.2 0.3 0.0
    ]),
    "b" => FactorGraph([
        0.3 0.1 0.2;
        0.1 0.2 0.3;
        0.2 0.3 0.1
    ]),
    "c" => FactorGraph([
        0.0 0.1 0.2 0.3;
        0.1 0.0 0.2 0.3;
        0.2 0.2 0.0 0.3;
        0.3 0.3 0.3 0.0
    ])
)

mutable struct AccuracyTest
   formulation::GMLFormulation
   samples::Int
   threshold::Float64
end
