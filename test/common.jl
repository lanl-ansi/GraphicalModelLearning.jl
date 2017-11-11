
# if any of these data are modified make sure to run data/build_data.jl

# gibbs_test_name = "$(name)_samples.csv"
gibbs_test_samples = 1000

# formulation_test_name = "$(name)_$(formulation)_learned.csv"
formulations = [:RISE, :logRISE, :RPLE]

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
