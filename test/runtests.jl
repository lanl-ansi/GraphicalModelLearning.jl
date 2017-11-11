using GraphicalModelLearning

gms = Dict(
    "a" => [
    0, 1, 2;
    1, 0, 3;
    2, 3, 0;
    ],
    "b" => [
    3, 1, 2;
    1, 2, 3;
    2, 3, 1;
    ],
)

println(gms)
