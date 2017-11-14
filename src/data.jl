# data structures for models and samples


type GMLSample
    count::Int
    assignment::Array{Int,1}
    value::Nullable{Real}
end
GMLSample(count::Int, assignment::Array{Int,1}) = GMLSample(count, assignment, Nullable{Real}())

function Base.show(io::IO, s::GMLSample)
    if isnull(s.value)
        print(io, s.count, " ", s.assignment)
    else
        print(io, s.count, " ", s.assignment, " ", get(s.value))
    end
end


alphabets = [:spin]

type GMLSamples
    varible_count::Int
    alphabet::Symbol
    samples::Array{GMLSample,1}
    variable_names::Nullable{Array{String,1}}
    GMLSamples(a,b,c,d) = check_samples_data(a,b,c,d) ? new(a,b,c,d) : error("generic init problem")
end
GMLSamples(varible_count::Int, alphabet::Symbol, samples::Array{GMLSample,1}) = GMLSamples(varible_count, alphabet, samples, Nullable{Array{String,1}}())

function check_samples_data(varible_count::Int, alphabet::Symbol, samples::Array{GMLSample,1}, variable_names::Nullable{Array{String,1}})
    if !in(alphabet, alphabets)
        error("alphabet $(alphabet) is not supported")
        return false 
    end
    for sample in samples
        if length(sample.assignment) != varible_count
            error("a sample has $(length(sample.assignment)) values but should have $(varible_count) values")
            return false
        end
        # TODO check for correct alphabet
    end
    return true
end

function Base.show(io::IO, s::GMLSamples)
    println(io, s.varible_count, " ", s.alphabet)
    if !isnull(s.variable_names)
        println(io, get(s.variable_names))
    end
    for sample in s.samples
        println(sample)
    end
end
