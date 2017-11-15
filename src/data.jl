# data structures for models and samples

export GMLSample, GMLSamples

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


alphabets = [:spin, :boolean, :integer, :integer_pos, :real, :real_pos]

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

Base.getindex(samples::GMLSamples, i) = samples.samples[i]
Base.getindex(samples::GMLSamples, i, j) = samples.samples[i].assignment[j]

Base.start(samples::GMLSamples) = start(samples.samples)
Base.next(samples::GMLSamples, state) = next(samples.samples, state)
Base.done(samples::GMLSamples, state) = done(samples.samples, state)

Base.length(samples::GMLSamples) = length(samples.samples)

function Base.show(io::IO, s::GMLSamples)
    println(io, "vars: ", s.varible_count)
    println(io, "alphabet: ", s.alphabet)
    if !isnull(s.variable_names)
        println(io, "variable names: ")
        println(io, get(s.variable_names))
    end

    println(io, "samples: ")
    for sample in s.samples
        println(sample)
    end
end

function Base.convert{T <: Int}(::Type{GMLSamples}, m::Array{T,2})
    alphabet = :integer
    varible_count = size(m,2) - 1

    values = Set{Int}()
    samples::Array{GMLSample,1} = []
    for r in 1:size(m, 1)
        row = m[r,:]
        push!(samples, GMLSample(row[1], row[2:end]))
        push!(values, row[2:end]...)
    end

    if length(values) == 2
        if -1 in values && 1 in values
            info("detected spin alphabet")
            alphabet = :spin
        elseif 0 in values && 1 in values
            info("detected spin alphabet")
            alphabet = :boolean
        end
    else
        if prod([v >= 1 for v in values]) == 1
            info("detected integer_pos alphabet")
            alphabet = :integer_pos
        end
    end

    return GMLSamples(varible_count, alphabet, samples)
end

function Base.convert{T <: Int}(::Type{Array{T,2}}, s::GraphicalModelLearning.GMLSamples)
    assigments = [s[i,j] for i in 1:length(s), j in 1:s.varible_count]
    counts = [sample.count for sample in s.samples]
    return hcat(counts, assigments)
end

