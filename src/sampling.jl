export sample

export GMSampler, Gibbs

using StatsBase

@compat abstract type GMSampler end

type Gibbs <: GMSampler end

function int_to_spin(int_representation::Int, spin_number::Int)
    spin = 2*digits(int_representation, 2, spin_number)-1
    return spin
end


function weigh_proba{T <: Real}(int_representation::Int, adj::Array{T,2}, prior::Array{T,1})
    spin_number = size(adj,1)
    spins = int_to_spin(int_representation, spin_number)
    return exp(((0.5) * spins' * adj * spins + prior' * spins)[1])
end


bool_to_spin(bool::Int) = 2*bool-1

function weigh_proba{T <: Real}(int_representation::Int, adj::Array{T,2}, prior::Array{T,1}, spins::Array{Int,1})
    digits!(spins, int_representation, 2)
    spins .= bool_to_spin.(spins)
    return exp((0.5 * spins' * adj * spins + prior' * spins)[1])
end


# assumes second order
function sample_generation_ising{T <: Real}(gm::FactorGraph{T}, samples_per_bin::Integer, bins::Int)
    @assert bins >= 1

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]

    items   = [i for i in 0:(config_number-1)]
    assignment_tmp = [0 for i in 1:spin_number] # pre allocate assignment memory
    weights = [weigh_proba(i, adjacency_matrix, prior_vector, assignment_tmp) for i in (0:config_number-1)]

    raw_sample = StatsBase.sample(items, StatsBase.Weights(weights), samples_per_bin*bins, ordered=false)
    raw_sample_bins = reshape(raw_sample, bins, samples_per_bin)

    spin_samples = []
    for b in 1:bins
        raw_binning = countmap(raw_sample_bins[b,:])
        spin_sample = [ vcat(raw_binning[i], int_to_spin(i, spin_number)) for i in keys(raw_binning)]
        push!(spin_samples, hcat(spin_sample...)')
    end
    return spin_samples
end


function weigh_proba{T <: Real}(int_representation::Int, gm::FactorGraph{T}, spins::Array{Int,1})
    digits!(spins, int_representation, 2)
    spins .= bool_to_spin.(spins)
    evaluation = sum( weight*prod(spins[i] for i in term) for (term, weight) in gm) 
    return exp(evaluation)
end

function sample_generation{T <: Real}(gm::FactorGraph{T}, samples_per_bin::Integer, bins::Int)
    @assert bins >= 1
    info("use general sample model")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    items   = [i for i in 0:(config_number-1)]
    assignment_tmp = [0 for i in 1:spin_number] # pre allocate assignment memory
    weights = [weigh_proba(i, gm, assignment_tmp) for i in (0:config_number-1)]

    raw_sample = StatsBase.sample(items, StatsBase.Weights(weights), samples_per_bin*bins, ordered=false)
    raw_sample_bins = reshape(raw_sample, bins, samples_per_bin)

    spin_samples = []
    for b in 1:bins
        raw_binning = countmap(raw_sample_bins[b,:])
        spin_sample = [ vcat(raw_binning[i], int_to_spin(i, spin_number)) for i in keys(raw_binning)]
        push!(spin_samples, hcat(spin_sample...)')
    end
    return spin_samples
end

sample{T <: Real}(gm::FactorGraph{T}, number_sample::Integer) = sample(gm, number_sample, 1, Gibbs())[1]
sample{T <: Real}(gm::FactorGraph{T}, number_sample::Integer, replicates::Integer) = sample(gm, number_sample, replicates, Gibbs())


function sample{T <: Real}(gm::FactorGraph{T}, number_sample::Integer, replicates::Integer, sampler::Gibbs)
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs, given alphabet $(gm.alphabet)")
    end

    if gm.order <= 2
        samples = sample_generation_ising(gm, number_sample, replicates)
    else
        samples = sample_generation(gm, number_sample, replicates)
    end

    return samples
end
