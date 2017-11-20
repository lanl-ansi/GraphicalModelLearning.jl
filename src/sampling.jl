export sample

export GMSampler, Gibbs

using StatsBase

@compat abstract type GMSampler end

type Gibbs <: GMSampler end

function int_to_spin(int_representation, spin_number)
    spin = 2*digits(int_representation, 2, spin_number)-1
    return spin
end

function weigh_proba{T <: Real}(int_representation, gm::FactorGraph{T}, dkeys::Vector{Tuple})
    current_spin = int_to_spin(int_representation, gm.varible_count)

    base = sum(Float64[weight * current_spin[i] * current_spin[j] for ((i,j), weight) in gm])
    feild = sum(Float64[gm[(i,j)] * current_spin[i] for (i,j) in dkeys])

    return exp(base + feild)
end

function sample_generation{T <: Real}(gm::FactorGraph{T}, samples_per_bin::Integer, bins::Int)
    @assert bins >= 1

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    dkeys = diag_keys(gm)

    items   = [i for i in 0:(config_number-1)]
    weights = [weigh_proba(i, gm, dkeys) for i in (0:config_number-1)]

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
    if gm.order != 2
        error("sampling is only supported for FactorGraphs of order 2, given order $(gm.order)")
    end
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs, given alphabet $(gm.alphabet)")
    end

    # generation of samples
    samples = sample_generation(gm, number_sample, replicates)

    return samples
end
