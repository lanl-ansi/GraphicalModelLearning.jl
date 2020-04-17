export sample

export GMSampler, Gibbs, Glauber

export gibbsMCsampler

using StatsBase
using LinearAlgebra

abstract type GMSampler end

struct Gibbs <: GMSampler end

mutable struct Glauber <: GMSampler
    initial_state::Array{Int8,1}
    initial_steps::Int64
    spacing_steps::Int64
end

function int_to_spin(int_representation::Int, spin_number::Int)
    spin = 2*digits(int_representation, base=2, pad=spin_number) .- 1
    return spin
end


function weigh_proba(int_representation::Int, adj::Array{T,2}, prior::Array{T,1}) where T <: Real
    spin_number = size(adj,1)
    spins = int_to_spin(int_representation, spin_number)
    return exp(((0.5) * spins' * adj * spins + prior' * spins)[1])
end


bool_to_spin(bool::Int) = 2*bool-1

function weigh_proba(int_representation::Int, adj::Array{T,2}, prior::Array{T,1}, spins::Array{Int,1}) where T <: Real
    digits!(spins, int_representation, base=2)
    spins .= bool_to_spin.(spins)
    return exp((0.5 * spins' * adj * spins + prior' * spins)[1])
end


# assumes second order
function sample_generation_ising(gm::FactorGraph{T}, samples_per_bin::Integer, bins::Int) where T <: Real
    @assert bins >= 1

    spin_number   = gm.variable_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector = copy(transpose(diag(adjacency_matrix)))[1,:]

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


function weigh_proba(int_representation::Int, gm::FactorGraph{T}, spins::Array{Int,1}) where T <: Real
    digits!(spins, int_representation, base=2)
    spins .= bool_to_spin.(spins)
    evaluation = sum( weight*prod(spins[i] for i in term) for (term, weight) in gm)
    return exp(evaluation)
end

function sample_generation(gm::FactorGraph{T}, samples_per_bin::Integer, bins::Int; sample_increment::Int64=1000000) where T <: Real
    @assert bins >= 1
    #info("use general sample model")

    spin_number   = gm.variable_count
    config_number = 2^spin_number

    items   = [i for i in 0:(config_number-1)]
    assignment_tmp = [0 for i in 1:spin_number] # pre allocate assignment memory
    weights = [weigh_proba(i, gm, assignment_tmp) for i in (0:config_number-1)]

    # To reduce memory overhead for models requiring billions of samples, we break the sampling
    # into multiple steps and convert to a histogram representation after each

    num_samples = 0
    raw_binnings = [Dict{Int64, Int64}() for i in 1:bins]
    while num_samples < samples_per_bin
        samples_step = min(sample_increment, samples_per_bin - num_samples)

        raw_sample = StatsBase.sample(items, StatsBase.Weights(weights), samples_step*bins, ordered=false)
        raw_sample_bins = reshape(raw_sample, bins, samples_step)

        for b in 1:bins
            addcounts!(raw_binnings[b], raw_sample_bins[b,:])
        end
        num_samples += samples_step
    end
    spin_samples = []
    for b in 1:bins
        spin_sample = [ vcat(raw_binnings[b][i], int_to_spin(i, spin_number)) for i in keys(raw_binnings[b])]
        push!(spin_samples, hcat(spin_sample...)')
    end
    return spin_samples
end

sample(gm::FactorGraph{T}, number_sample::Integer) where T <: Real = sample(gm, number_sample, 1, Gibbs())[1]
sample(gm::FactorGraph{T}, number_sample::Integer, replicates::Integer) where T <: Real = sample(gm, number_sample, replicates, Gibbs())


function sample(gm::FactorGraph{T}, number_sample::Integer, replicates::Integer, sampler::Gibbs) where T <: Real
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

function sample(gm::FactorGraph{T}, number_sample::Integer, sampler::Glauber) where T<: Real
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs, given alphabet $(gm.alphabet)")
    end
    equilibrated_state = gibbsMCsampler(gm, sampler.initial_steps, sampler.initial_state)

    raw_sample = sample_trajectory(gm, number_sample, sampler.spacing_steps, equilibrated_state)


    raw_binning = countmap(raw_sample)
    spin_sample = [vcat(raw_binning[state], state) for state in keys(raw_binning)]

    return hcat(spin_sample...)'
end

function gibbsMCsampler(gm::FactorGraph{T}, numsteps::Integer, initial_spin_state::Array{Int8,1}) where T <: Real
    all_neighborhoods = generate_neighborhoods(gm)

    current_state = deepcopy(initial_spin_state)

    @debug "Beginning from state $current_state"


    for i in 1:numsteps
        flipping_spin = rand(1:gm.variable_count)
        site_state = current_state[flipping_spin]

        site_contrib = 0.
        if haskey(all_neighborhoods, flipping_spin)
            for (term_sites, weight) in all_neighborhoods[flipping_spin]
                site_contrib += prod(current_state[term_sites])*weight
            end
        end


        weight_noflip = exp(site_contrib)
        weight_flip = 1/weight_noflip
        new_spin = StatsBase.sample([site_state, -site_state], StatsBase.Weights([weight_noflip, weight_flip]))
        current_state[flipping_spin] = new_spin


        @debug """
               Proposing flip on spin $flipping_spin with value $site_state
               Found local contribution $site_contrib
               Will remain in state with p∝$weight_noflip and
                              flip  with p∝$weight_flip
               New spin value is $new_spin
               New state is $current_state
               """


    end
    current_state
end

function sample_trajectory(gm::FactorGraph,
                           num_samples::Int,
                           sample_steps::Int,
                           initial_state::Array{Int8, 1})

    trajectory = [initial_state for i in 1:num_samples+1]
    step_idx = 1
    state = deepcopy(initial_state)

    while step_idx <= num_samples
        trajectory[step_idx+1] = gibbsMCsampler(gm, sample_steps, state)
        state = deepcopy(trajectory[step_idx+1])
        step_idx += 1
    end
    return trajectory
end
