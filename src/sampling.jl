export sample, sample_neighborhoods, samplesubset, samplearray

export GMSampler, Gibbs, Glauber

export gibbsMCsampler, glauber_step, glauber_step!, glauberstep_contribchange!

using StatsBase
using Random: shuffle
using LinearAlgebra

abstract type GMSampler end

struct Gibbs <: GMSampler end

mutable struct Glauber <: GMSampler
    initialstate::Array{Int8,1}
#    initial_steps::Int64
#    spacing_steps::Int64
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

"""
Converts a sample dictionary conf -> counts to a sample array where rows contain
key value pairs as [count, conf]
"""
function samplearray(sampledict::Dict{Vector{Int8}, Int64})
    vcat([vcat(count, conf)' for (conf, count) in sampledict]...)
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


function glauber_step!(state::Array{Int8, 1},
                     spin::Int64,
                     neighborhood::Array{Tuple{Array{Int64, 1}, Float64}, 1})

    site_contrib = 0.
    for (term_spins, w) in neighborhood
        site_contrib += prod(state[term_spins])*w
    end

    weight_noflip = exp(site_contrib)
    # state[spin] = StatsBase.sample([state[spin], -state[spin]], StatsBase.Weights([weight_noflip, 1/weight_noflip]))
    if rand() < (1 / (weight_noflip^2 + 1))
        state[spin] = -state[spin]
    end
end

function glauber_step(state::Array{Int8, 1},
                     spin::Int64,
                     neighborhood::Array{Tuple{Array{Int64,1}, Float64}, 1})

    site_contrib = 0.
    for (term_spins, w) in neighborhood
        site_contrib += prod(state[term_spins])*w
    end

    new_state = deepcopy(state)
    weight_noflip = exp(site_contrib)
    if rand() < (1 / (weight_noflip^2 + 1)) # flip if rand is less than pflip
        new_state[spin] = -state[spin]
    end

    # new_spin_state = StatsBase.sample([state[spin], -state[spin]], StatsBase.Weights([weight_noflip, 1/weight_noflip]))
    # new_state = deepcopy(state)
    # new_state[spin] = new_spin_state

    return new_state
end

function glauberstep_contribchange!(state::Array{Int8, 1},
                     spin::Int64,
                     neighborhood::Array{Tuple{Array{Int64, 1}, Float64}, 1})

    site_contrib = 0.
    for (term_spins, w) in neighborhood
        site_contrib += prod(state[term_spins])*w
    end

    weight_noflip = exp(site_contrib)
    # state[spin] = StatsBase.sample([state[spin], -state[spin]], StatsBase.Weights([weight_noflip, 1/weight_noflip]))
    if rand() < (1 / (weight_noflip^2 + 1))
        state[spin] = -state[spin]
        return -2*site_contrib
    end
    return 0
end

function gibbsMCsampler(gm::FactorGraph{T},
                        numsteps::Int64,
                        initial_spin_state::Array{Int8,1};
                        neighborhoods=nothing,
                        replace=true) where T <: Real

    if neighborhoods == nothing
        neighborhoods = generate_neighborhoods(gm)
    end

    current_state = deepcopy(initial_spin_state)

    @debug "Beginning from state $current_state"

    # If not sampling with replacement, iterate over a shuffled list of spins
    if !replace
        sampling_indices = shuffle(1:gm.variable_count)
    end

    for i in 1:numsteps
        if replace
            flipping_spin = rand(1:gm.variable_count)
        else
            spin_idx = ((i-1) % gm.variable_count) + 1
            flipping_spin = sampling_indices[spin_idx]
            # If we've iterated through all of them, shuffle again
            if spin_idx == gm.variable_count
                sampling_indices = shuffle(1:gm.variable_count)
            end
        end

        try
            glauber_step!(current_state, flipping_spin, neighborhoods[flipping_spin])
        catch KeyError
            # If the spin is disconnected just randomize it
            current_state[flipping_spin] = rand([1, -1])
        end

    end
    current_state
end

function truecondition(state::Array{T, 1}) where T <: Integer
    true
end

function sample_trajectory(gm::FactorGraph,
                           num_samples::Int64,
                           sample_steps::Int64,
                           initial_state::Array{Int8, 1};
                           replace::Bool=true,
                           condition::Function=truecondition)

    neighborhoods = generate_neighborhoods(gm)

    trajectory = fill(initial_state, num_samples+1)

    step_idx = 1

    while step_idx <= num_samples
        trajectory[step_idx+1] = gibbsMCsampler(gm, sample_steps, trajectory[step_idx], replace=replace, neighborhoods=neighborhoods)
        if condition(trajectory[step_idx+1])
            step_idx += 1
        end
    end
    return trajectory
end

function sample_trajectory(gm::FactorGraph,
                           num_samples::Int64,
                           initial_state::Array{Int8, 1};
                           replace::Bool=true,
                           condition::Function=truecondition)

    neighborhoods = generate_neighborhoods(gm)

    trajectory = fill(initial_state, num_samples+1)
    # If not sampling with replacement, iterate over a shuffled list of spins
    if !replace
        sampling_indices = shuffle(1:gm.variable_count)
    end

    step_idx = 1
    state = initial_state

    while step_idx <= num_samples
        if replace
            flipping_spin = rand(1:gm.variable_count)
        else
            spin_idx = ((step_idx-1) % gm.variable_count) + 1
            flipping_spin = sampling_indices[spin_idx]
            # If we've iterated through all of them, shuffle again
            if spin_idx == gm.variable_count
                sampling_indices = shuffle(1:gm.variable_count)
            end
        end
        try
            trajectory[step_idx+1] = glauber_step(trajectory[step_idx], flipping_spin, neighborhoods[flipping_spin])
        catch KeyError
            # If the spin is disconnected just randomize it
            trajectory[step_idx+1] = copy(trajectory[step_idx])
            trajectory[step_idx+1][flipping_spin] = rand([1, -1])
        end

        if condition(trajectory[step_idx+1])
            step_idx += 1
        end
    end
    return trajectory
end


"""
Performs MCMC sampling with Glauber dynamics on the graphical model gm. As
configured this returns ALL samples along the trajectory and so the initial
state provided in Glauber.initialstate should already be equilibrated. Kwarg
replace specifies whether steps should be selected with replacement, or whether
all sites in the model are shuffled and then iterated through. returnfinal
returns the final state of sampling so that it may be continued with a further
sampling run.

This returns a dictionary where keys are configurations and values are sample
counts.
"""

function sample(gm::FactorGraph{T},
                 numsamples::Int64,
                 sampler::Glauber;
                 replace::Bool=true,
                 returnfinal::Bool = false) where T <: Real

    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs. Given alphabet $(gm.alphabet)")
    end

    neighborhoods = generate_neighborhoods(gm)

    state = deepcopy(sampler.initialstate)
    sample_binning = countmap([state])

    if !replace
        sampling_indices = shuffle(1:gm.variable_count)
    end


    for sampleindex=1:numsamples-1

        if replace
            flipping_spin = rand(1:gm.variable_count)
        else
            flipping_count = ((sampleindex-1) % gm.variable_count) + 1
            flipping_spin = sampling_indices[flipping_count]
            # If we've iterated through all of them, shuffle again
            if flipping_count == gm.variable_count
                sampling_indices = shuffle(1:gm.variable_count)
            end
        end

        try
            glauber_step!(state, flipping_spin, neighborhoods[flipping_spin])
        catch KeyError
            # If the spin is disconnected just randomize it
            state[flipping_spin] = rand([1, -1])
        end

        addcounts!(sample_binning, [deepcopy(state)])

    end

    if returnfinal
        return sample_binning, state
    else
        return sample_binning
    end
end

function sample(gm::FactorGraph{T},
                 numsamples::Int64,
                 samplesteps::Int64,
                 sampler::Glauber;
                 replace::Bool=true,
                 returnfinal::Bool = false) where T <: Real

    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs. Given alphabet $(gm.alphabet)")
    end

    neighborhoods = generate_neighborhoods(gm)

    state = deepcopy(sampler.initialstate)
    sample_binning = countmap([state])




    for sampleindex=1:numsamples-1

        state = gibbsMCsampler(gm, samplesteps, state; neighborhoods=neighborhoods, replace = replace)

        addcounts!(sample_binning, [state])

    end

    if returnfinal
        return sample_binning, state
    else
        return sample_binning
    end
end



function sample_neighborhoods(gm::FactorGraph{T},
                            neighbors::Vector{Vector{Int64}},
                            number_sample::Int64,
                            sampler::Glauber;
                            replace::Bool = true,
                            sample_batch::Int64 = 100000,
                            condition::Function=truecondition,
                            returnfinal::Bool = false) where T<: Real

    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs. Given alphabet $(gm.alphabet)")
    end

    neighborhoods = generate_neighborhoods(gm)

    # Initial Run
    batch_size = min(sample_batch, number_sample)

    raw_sample = sample_trajectory(gm, batch_size, sampler.initialstate,
                                    replace=replace, condition=condition)

    neighbor_binning = Dict{Int64, Dict{Vector{Int8}, Int64}}()
    for spin in keys(neighborhoods)
        neighbor_binning[spin] = countmap([state[neighbors[spin]] for state in raw_sample[2:end]])
    end

    current_samples= batch_size

    # Then do runs of min(batch_size, number_sample - current_samples)
    while current_samples < number_sample
        batch_size = min(sample_batch, number_sample - current_samples)
        # Be sure to initialize at the last state of the previous run

        raw_sample .= sample_trajectory(gm, batch_size, raw_sample[end], replace=replace, condition=condition)

        current_samples += batch_size

        for spin in keys(neighborhoods)
            addcounts!(neighbor_binning[spin], [copy(state[neighbors[spin]]) for state in raw_sample[2:end]])
        end
    end

    if returnfinal
        return neighbor_binning, raw_sample[end]
    else
        return neighbor_binning
    end
end

function samplesubset(gm::FactorGraph{T},
                      spin_list::Vector{Int64},
                      num_samples::Int64,
                      sampler::Glauber;
                      returnfinal::Bool = false) where T <: Real

    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs. Given alphabet $(gm.alphabet)")
    end

    neighborhoods = generate_neighborhoods(gm)

    state = deepcopy(sampler.initialstate)
    sample_binning = countmap([state[spin_list]])
    sampling_indices = shuffle(1:gm.variable_count)


    for sampleindex=1:num_samples-1

        flipping_count = ((sampleindex-1) % gm.variable_count) + 1
        flipping_spin = sampling_indices[flipping_count]
        # If we've iterated through all of them, shuffle again
        if flipping_count == gm.variable_count
            sampling_indices = shuffle(1:gm.variable_count)
        end

        try
            glauber_step!(state, flipping_spin, neighborhoods[flipping_spin])
        catch KeyError
            # If the spin is disconnected just randomize it
            state[flipping_spin] = rand([1, -1])
        end

        addcounts!(sample_binning, [deepcopy(state[spin_list])])

    end
    if returnfinal
        return sample_binning, state
    else
        return sample_binning
    end
end
