export sample, sample_neighborhood

export GMSampler, Gibbs, Glauber

export gibbsMCsampler, glauber_step, glauber_step!

using StatsBase
using Random: shuffle
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

function sample(gm::FactorGraph{T},
                number_sample::Integer,
                sampler::Glauber;
                replace::Bool = true,
                sample_batch::Int64 = 10000,
                condition::Function=true_condition) where T<: Real
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs. Given alphabet $(gm.alphabet)")
    end

    equilibrated_state = gibbsMCsampler(gm, sampler.initial_steps, sampler.initial_state, replace=replace)
    # Possibly problematic.  If we end up messing with conditions more then
    # should fix so that condition is attached to the individual steps
    while !condition(equilibrated_state)
        equilibrated_state = gibbsMCsampler(gm, sampler.initial_steps, sampler.initial_state, replace=replace)
    end

    # Initial Run
    batch_size = min(sample_batch, number_sample)

    if sampler.spacing_steps==1
        raw_sample = sample_trajectory(gm, batch_size, equilibrated_state, replace=replace, condition=condition)
    else
        raw_sample = sample_trajectory(gm, batch_size, sampler.spacing_steps, equilibrated_state, replace=replace, condition=condition)
    end

    raw_binning = countmap(raw_sample[2:end])
    current_samples= batch_size

    # Then do runs of min(batch_size, number_sample - current_samples)
    while current_samples < number_sample
        batch_size = min(sample_batch, number_sample - current_samples)
        # Be sure to initialize at the last state of the previous run
        if sampler.spacing_steps==1
            raw_sample = sample_trajectory(gm, batch_size, raw_sample[end], replace=replace, condition=condition)
        else
            raw_sample = sample_trajectory(gm, batch_size, sampler.spacing_steps, raw_sample[end], replace=replace, condition=condition)
        end
        current_samples += batch_size

        raw_binning = addcounts!(raw_binning, raw_sample[2:end])
    end
    # returns array with counts in first column followed by states in rows
    return vcat([vcat(count, state)' for (state, count) in raw_binning]...)

end


function glauber_step!(state::Array{Int8, 1},
                     spin::Int,
                     neighborhood::Array{Tuple{Array, Float64}, 1})

    spin_state = state[spin]

    site_contrib = 0.
    for (term_spins, w) in neighborhood
        site_contrib += prod(state[term_spins])*w
    end

    weight_noflip = exp(site_contrib)
    weight_flip = 1/weight_noflip
    new_spin = StatsBase.sample([spin_state, -spin_state], StatsBase.Weights([weight_noflip, weight_flip]))
    state[spin] = new_spin


    @debug """
           Proposing flip on spin $spin with value $spin_state
           Found local contribution $site_contrib
           Will remain in state with p∝$weight_noflip and
                          flip  with p∝$weight_flip
           New spin value is $new_spin_state
           New state is $new_state
           """
end

function glauber_step(state::Array{Int8, 1},
                     spin::Int,
                     neighborhood::Array{Tuple{Array, Float64}, 1})

    spin_state = state[spin]

    site_contrib = 0.
    for (term_spins, w) in neighborhood
        site_contrib += prod(state[term_spins])*w
    end

    weight_noflip = exp(site_contrib)
    weight_flip = 1/weight_noflip
    new_spin_state = StatsBase.sample([spin_state, -spin_state], StatsBase.Weights([weight_noflip, weight_flip]))
    new_state = deepcopy(state)
    new_state[spin] = new_spin_state


    @debug """
           Proposing flip on spin $spin with value $spin_state
           Found local contribution $site_contrib
           Will remain in state with p∝$weight_noflip and
                          flip  with p∝$weight_flip
           New spin value is $new_spin_state
           New state is $new_state
           """
    return new_state
end

function gibbsMCsampler(gm::FactorGraph{T},
                        numsteps::Integer,
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

        if haskey(neighborhoods, flipping_spin)
            glauber_step!(current_state, flipping_spin, neighborhoods[flipping_spin])
        else
            # If the spin is disconnected just randomize it
            current_state[flipping_spin] = rand([1, -1])
        end

    end
    current_state
end

function true_condition(state::Array{T, 1}) where T <: Integer
    true
end

function sample_trajectory(gm::FactorGraph,
                           num_samples::Int,
                           sample_steps::Int,
                           initial_state::Array{Int8, 1};
                           replace::Bool=true,
                           condition::Function=true_condition)

    neighborhoods = generate_neighborhoods(gm)

    trajectory = [copy(initial_state) for i in 1:num_samples+1]

    step_idx = 1
    state = initial_state

    while step_idx <= num_samples
        trajectory[step_idx+1] = gibbsMCsampler(gm, sample_steps, state, replace=replace, neighborhoods=neighborhoods)
        if condition(trajectory[step_idx+1])
            state = deepcopy(trajectory[step_idx+1])
            step_idx += 1
        end
    end
    return trajectory
end

function sample_trajectory(gm::FactorGraph,
                           num_samples::Int,
                           initial_state::Array{Int8, 1};
                           replace::Bool=true,
                           condition::Function=true_condition)

    neighborhoods = generate_neighborhoods(gm)

    trajectory = [copy(initial_state) for i in 1:num_samples+1]
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
        if haskey(neighborhoods, flipping_spin)
            trajectory[step_idx+1] = glauber_step(state, flipping_spin, neighborhoods[flipping_spin])
        else
            # If the spin is disconnected just randomize it
            trajectory[step_idx+1] = copy(state)
            trajectory[step_idx+1][flipping_spin] = rand([1, -1])
        end

        if condition(trajectory[step_idx+1])
            state = trajectory[step_idx+1]
            step_idx += 1
        end
    end
    return trajectory
end

"""
For learning a structured model, we only require the values of the
spin products over interactions for each site.  As these will
vary over a much smaller space than the full configuration, memory
can be saved by storing samples as histograms of interactions for each site.
The dictionary generate neighbors provides ordering of the interactions.  The
output is a dictionary with keys over spin indices and values containing a dictionary
of counts (as returned by countmap).
"""
function sample_neighborhood(gm::FactorGraph{T},
                            number_sample::Integer,
                            sampler::Glauber;
                            replace::Bool = true,
                            sample_batch::Int64 = 100000,
                            condition::Function=true_condition) where T<: Real
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs. Given alphabet $(gm.alphabet)")
    end

    neighborhoods = generate_neighborhoods(gm)

    equilibrated_state = gibbsMCsampler(gm, sampler.initial_steps, sampler.initial_state, replace=replace)

    # Possibly problematic.  If we end up messing with conditions more then
    # should fix so that condition is attached to the individual steps
    while !condition(equilibrated_state)
        equilibrated_state = gibbsMCsampler(gm, sampler.initial_steps, sampler.initial_state, replace=replace)
    end

    # Initial Run
    batch_size = min(sample_batch, number_sample)

    if sampler.spacing_steps==1
        raw_sample = sample_trajectory(gm, batch_size, equilibrated_state, replace=replace, condition=condition)
    else
        raw_sample = sample_trajectory(gm, batch_size, sampler.spacing_steps, equilibrated_state, replace=replace, condition=condition)
    end

    neighbor_binning = Dict{Int, Dict{Vector{Int}, Int}}()
    for spin in keys(neighborhoods)
        conf_sample = [[prod(state[interacting]) for (interacting, w) in neighborhoods[spin]] for state in raw_sample[2:end]]
        neighbor_binning[spin] = countmap(conf_sample)
    end

    current_samples= batch_size

    # Then do runs of min(batch_size, number_sample - current_samples)
    while current_samples < number_sample
        batch_size = min(sample_batch, number_sample - current_samples)
        # Be sure to initialize at the last state of the previous run
        if sampler.spacing_steps==1
            raw_sample = sample_trajectory(gm, batch_size, raw_sample[end], replace=replace, condition=condition)
        else
            raw_sample = sample_trajectory(gm, batch_size, sampler.spacing_steps, raw_sample[end], replace=replace, condition=condition)
        end
        current_samples += batch_size

        for spin in keys(neighborhoods)
            conf_sample = [[prod(state[interacting]) for (interacting, w) in neighborhoods[spin]] for state in raw_sample[2:end]]
             addcounts!(neighbor_binning[spin], conf_sample)
        end
    end
    # returns array with counts in first column followed by states in rows
    return neighbor_binning
end
