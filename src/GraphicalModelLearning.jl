isdefined(Base, :__precompile__) && __precompile__()

module GraphicalModelLearning

export learn, inverse_ising, gibbs_sampler

export GMLFormulation, RISE, logRISE, RPLE
export GMLMethod, NLP

using JuMP
using Ipopt

using Compat # used for julia v0.5 abstract types

@compat abstract type GMLFormulation end

type RISE <: GMLFormulation
    regularizer::Float64
    symmetrization::Bool
end
RISE() = RISE(0.8, true) # default values

type logRISE <: GMLFormulation
    regularizer::Float64
    symmetrization::Bool
end
logRISE() = logRISE(0.8, true) # default values

type RPLE <: GMLFormulation
    regularizer::Float64
    symmetrization::Bool
end
RPLE() = RPLE(0.8, true) # default values


@compat abstract type GMLMethod end

type NLP <: GMLMethod
    solver
end
NLP() = NLP(IpoptSolver(tol=1e-12, print_level=0))


# default settings
learn{T <: Real}(samples::Array{T,2}) = learn(samples, RISE(), NLP())
learn{T <: Real, S <: GMLFormulation}(samples::Array{T,2}, formulation::S) = learn(samples, formulation, NLP())


function learn{T <: Real}(samples::Array{T,2}, formulation::RISE, method::NLP)
    (num_conf, num_row) = size(samples)
    num_spins           = num_row - 1

    reconstruction = Array{Float64}(num_spins, num_spins)

    num_samples = sum(samples[1:num_conf,1])
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        m = Model(solver = method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples[k,1]/num_samples)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        status = solve(m)
        @assert status == :Optimal
        reconstruction[current_spin,1:num_spins] = deepcopy(getvalue(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction 
end


function learn{T <: Real}(samples::Array{T,2}, formulation::logRISE, method::NLP)
    (num_conf, num_row) = size(samples)
    num_spins           = num_row - 1

    reconstruction = Array{Float64}(num_spins, num_spins)

    num_samples = sum(samples[1:num_conf,1])
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        m = Model(solver = method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            log(sum((samples[k,1]/num_samples)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf)) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        status = solve(m)
        @assert status == :Optimal
        reconstruction[current_spin,1:num_spins] = deepcopy(getvalue(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction 
end


function learn{T <: Real}(samples::Array{T,2}, formulation::RPLE, method::NLP)
    (num_conf, num_row) = size(samples)
    num_spins           = num_row - 1

    reconstruction = Array{Float64}(num_spins, num_spins)

    num_samples = sum(samples[1:num_conf,1])
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        m = Model(solver = method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples[k,1]/num_samples)*log(1 + exp(-2*sum(x[i]*nodal_stat[k,i] for i=1:num_spins))) for k=1:num_conf) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        status = solve(m)
        @assert status == :Optimal
        reconstruction[current_spin,1:num_spins] = deepcopy(getvalue(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction 
end



# formulations: :RISE, :logRISE, :RPLE

function inverse_ising_archive(samples_histo; method::Symbol=:logRISE, regularizing_value::Float64=0.8, symmetrization::Symbol=:Yes)
    (num_conf, num_row) = size(samples_histo)
    num_spins           = num_row - 1

    reconstruction = Array{Float64}(num_spins, num_spins)

    #Extraction of the total number of samples contained in the histogram.
    num_samples = sum(samples_histo[1:num_conf,1])

    #Initialization of the regularizing parameter from regularization coefficient. Declaration of the RPLE and RISE objective funcions.
    lambda           = regularizing_value*sqrt(log((num_spins^2)/0.05)/num_samples)
    RISEobjective(h) = exp(-h)
    RPLEobjective(h) = log(1 + exp(-2h))

    #Loop for the reconstruction of couplings around "current_spin"
    for current_spin = 1:num_spins
        #Printing out progress
        #println("Reconstructing the parameters adjacent to node ", current_spin);

        #Construction of the statistics around node "current_spin" in the histogram format.
        #An element (k,i) reads "configuration[k,current_spin] * configuration[k,i]" if i != current_spin and "configuration[k,current_spin]" otherwise.
        nodal_stat  = [ samples_histo[k, 1 + current_spin] * (i == current_spin ? 1 : samples_histo[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        #Declaration in JuMP of the optimization model "m" and the convex solver. Here the convex solver is choosen to be Ipopt.
        #The tolerance tol is chosen based on experiments in the paper, remove for using a default tolerance of the solver, e.g. solver = IpoptSolver()
        #The option print_level=0 disables the output of the Ipopt solver, remove for reading a default detailed information on the convergence, e.g. solver = IpoptSolver()
        m = Model(solver = IpoptSolver(tol=1e-12, print_level=0))

        #Initialization of the loss function for JuMP. RISE is choosen by default unless the user specifies RPLE in the arguments.
        JuMP.register(m, :IIPobjective, 1, (method == :RPLE ? RPLEobjective : RISEobjective), autodiff=true)

        #Declaration in JuMP of "x", the array of variables for couplings and magnetic fields and "z", the array of slack variables for the l1 norm.
        #The magnetic field variable is x[current_spin] and the coupling variables are x[i] for i!= current_spin. The slack variable z[current_spin] is uneccessary.
        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        #Declaration in JuMP of the objective function: (log(RISE) + l1, RISE + l1, RPLE + l1). RISE is choosen by default unless the user specifies RPLE or logRISE in the arguments.
        if method == :logRISE
            @NLobjective(m, Min,
                log(sum((samples_histo[k,1]/num_samples)* IIPobjective(sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf)) +
                lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
                )
        else
            @NLobjective(m, Min,
                sum((samples_histo[k,1]/num_samples)* IIPobjective(sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf) +
                lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
                )
        end

        #Declaration in JuMP of slack constraints for the l1 penalty.
        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        #Lauching convex optimization, printing results and updating the matrix of reconstructed parameters accordingly.
        status = solve(m)
        #println(current_spin, " = ", getvalue(x))
        reconstruction[current_spin,1:num_spins] = deepcopy(getvalue(x))

    end

    #symmetrization of the couplings. No symmetrization is choosen by defaut unless the user specifies "Y" in the arguments.
    if symmetrization == "Y"
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction 
end



using StatsBase

function int_to_spin(int_representation, spin_number)
  spin = 2*digits(int_representation, 2, spin_number)-1
  return spin
end

function weigh_proba(int_representation, adj, prior)
  spin_number  = size(adj,1)
  current_spin = int_to_spin(int_representation, spin_number)
  return exp(((0.5) * current_spin' * adj * current_spin + prior * current_spin)[1])
end

function sample_generation(sample_number, adj, prior)
  spin_number   = size(adj,1)
  config_number = 2^spin_number

  items   = [i for i in 0:(config_number-1)]
  weights = [weigh_proba(i, adj, prior) for i in (0:config_number-1)]

  raw_sample = sample(items, StatsBase.Weights(weights), sample_number)
  raw_binning= countmap(raw_sample)

  spin_sample = [ vcat(raw_binning[i], int_to_spin(i, spin_number)) for i in keys(raw_binning)]
  spin_sample = hcat(spin_sample...)'
  return spin_sample
end

function gibbs_sampler(adjacency_matrix, number_sample::Int64)
    prior_vector = transpose(diag(adjacency_matrix)) #priors, or magnetic fields part

    #Generation of samples
    samples = sample_generation(number_sample, adjacency_matrix, prior_vector)

    return samples
end

end