module GraphicalModelLearning

export learn, inverse_ising, learn_old

export GMLFormulation, RISE, logRISE, RPLE, RISEA, multiRISE
export GMLMethod, NLP, EntropicDescent

using JuMP
using Ipopt

import LinearAlgebra
import LinearAlgebra: diag
import Statistics: mean


include("models.jl")

include("sampling.jl")

abstract type GMLFormulation end

mutable struct multiRISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
    interaction_order::Integer
end
# default values
multiRISE() = multiRISE(0.4, true, 2)

mutable struct RISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RISE() = RISE(0.4, true)

mutable struct RISEA <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RISEA() = RISEA(0.4, true)

mutable struct logRISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
logRISE() = logRISE(0.8, true)

mutable struct RPLE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RPLE() = RPLE(0.2, true)


abstract type GMLMethod end

mutable struct NLP <: GMLMethod
    solver::Any
end
# default values
NLP() = NLP(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

mutable struct EntropicDescent <: GMLMethod
    max_steps::Int64
    init_stepsize::Float64
    l1_bound::Float64
    grad_termination::Float64
end
# default values
EntropicDescent() = EntropicDescent(1e4, 0.05, 2., 1e-8)

# default settings
learn(samples::Array{T,2}) where T <: Real = learn(samples, RISE(), NLP())
learn(samples::Array{T,2}, formulation::S) where {T <: Real, S <: GMLFormulation} = learn(samples, formulation, NLP())

#TODO add better support for Adjoints
learn(samples::LinearAlgebra.Adjoint, args...) = learn(copy(samples), args...)


function data_info(samples::Array{T,2}) where T <: Real
    (num_conf, num_row) = size(samples)
    num_spins = num_row - 1
    num_samples = sum(samples[1:num_conf,1])
    return num_conf, num_spins, num_samples
end

function learn(samples::Array{T,2}, formulation::multiRISE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    inter_order = formulation.interaction_order

    reconstruction = Dict{Tuple,Real}()

    for current_spin = 1:num_spins
        nodal_stat = Dict{Tuple,Array{Real,1}}()

        for p = 1:inter_order
                nodal_keys = Array{Tuple{},1}()
                neighbours = [i for i=1:num_spins if i!=current_spin]
                if p == 1
                    nodal_keys = [(current_spin,)]
                else
                    perm = permutations(neighbours, p - 1)
                    if length(perm) > 0
                        nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
                    end
                end

                for index = 1:length(nodal_keys)
                    nodal_stat[nodal_keys[index]] =  [ prod(samples[k, 1 + i] for i=nodal_keys[index]) for k=1:num_conf]
                end
        end

        model = Model(method.solver)

        @variable(model, x[keys(nodal_stat)])
        @variable(model, z[keys(nodal_stat)])

        @NLobjective(model, Min,
            sum((samples[k,1]/num_samples)*exp(-sum(x[inter]*stat[k] for (inter,stat) = nodal_stat)) for k=1:num_conf) +
            lambda*sum(z[inter] for inter = keys(nodal_stat) if length(inter)>1)
        )

        for inter in keys(nodal_stat)
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED

        nodal_reconstruction = JuMP.value.(x)
        for inter = keys(nodal_stat)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
        end
    end

    if formulation.symmetrization
        reconstruction_list = Dict{Tuple,Vector{Real}}()
        for (k,v) in reconstruction
            key = tuple(sort([i for i in k])...)
            if !haskey(reconstruction_list, key)
                reconstruction_list[key] = Vector{Real}()
            end
            push!(reconstruction_list[key], v)
        end

        reconstruction = Dict{Tuple,Real}()
        for (k,v) in reconstruction_list
            reconstruction[k] = mean(v)
        end
    end

    return FactorGraph(inter_order, num_spins, :spin, reconstruction)
end

function learn(samples::Array{T,2}, formulation::RISE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        model = Model(method.solver)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])

        @NLobjective(model, Min,
            sum((samples[k,1]/num_samples)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

function risea_obj(var, stat, weight)
    (num_conf, num_spins) = size(stat)
    #chvar = cosh.(var)
    #shvar = sinh.(var)
    #return sum(weight[k]*prod(chvar[i] - shvar[i]*stat[k,i] for i=1:num_spins) for k=1:num_conf)
    return sum(weight[k]*exp(-sum(var[i]*stat[k,i] for i=1:num_spins)) for k=1:num_conf)
end

function grad_risea_obj(g, var, stat, weight)
    (num_conf, num_spins) = size(stat)
    #chvar = cosh.(var)
    #shvar = sinh.(var)
    #partial_obj = [- weight[k] * prod(chvar[i] - shvar[i]*stat[k,i] for i=1:num_spins) for k=1:num_conf]
    partial_obj = [- weight[k]*exp(-sum(var[i]*stat[k,i] for i=1:num_spins)) for k=1:num_conf]
    for i=1:num_spins
        g[i] = sum(stat[k,i]*partial_obj[k] for k=1:num_conf)
    end
end

function learn(samples::Array{T,2}, formulation::RISEA, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]
        weight = samples[1:num_conf,1] / num_samples

        obj(x...) = risea_obj(x, nodal_stat, weight)
        function grad(g, x...)
            grad_risea_obj(g, x, nodal_stat, weight)
        end

        function l1norm(z...)
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        end

        model = Model(method.solver)

        JuMP.register(model, :obj, num_spins, obj, grad)
        JuMP.register(model, :l1norm, num_spins, l1norm, autodiff=true)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])


        JuMP.setNLobjective(model, :Min, Expr(:call, :+,
                                            Expr(:call, :obj, x...),
                                            Expr(:call, :l1norm, z...)
                                        )
                            )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end


function learn(samples::Array{T,2}, formulation::logRISE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        model = Model(method.solver)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])

        @NLobjective(model, Min,
            log(sum((samples[k,1]/num_samples)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf)) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end


function learn(samples::Array{T,2}, formulation::RPLE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        model = Model(method.solver)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])

        @NLobjective(model, Min,
            sum((samples[k,1]/num_samples)*log(1 + exp(-2*sum(x[i]*nodal_stat[k,i] for i=1:num_spins))) for k=1:num_conf) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end


function learn(samples::Array{T,2}, formulation::RISE, method::EntropicDescent; return_objectives=false) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    #lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    max_steps = method.max_steps
    η_init = method.init_stepsize
    l1_bound = method.l1_bound
    grad_termination = method.grad_termination

    @info "Running Entropic descent: $max_steps steps, $l1_bound element bound"
    reconstruction = Array{Float64}(undef, num_spins, num_spins)
    if return_objectives
        objectives = Vector{Array{Float64}}(undef, num_spins)
    end

    for current_spin = 1:num_spins
        #contains all spin products with current spin in each configuration (current spin value in its spot)
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        #Initialize
        x_plus = [1/(2*num_spins + 1) for i=1:num_spins]
        x_minus = [1/(2*num_spins + 1) for i=1:num_spins]
        y = 1/(2*num_spins + 1)
        η = η_init

        est = l1_bound .* (x_plus - x_minus)
        exp_arg = nodal_stat * est
        obj = sum((samples[k,1]/num_samples)*exp(-exp_arg[k]) for k=1:num_conf)
        grad = ones(num_spins)

        best_est = est
        best_obj = obj

        # Track objective
        if return_objectives
            spin_objective = [best_obj]
        end



        t = 1
        while t <= max_steps && maximum(abs.(grad)) > grad_termination

            # gradient step
            grad = [l1_bound*sum((samples[k,1]/num_samples)*(-nodal_stat[k,i])*exp(-exp_arg[k]) for k=1:num_conf) for i=1:num_spins]
            grad_obj = grad ./ obj
            w_plus = x_plus .* exp.(-η .* grad_obj)
            w_minus = x_minus .* exp.(η .* grad_obj)

            # projection step
            z = y + sum(w_plus + w_minus)
            x_plus = w_plus ./ z
            x_minus = w_minus ./ z
            y = y/z
            η = η * sqrt(t/(t+1))

            # track lowest objective and estimate
            est .= l1_bound .* (x_plus - x_minus)
            exp_arg .= nodal_stat*est
            obj = sum((samples[k,1]/num_samples)*exp(-exp_arg[k]) for k=1:num_conf)
            if obj < best_obj
                best_obj = obj
                best_est .= est
            end
            # For debugging and plotting the objectives
            if return_objectives
                push!(spin_objective, obj)
            end
            t += 1
        end

        if t > max_steps
            constraint_slack = l1_bound - sum(abs.(est))
            @warn "Maximum steps reached for site $current_spin, max gradient=$(maximum(abs.(grad))), constraint slack = $constraint_slack "
        end
        reconstruction[current_spin,1:num_spins] = deepcopy(best_est)
        if return_objectives
            objectives[current_spin] = spin_objective
        end
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end
    if return_objectives
        return reconstruction, objectives
    else
        return reconstruction
    end
end

end
