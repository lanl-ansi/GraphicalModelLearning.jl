module GraphicalModelLearning

export learn, inverse_ising

export GMLFormulation, RISE, logRISE, RPLE, RISEA, multiRISE
export GMLMethod, NLP

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
    solver::JuMP.OptimizerFactory
end
# default values
NLP() = NLP(with_optimizer(Ipopt.Optimizer, print_level=0))


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


end
