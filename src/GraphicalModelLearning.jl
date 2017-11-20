isdefined(Base, :__precompile__) && __precompile__()

module GraphicalModelLearning

export learn, inverse_ising

export GMLFormulation, RISE, logRISE, RPLE, RISEA
export GMLMethod, NLP

using JuMP
using MathProgBase # for solver type
using Ipopt

using Compat # used for julia v0.5 abstract types

include("models.jl")

include("sampling.jl")

@compat abstract type GMLFormulation end

type RISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RISE() = RISE(0.4, true)

type RISEA <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RISEA() = RISEA(0.4, true)

type logRISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
logRISE() = logRISE(0.8, true)

type RPLE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RPLE() = RPLE(0.2, true)


@compat abstract type GMLMethod end

type NLP <: GMLMethod
    solver::MathProgBase.AbstractMathProgSolver
end
# default values
NLP() = NLP(IpoptSolver(print_level=0))


# default settings
learn{T <: Real}(samples::Array{T,2}) = learn(samples, RISE(), NLP())
learn{T <: Real, S <: GMLFormulation}(samples::Array{T,2}, formulation::S) = learn(samples, formulation, NLP())


function data_info{T <: Real}(samples::Array{T,2})
    (num_conf, num_row) = size(samples)
    num_spins = num_row - 1
    num_samples = sum(samples[1:num_conf,1])
    return num_conf, num_spins, num_samples
end


function learn{T <: Real}(samples::Array{T,2}, formulation::RISE, method::NLP)
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(num_spins, num_spins)

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

function learn{T <: Real}(samples::Array{T,2}, formulation::RISEA, method::NLP)
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(num_spins, num_spins)

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

        m = Model(solver = method.solver)

        JuMP.register(m, :obj, num_spins, obj, grad)
        JuMP.register(m, :l1norm, num_spins, l1norm, autodiff=true)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])


        JuMP.setNLobjective(m, :Min, Expr(:call, :+,
                                            Expr(:call, :obj, x...),
                                            Expr(:call, :l1norm, z...)
                                        )
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
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(num_spins, num_spins)

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
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(num_spins, num_spins)

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


end
