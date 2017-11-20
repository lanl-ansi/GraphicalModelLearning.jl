# data structures graphical models

export FactorGraph

alphabets = [:spin, :boolean, :integer, :integer_pos, :real, :real_pos]

type FactorGraph{T <: Real}
    order::Int
    varible_count::Int
    alphabet::Symbol
    terms::Dict{Tuple,T} # TODO, would be nice to have a stronger tuple type here
    variable_names::Nullable{Vector{String}}
    FactorGraph(a,b,c,d,e) = check_model_data(a,b,c,d,e) ? new(a,b,c,d,e) : error("generic init problem")
end
FactorGraph{T <: Real}(order::Int, varible_count::Int, alphabet::Symbol, terms::Dict{Tuple,T}) = FactorGraph{T}(order, varible_count, alphabet, terms, Nullable{Vector{String}}())
FactorGraph{T <: Real}(matrix::Array{T,2}) = convert(FactorGraph{T}, matrix)

function check_model_data{T <: Real}(order::Int, varible_count::Int, alphabet::Symbol, terms::Dict{Tuple,T}, variable_names::Nullable{Vector{String}})
    if !in(alphabet, alphabets)
        error("alphabet $(alphabet) is not supported")
        return false 
    end
    if !isnull(variable_names) && length(variable_names) != varible_count
        error("expected $(varible_count) but only given $(length(variable_names))")
        return false 
    end
    for (k,v) in terms
        if length(k) != order
            error("a term has $(length(k)) indices but should have $(order) indices")
            return false
        end
        for (i,index) in enumerate(k)
            #println(i," ",index)
            if index < 1 || index > varible_count
                error("a term has an index of $(index) but it should be in the range of 1:$(varible_count)")
                return false
            end
            if i > 1
                if k[i-1] > index
                    error("the term $(k) does not have ascending indices")
                end
            end
        end
    end
    return true
end

function Base.show(io::IO, gm::FactorGraph)
    println(io, "alphabet: ", gm.alphabet)
    println(io, "vars: ", gm.varible_count)
    if !isnull(gm.variable_names)
        println(io, "variable names: ")
        println(io, "  ", get(gm.variable_names))
    end

    println(io, "terms: ")
    for k in sort(collect(keys(gm.terms)))
        println("  ", k, " => ", gm.terms[k])
    end
end

Base.start(gm::FactorGraph) = start(gm.terms)
Base.next(gm::FactorGraph, state) = next(gm.terms, state)
Base.done(gm::FactorGraph, state) = done(gm.terms, state)

Base.length(gm::FactorGraph) = length(gm.terms)

Base.getindex(gm::FactorGraph, i) = gm.terms[i]
Base.keys(gm::FactorGraph) = keys(gm.terms)


function diag_keys(gm::FactorGraph)
    dkeys = Tuple[]
    for i in 1:gm.varible_count
        key = diag_key(gm, i)
        if key in keys(gm.terms)
            push!(dkeys, key)
        end
    end
    return sort(dkeys)
end

diag_key(gm::FactorGraph, i::Int) = tuple(fill(i, gm.order)...)

#Base.diag{T <: Real}(gm::FactorGraph{T}) = [ get(gm.terms, diag_key(gm, i), zero(T)) for i in 1:gm.varible_count ]

Base.DataFmt.writecsv{T <: Real}(io, gm::FactorGraph{T}, args...; kwargs...) = writecsv(io, convert(Array{T,2}, gm), args...; kwargs...)

Base.convert{T <: Real}(::Type{FactorGraph}, m::Array{T,2}) = convert(FactorGraph{T}, m)
function Base.convert{T <: Real}(::Type{FactorGraph{T}}, m::Array{T,2})
    @assert size(m,1) == size(m,2) #check matrix is square

    info("assuming spin alphabet")
    alphabet = :spin
    varible_count = size(m,1)

    terms = Dict{Tuple,T}()
    for key in permutations(1:varible_count, 2)
        weight = m[key...]
        if !isapprox(weight, 0.0)
            terms[key] = weight
        end

        rev = reverse(key)
        if !isapprox(m[rev...], 0.0) && !isapprox(m[key...], m[rev...])
            delta = abs(m[key...] - m[rev...])
            warn("values at $(key) and $(rev) differ by $(delta), only $(key) will be used")
        end
    end

    return FactorGraph(2, varible_count, alphabet, terms)
end

function Base.convert{T <: Real}(::Type{Array{T,2}}, gm::FactorGraph{T})
    if gm.order != 2
        error("cannot convert a FactorGraph of order $(gm.order) to a matrix")
    end

    matrix = zeros(gm.varible_count, gm.varible_count)
    for (k,v) in gm
        matrix[k...] = v
        r = reverse(k)
        matrix[r...] = v
    end

    return matrix
end



permutations(items, order::Int; asymmetric::Bool = false) = sort(permutations([], items, order, asymmetric))

function permutations(partical_perm::Array{Any,1}, items, order::Int, asymmetric::Bool)
    if order == 0
        return [tuple(partical_perm...)]
    else
        perms = []
        for item in items
            if !asymmetric && length(partical_perm) > 0 
                if partical_perm[end] < item
                    continue
                end
            end
            perm = permutations(vcat([item], partical_perm), items, order-1, asymmetric)
            append!(perms, perm)
        end
        return perms
    end
end

