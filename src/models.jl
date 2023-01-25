# data structures graphical models

export FactorGraph, jsondata

export completegraph, curieweiss, cubiclattice, edwardsanderson_spinglass

export ferromagnet_nbodypoisson, spinglass_nbodypoisson

export generate_neighborhoods, find_term, localterms, neighborarray

using StatsBase

alphabets = [:spin, :boolean, :integer, :integer_pos, :real, :real_pos]


mutable struct FactorGraph{T <: Real}
    order::Int
    variable_count::Int
    alphabet::Symbol
    terms::Dict{Tuple,T} # TODO, would be nice to have a stronger tuple type here
    variable_names::Union{Vector{String}, Nothing}
    #FactorGraph(a,b,c,d,e) = check_model_data(a,b,c,d,e) ? new(a,b,c,d,e) : error("generic init problem")
end

FactorGraph(order::Int, variable_count::Int, alphabet::Symbol, terms::Dict{Tuple,T}) where T <: Real = FactorGraph{T}(order, variable_count, alphabet, terms, nothing)
FactorGraph(matrix::Array{T,2}) where T <: Real = convert(FactorGraph{T}, matrix)
FactorGraph(dict::Dict{Tuple,T}) where T <: Real = convert(FactorGraph{T}, dict)
FactorGraph(list::Array{Any,1}) = convert(FactorGraph, list)


function check_model_data(order::Int, variable_count::Int, alphabet::Symbol, terms::Dict{Tuple,T}, variable_names::Union{Vector{String}, Nothing}) where T <: Real
    if !in(alphabet, alphabets)
        error("alphabet $(alphabet) is not supported")
        return false
    end
    if variable_names != nothing && length(variable_names) != variable_count
        error("expected $(variable_count) but only given $(length(variable_names))")
        return false
    end
    for (k,v) in terms
        if length(k) > order
            error("a term has $(length(k)) indices but should have $(order) indices")
            return false
        end
        for (i,index) in enumerate(k)
            #println(i," ",index)
            if index < 1 || index > variable_count
                error("a term has an index of $(index) but it should be in the range of 1:$(variable_count)")
                return false
            end
            #=
            # TODO see when this should be enforced
            if i > 1
                if k[i-1] > index
                    error("the term $(k) does not have ascending indices")
                end
            end
            =#
        end
    end
    return true
end

function Base.show(io::IO, gm::FactorGraph)
    println(io, "alphabet: ", gm.alphabet)
    println(io, "vars: ", gm.variable_count)
    if gm.variable_names != nothing
        println(io, "variable names: ")
        println(io, "  ", get(gm.variable_names))
    end

    println(io, "terms: $(length(gm.terms))")
    for k in sort(collect(keys(gm.terms)), by=(x)->(length(x),x))
        println("  ", k, " => ", gm.terms[k])
    end
end

function jsondata(gm::FactorGraph{T}) where T <: Real
    data = []
    for k in sort(collect(keys(gm.terms)), by=(x)->(length(x),x))
        push!(data, Dict("term" => k, "weight" => gm.terms[k]))
    end
    return data
end


Base.iterate(gm::FactorGraph, kwargs...) = Base.iterate(gm.terms, kwargs...)

Base.length(gm::FactorGraph) = length(gm.terms)
#Base.size(gm::FactorGraph, a...) = size(gm.terms, a...)

Base.getindex(gm::FactorGraph, i) = gm.terms[i]
Base.keys(gm::FactorGraph) = keys(gm.terms)

function diag_keys(gm::FactorGraph)
    dkeys = Tuple[]
    for i in 1:gm.variable_count
        key = diag_key(gm, i)
        if key in keys(gm.terms)
            push!(dkeys, key)
        end
    end
    return sort(dkeys)
end

diag_key(gm::FactorGraph, i::Int) = tuple(fill(i, gm.order)...)

#Base.diag(gm::FactorGraph{T}) where T <: Real = [ get(gm.terms, diag_key(gm, i), zero(T)) for i in 1:gm.variable_count ]

#Base.DataFmt.writecsv(io, gm::FactorGraph{T}, args...; kwargs...) where T <: Real = writecsv(io, convert(Array{T,2}, gm), args...; kwargs...)

Base.convert(::Type{FactorGraph}, m::Array{T,2}) where T <: Real = convert(FactorGraph{T}, m)
function Base.convert(::Type{FactorGraph{T}}, m::Array{T,2}) where T <: Real
    @assert size(m,1) == size(m,2) #check matrix is square

    @info "assuming spin alphabet"
    alphabet = :spin
    variable_count = size(m,1)

    terms = Dict{Tuple,T}()

    for key in permutations(1:variable_count, 1)
        weight = m[key..., key...]
        if !isapprox(weight, 0.0)
            terms[key] = weight
        end
    end

    for key in permutations(1:variable_count, 2)
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
    return FactorGraph(2, variable_count, alphabet, terms)
end

function Base.convert(::Type{Array{T,2}}, gm::FactorGraph{T}) where T <: Real
    if gm.order != 2
        error("cannot convert a FactorGraph of order $(gm.order) to a matrix")
    end

    matrix = zeros(gm.variable_count, gm.variable_count)
    for (k,v) in gm
        if length(k) == 1
            matrix[k..., k...] = v
        else
            matrix[k...] = v
            r = reverse(k)
            matrix[r...] = v
        end
    end

    return matrix
end

function Base.convert(::Type{Array{T,3}}, gm::FactorGraph{T}) where T <: Real
    if gm.order != 3
        error("cannot convert a FactorGraph of order $(gm.order) to a matrix")
    end

    matrix = zeros(gm.variable_count, gm.variable_count, gm.variable_count)
    for (k,v) in gm
        if length(k) == 1 || length(k) == 2
            error("Does not support 1 or 2 body interactions")
        else
            a, b, c = k
            matrix[a, b, c] = v
            matrix[a, c, b] = v
            matrix[b, a, c] = v
            matrix[b, c, a] = v
            matrix[c, a, b] = v
            matrix[c, b, a] = v
        end
    end

    return matrix
end

Base.convert(::Type{Dict}, m::Array{T,2}) where T <: Real = convert(Dict{Tuple,T}, m)
function Base.convert(::Type{Dict{Tuple,T}}, m::Array{T,2}) where T <: Real
    @assert size(m,1) == size(m,2) #check matrix is square

    variable_count = size(m,1)

    terms = Dict{Tuple,T}()

    for key in permutations(1:variable_count, 1)
        weight = m[key..., key...]
        if !isapprox(weight, 0.0)
            terms[key] = weight
        end
    end

    for key in permutations(1:variable_count, 2, asymmetric=true)
        if key[1] != key[2]
            weight = m[key...]
            if !isapprox(weight, 0.0)
                terms[key] = weight
            end
        end
    end

    return terms
end


function Base.convert(::Type{FactorGraph}, list::Array{Any,1})
    @info "assuming spin alphabet"
    alphabet = :spin

    max_variable = 0
    max_order = 0
    terms = Dict{Tuple,Float64}()

    for item in list
        term = item["term"]
        weight = item["weight"]
        terms[tuple(term...)] = weight

        @assert minimum(term) > 0
        max_order = max(max_order, length(term))
        max_variable = max(max_variable, maximum(term))
    end

    @info "detected $(max_variable) variables with order $(max_order)"

    return FactorGraph(max_order, max_variable, alphabet, terms)
end


Base.convert(::Type{FactorGraph}, dict::Dict{Tuple,T}) where T <: Real = convert(FactorGraph{T}, dict)
function Base.convert(::Type{FactorGraph{T}}, dict::Dict{Tuple,T}) where T <: Real
    @info "assuming spin alphabet"
    alphabet = :spin

    max_variable = 0
    max_order = 0
    for (term,weight) in dict
        @assert minimum(term) > 0
        max_order = max(max_order, length(term))
        max_variable = max(max_variable, maximum(term))
    end

    @info "detected $(max_variable) variables with order $(max_order)"

    return FactorGraph(max_order, max_variable, alphabet, dict)
end


permutations(items, order::Int; asymmetric::Bool = false) = sort(permutations([], items, order, asymmetric))

function permutations(partial_perm::Array{Any,1}, items, order::Int, asymmetric::Bool)
    if order == 0
        return [tuple(partial_perm...)]
    else
        perms = []
        for item in items
            if !asymmetric && length(partial_perm) > 0
                if partial_perm[end] >= item
                    continue
                end
            end
            perm = permutations(vcat(partial_perm, item), items, order-1, asymmetric)
            append!(perms, perm)
        end
        return perms
    end
end

function completegraph(numsites::Int; J=1.::Float64)
    terms = Dict{Tuple, Float64}()

    for site1 in 1:(numsites-1)
        for site2 in (site1+1):numsites
            terms[(site1, site2)] = J
        end
    end
    FactorGraph(terms)
end

function curieweiss(N::Int, J_N::Float64, h::Float64)
    model = completegraph(N; J=J_N)

    for site in 1:N
        model.terms[(site,)] = h
    end
    return model
end

function sherringtonkirkpatrick(N::Int, J::Float64, h::Float64)
    model = completegraph(N)

    for (interacting, weight) in model.terms
        model.terms[interacting] = J/sqrt(N)*randn()
    end
    return model
end

function cubiclattice(L::Int, dim::Int; J=1.::Float64, periodic=true::Bool)
    terms = Dict{Tuple, Float64}()
    # Index terms from top left going down columns
    # with periodic boundaries connections in
    for site in 1:L^dim
        for d in 1:dim
            d_idx = fld((site-1) % L^d, L^(d-1)) + 1

            if d_idx < L
                terms[(site, site + L^(d-1))] = J
            elseif periodic
                terms[(site, site - L^(d-1)*(L-1))] = J
            end

        end
    end

    FactorGraph(terms)
end


function edwardsanderson_spinglass(L::Int, dim::Int, couplings::Symbol; J=1.::Float64, periodic=true::Bool)
    model = cubiclattice(L, dim; J=1., periodic=periodic)

    for (interacting, weight) in model.terms
        if couplings == :pm
            model.terms[interacting] = rand([-1., 1.])
        elseif couplings == :gaussian
            model.terms[interacting] = J*randn()
        end
    end
    return model
end

function ferromagnet_nbodypoisson(nbody::Int64, numsites::Int64, numterms::Int64, J::Float64)
    terms = Dict{Tuple, Float64}()
    currentterms = 0
    while currentterms < numterms
        potentialterm = Tuple(sort(rand(1:numsites, (nbody,))))
        if allunique(potentialterm) && !haskey(terms, potentialterm)
            terms[potentialterm] = J
            currentterms += 1
        end
    end
    FactorGraph(terms)
end

function spinglass_nbodypoisson(nbody::Int64, numsites::Int64, numterms::Int64, J::Float64)
    terms = Dict{Tuple, Float64}()
    currentterms = 0
    while currentterms < numterms
        potentialterm = Tuple(sort(rand(1:numsites, (nbody,))))
        if allunique(potentialterm) && !haskey(terms, potentialterm)
            terms[potentialterm] = rand([J, -J])
            currentterms += 1
        end
    end
    FactorGraph(terms)
end

"""
Generates a dictionary containing the neighboring terms
of a site.  neighbors[i] returns an array where each entry
corresponds to a term in the factor graph as ([sites], weight)
The contribution of this term to the 'energy' can be calculated as
weight*product(state[[sites]]).  Note that value assigned to key i
also contains i so that the energy contribution can be
calculated directly.
"""
function generate_neighborhoods(gm::FactorGraph{T}) where T <: Real
    neighborhoods = Dict{Int64, Array{Tuple{Array{Int64,1}, T}}}()

    for (interacting, weight) in gm.terms
        for site in interacting
            if !haskey(neighborhoods, site)
                neighborhoods[site] = []
            end

            push!(neighborhoods[site], ([interacting...], weight))
        end
    end
    return neighborhoods
end


"""
For models with 2-body interactions, this returns an array neighbors
such that neighbors[i] returns an array of site indices that interact
with site i.  If the model includes 1-body a one body field at site i,
neighbors[i] contains i.
"""
function neighbor_array(gm::FactorGraph{T}) where T <: Real
    @assert gm.order == 2

    neighbors = [[] for site in 1:gm.variable_count]
    for (sites, weight) in gm.terms
        if length(sites) == 2
            push!(neighbors[sites[1]], sites[2])
            push!(neighbors[sites[2]], sites[1])
        elseif length(sites) == 1
            push!(neighbors[sites[1]], sites[1])
        end
    end
    [sort(l) for l in neighbors]
end

function find_term(gm::FactorGraph{T}, term::Tuple{Int64,Int64}) where T <: Real

    (i, j) = term
    if i == j
        return gm.terms[(i,)]
    else
        try
            return gm.terms[(i, j)]
        catch KeyError
            return gm.terms[(j, i)]
        end
    end

end


"""
Creates an array for each spin containing the indices of its neighbors.
This will be used in sampling to extract the relevant configurations
in the neighborhood of each spin
"""
function neighborarray(gm::FactorGraph{T}) where T <: Real

    neighbors = convert(Vector{Vector{Int64}}, [[] for spin in 1:gm.variable_count])
    for (term_spins, w) in gm.terms
        for spin in term_spins
            for neighborspin in term_spins
                if !any(neighbors[spin] .== neighborspin)
                    push!(neighbors[spin], neighborspin)
                end
            end
        end
    end
    for arr in neighbors
        sort!(arr)
    end
    neighbors
end

"""
Builds a version of neighborhoods where indices refer to the elements of
neighbor array.  This way, after binning the spin products can be efficiently
constructed.
"""
function localterms(gm::FactorGraph{T}, neighborarray::Array{Array{Int64,1},1}) where T <: Real

    localterms = Dict{Int64, Array{Tuple{Array{Int64,1}, T}}}()
    for spin in 1:gm.variable_count
        localterms[spin] = []
    end
    for (term_spins, w) in gm.terms
        for spin in term_spins
            localarray = neighborarray[spin]
            term = []
            for neighborspin in term_spins
                push!(term, findfirst(isequal(neighborspin), localarray))
            end
            push!(localterms[spin], (term, w))
        end
    end
    localterms
end
