import DataStructures: OrderedDict
using LaTeXStrings

export Parameter, Parameters, num_varied, is_varied, to_vecs, set_values!, add_prior!, compute_prior_logprob, index_from_pname

"""
Type for a RadVel-like Bayesian Parameter.
# Fields:
- `name::Union{String, Nothing}` The name of the parameter.
- `value::Float64` The value of the parameter.
- `vary::Bool` Whether or not to to fit for this parameter.
- `priors::Vector{Priors.Prior}` A `Vector` of priors for this parameter.
- `latex_str::Union{LaTeXString, Nothing}` A LaTeX-formatted string, useful for plotting.
"""
mutable struct Parameter
    name::String
    value::Float64
    vary::Bool
    priors::Vector{Priors.Prior}
    latex_str::LaTeXString
end

"""
Container for multiple Bayesian parameters through a dictionary-like API.
"""
struct Parameters
    dict::OrderedDict{String, Parameter}
end

function index_from_pname(pars::Parameters, pname::String; rel_vary=false)
    if rel_vary
        ii = 1
        for _pname ∈ keys(pars)
            if _pname == pname
                return ii
            elseif pars[_pname].vary
                ii += 1
            end
        end
    else
        for (i, _pname) ∈ enumerate(values(pars))
            if _pname == pname
                return i
            end
        end
    end
end

"""
    Parameter(;name=nothing, value::Real, vary::Bool=true, priors=nothing, latex_str=nothing)
Construct a Parameter object. If using the dictionary interface, the name will automatically be passed.
"""
Parameter(;name="", value::Real, vary::Bool=true, priors=Priors.Prior[], latex_str=L"") = Parameter(name, value, vary, priors, latex_str)

"""
    Parameters()
Construct an empty Parameters object.
"""
function Parameters()
    return Parameters(OrderedDict{String, Parameter}())
end

function add_prior!(par::Parameter, prior::Priors.Prior)
    push!(par.priors, prior)
end

Base.length(pars::Parameters) = length(pars.dict)
Base.merge!(pars::Parameters, pars2::Parameters) = merge!(pars.dict, pars2.dict)
Base.getindex(pars::Parameters, key::String) = getindex(pars.dict, key)
Base.firstindex(pars::Parameters) = firstindex(pars.dict)
Base.lastindex(pars::Parameters) = lastindex(pars.dict)
Base.iterate(pars::Parameters) = iterate(pars.dict)
Base.keys(pars::Parameters) = keys(pars.dict)
Base.values(pars::Parameters) = values(pars.dict)

function Base.setindex!(pars::Parameters, par::Parameter, key::String)
    if par.name == ""
        par.name = key
    end
    @assert par.name == key
    setindex!(pars.dict, par, key)
end

function Base.show(io::IO, par::Parameter)
    if par.vary
        println(" $(par.name) | Value = $(par.value)")
        for prior ∈ par.priors
            println(io, "    $(prior)")
        end
    else
        println(io, " $(par.name) | Value = $(par.value) 🔒")
    end
end

function Base.show(io::IO, pars::Parameters)
    for par ∈ values(pars)
        show(io, par)
    end
end

function num_varied(pars::Parameters)
    n = 0
    for par ∈ values(pars)
        if is_varied(par)
            n += 1
        end
    end
    return n
end

function is_varied(par::Parameter)
    return par.vary
end

function to_vecs(pars::Parameters)
    names = String[par.name for par ∈ values(pars)]
    _values = Float64[par.value for par ∈ values(pars)]
    vary = BitVector([is_varied(par) for par ∈ values(pars)])
    latex_str = String[par.latex_str for par ∈ values(pars)]
    out = (;names=names, values=_values, vary=vary, latex_str=latex_str)
    return out
end

function set_values!(pars::Parameters, x::Vector)
    for (i, par) ∈ enumerate(values(pars))
        par.value = x[i]
    end
end

function compute_prior_logprob(par::Parameter)
    lnL = 0
    for prior in par.priors
        lnL += logprob(prior, par.value)
        if !isfinite(lnL)
            return -Inf
        end
    end
    return lnL
end

function compute_prior_logprob(pars::Parameters)
    lnL = 0.0
    for par ∈ values(pars)
        if par.vary
            lnL += compute_prior_logprob(par)
        end
        if !isfinite(lnL)
            return -Inf
        end
    end
    return lnL
end

function get_scale(par::Parameter)
    if length(par.priors) == 0
        scale = abs(par.value) / 100
        if scale == 0
            return 1
        else
            return scale
        end
    end
    for prior ∈ par.priors
        if prior isa Priors.Gaussian
            return prior.σ / 2
        end
    end
    for prior ∈ par.priors
        if prior isa Priors.Uniform
            dx1 = abs(prior.upper_bound - par.value)
            dx2 = abs(par.value - prior.lower_bound)
            scale = min(dx1, dx2) / 10
            if scale == 0
                return 1
            else
                return scale
            end
        end
    end
    scale = abs(par.value) / 100
    if scale == 0
        return 1
    else
        return scale
    end
end