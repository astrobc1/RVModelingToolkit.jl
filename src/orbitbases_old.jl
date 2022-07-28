using RVModelingToolkit

export OrbitBasis, StandardOrbitBasis, TCOrbitBasis, convert_basis, parameters

"""
    Base type for a Keplerian orbit basis through RVs.
"""
abstract type OrbitBasis end

"""
    parameters(b::OrbitBasis, i::Int)
Returns a vector of parameter names for this basis and `planet_index`.
"""
parameters(b::OrbitBasis, i::Int) = ["$(pname)$(i)" for pname ∈ parameters(b)]

"""
The standard RadVel orbit basis
Orbital period (per; days)
Time of periastron (tp; days/BJD)
Orbital eccentricity (ecc; unitless)
Argument of periastron (w; radians)
RV semi-amplitude (k; m/s)
"""
struct StandardOrbitBasis <: OrbitBasis end
parameters(::StandardOrbitBasis) = ["per", "tp", "ecc", "w", "k"]

"""
A common orbit basis for transiting planets.
Orbital period (per; days)
Time of conjunction/transit (tc; days/BJD)
Orbital eccentricity (ecc; unitless)
Argument of periastron (w; radians)
RV semi-amplitude (k; m/s)
"""
struct TCOrbitBasis <: OrbitBasis end
parameters(::TCOrbitBasis) = ["per", "tc", "ecc", "w", "k"]



"""
    Convert from one orbit basis to another.
"""
function convert_basis end

"""
    convert_basis(pars::Parameters, planet_index::Int, ::StandardOrbitBasis, ::Type{TCOrbitBasis})
"""
function convert_basis(pars::Parameters, planet_index::Int, ::StandardOrbitBasis, ::Type{TCOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tp = pars["tp$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["w$ii"].value
    tc = tp_to_tc(tp, per, ecc, ω)
    k = pars["k$ii"].value
    pars_out = [per, tc, ecc, ω, k]
    return pars_out
end

"""
    convert_basis(pars::Parameters, planet_index::Int, ::TCOrbitBasis, ::Type{StandardOrbitBasis})
"""
function convert_basis(pars::Parameters, planet_index::Int, ::TCOrbitBasis, ::Type{StandardOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tc = pars["tc$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["w$ii"].value
    tp = tc_to_tp(tc, per, ecc, ω)
    k = pars["k$ii"].value
    pars_out = [per, tp, ecc, ω, k]
    return pars_out
end

"""
    convert_basis(pars::Parameters, planet_index::Int, ::StandardOrbitBasis, ::Type{StandardOrbitBasis})
"""
function convert_basis(pars::Parameters, planet_index::Int, ::StandardOrbitBasis, ::Type{StandardOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tp = pars["tp$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["w$ii"].value
    k = pars["k$ii"].value
    pars_out = [per, tp, ecc, ω, k]
    return pars_out
end

"""
    convert_basis(pars::Parameters, planet_index::Int, ::TCOrbitBasis, ::Type{TCOrbitBasis})
"""
function convert_basis(pars::Parameters, planet_index::Int, ::TCOrbitBasis, ::Type{TCOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tc = pars["tc$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["w$ii"].value
    k = pars["k$ii"].value
    pars_out = [per, tc, ecc, ω, k]
    return pars_out
end