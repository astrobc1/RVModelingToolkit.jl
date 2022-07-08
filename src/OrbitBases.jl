module OrbitBases

using Infiltrator
using RVModelingToolkit

export OrbitBasis, StandardOrbitBasis, TCOrbitBasis, convert_basis, parameters

abstract type OrbitBasis end
parameters(b::OrbitBasis, i::Int) = ["$(pname)i" for pname ∈ parameters(b)]

struct StandardOrbitBasis <: OrbitBasis end
parameters(::StandardOrbitBasis) = ["per", "tp", "ecc", "w", "k"]

struct TCOrbitBasis <: OrbitBasis end
parameters(::TCOrbitBasis) = ["per", "tc", "ecc", "w", "k"]

function convert_basis(pars::Parameters, planet_index::Int, ::StandardOrbitBasis, ::Type{TCOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tp = pars["tp$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["ω$ii"].value
    tc = Maths.tp_to_tc(tp, per, ecc, ω)
    k = pars["k$ii"].value
    pars_out = [per, tc, ecc, ω, k]
    return pars_out
end

function convert_basis(pars::Parameters, planet_index::Int, ::TCOrbitBasis, ::Type{StandardOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tc = pars["tc$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["ω$ii"].value
    tp = Maths.tc_to_tp(tc, per, ecc, ω)
    k = pars["k$ii"].value
    pars_out = [per, tp, ecc, ω, k]
    return pars_out
end

function convert_basis(pars::Parameters, planet_index::Int, ::StandardOrbitBasis, ::Type{StandardOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tp = pars["tp$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["ω$ii"].value
    k = pars["k$ii"].value
    pars_out = [per, tp, ecc, ω, k]
    return pars_out
end

function convert_basis(pars::Parameters, planet_index::Int, ::TCOrbitBasis, ::Type{TCOrbitBasis})
    ii = string(planet_index)
    per = pars["per$ii"].value
    tc = pars["tc$ii"].value
    ecc = pars["ecc$ii"].value
    ω = pars["ω$ii"].value
    k = pars["k$ii"].value
    pars_out = [per, tc, ecc, ω, k]
    return pars_out
end

end