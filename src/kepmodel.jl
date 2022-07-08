using Infiltrator, PyPlot
using DataStructures

export RVModel, build, build_planet, build_planets, build_global_trend, build_trend_zero

mutable struct RVModel{T}
    planets::OrderedDict{Int, OrbitBasis}
    trend_poly_deg::T
    t0::Float64
end

function RVModel(planets=nothing, trend_poly_deg=0; t0=2400000.0)
    if isnothing(planets)
        planets = Dict{Int, OrbitBasis}()
    end
    return RVModel(planets, trend_poly_deg, t0)
end

function build(m::RVModel, pars, t)
    y = build_planets(m, pars, t)
    y .+= build_global_trend(m, pars, t)
    return y
end

function build_planet(m::RVModel, pars::Parameters, t, planet_index)
    planet_pars = convert_basis(pars, planet_index, m.planets[planet_index], StandardOrbitBasis)
    vels = Maths.planet_signal(t, planet_pars...)
    return vels
end

function build_planets(m::RVModel, pars, t)
    y = zeros(length(t))
    for planet_index ∈ keys(m.planets)
        y .+= build_planet(m, pars, t, planet_index)
    end
    return y
end

    
function build_trend_zero(m::RVModel, data, pars, t; instname=nothing)
    trend_zero = zeros(length(t))
    if !isnothing(m.trend_poly_deg)
        if !isnothing(instname)
            pname = "gamma_$(instname)"
            trend_zero .= pars[pname].value
        else
            for instname in keys(data)
                pname = "gamma_$(instname)"
                inds = data.indices[instname]
                trend_zero[inds] .= pars[pname].value
            end
        end
    end
    return trend_zero
end


function build_global_trend(m::RVModel, pars, t)
        
    # Init trend
    trend = zeros(length(t))
            
    # Build trend
    if m.trend_poly_deg > 0
        for i=1:m.trend_poly_deg
            d = repeat("d", i)
            pname = "gamma_$(d)ot"
            trend .+= pars[pname].value .* (t .- m.t0).^i
        end
    end
    return trend
end


function disable_planet_parameters!(pars::Parameters, planets::OrderedDict{Int, OrbitBasis}, planet_index::Int)
    for par ∈ values(pars)
        for planet_par_name ∈ parameters(planets[planet_index], planet_index)
            if par.name == planet_par_name
                pars[par.name].vary = false
            end
        end
    end
end