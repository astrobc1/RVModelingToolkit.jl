using Infiltrator, PyPlot
using DataStructures

export RVModel, build, build_planet, build_planets, build_global_trend, build_trend_zero, true_anomaly, tp_to_tc, tc_to_tp, disable_planet_parameters!, solve_kepler, solve_kepler_all_times, get_phases

mutable struct RVModel{T}
    planets::OrderedDict{Int, OrbitBasis}
    trend_poly_deg::T
    t0::Float64
end

"""
    RVModel(planets::Union{OrderedDict{Int, OrbitBasis}, Nothing}=nothing, trend_poly_deg::Int=0; t0=2400000.0)
Construct an RVModel object to model RVs with one or multiple Keplerian orbits.
"""
function RVModel(planets::Union{OrderedDict{Int, OrbitBasis}, Nothing}=nothing, trend_poly_deg::Int=0; t0=2400000.0)
    if isnothing(planets)
        planets = Dict{Int, OrbitBasis}()
    end
    return RVModel(planets, trend_poly_deg, t0)
end

function RVModel(planets::Dict{Int, OrbitBasis}, trend_poly_deg::Int=0; t0=2400000.0)
    planets = OrderedDict{Int, OrbitBasis}(sort(collect(planets), by = x->x[1]))
    return RVModel(planets, trend_poly_deg, t0)
end

"""
    build(m::RVModel, pars::Parameters, t::AbstractVector{<:Real})
Build the Keplerian + trend model, not including zero point corrections (thus only degree > 0 corrections).
"""
function build(m::RVModel, pars::Parameters, t::AbstractVector{<:Real})
    y = build_planets(m, pars, t)
    if m.trend_poly_deg > 0
        y .+= build_global_trend(m, pars, t)
    end
    return y
end

"""
    build_planet(m::RVModel, pars::Parameters, t::AbstractVector{<:Real}, planet_index::Int)
    Build the model for the single planet `planet_index`.
"""
function build_planet(m::RVModel, pars::Parameters, t::AbstractVector{<:Real}, planet_index::Int)
    planet_pars = convert_basis(pars, planet_index, m.planets[planet_index], StandardOrbitBasis)
    vels = build_planet(t, planet_pars...)
    return vels
end

"""
    build_planets(m::RVModel, pars::Parameters, t::AbstractVector{<:Real})
Build and addd all planet models together.
"""
function build_planets(m::RVModel, pars::Parameters, t::AbstractVector{<:Real})
    y = zeros(length(t))
    for planet_index ∈ keys(m.planets)
        y .+= build_planet(m, pars, t, planet_index)
    end
    return y
end

"""
    build_trend_zero(m::RVModel, data::CompositeRVData, pars::Parameters, t::AbstractVector{<:Real}; instname::Union{Int, Nothing}=nothing)
Build the zero point model for each spectrograph (gamma parameters).
"""
function build_trend_zero(m::RVModel, data::CompositeRVData, pars::Parameters, t::AbstractVector{<:Real}; instname::Union{Int, Nothing}=nothing)
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

"""
    build_global_trend(m::RVModel, pars::Parameters, t::AbstractVector{<:Real})
Build the global trend model, not including the zero point gammas of each spectrograph (gamma dot parameters).
"""
function build_global_trend(m::RVModel, pars::Parameters, t::AbstractVector{<:Real})
        
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


"""
    disable_planet_parameters!(pars::Parameters, planets::OrderedDict{Int, OrbitBasis}, planet_index::Int)
Disables the parameters corresponding to `planet_index`.
"""
function disable_planet_parameters!(pars::Parameters, planets::OrderedDict{Int, OrbitBasis}, planet_index::Int)
    for par ∈ values(pars)
        for planet_par_name ∈ parameters(planets[planet_index], planet_index)
            if par.name == planet_par_name
                pars[par.name].vary = false
            end
        end
    end
end

function get_phases(t, per, tc)
    ϕ = @. mod(t - tc - per / 2, per) / per
    return ϕ
end

function period_from_transit_duration(mstar, rstar, tdur)
    mstar_mks = mstar * MSUN_KG
    rstar_mks = rstar * RSUN_M
    tdur_mks = tdur * 86400
    per = π * G_MKS * mstar_mks * tdur_mks^3 / (4 * rstar_mks^3)
    return per / 86400
end

"""
    build_planet(t::Union{AbstractVector{<:Real}, <:Real}, per::Real, tp::Real, ecc::Real, w::Real, k::Real)
Builds the Keplerian model for a single planet on the `StandardOrbitBasis`.
"""
function build_planet(t::Union{AbstractVector{<:Real}, <:Real}, per::Real, tp::Real, ecc::Real, w::Real, k::Real)
    
    # Circular orbit
    if ecc == 0.0
        m = @. 2 * π * (((t - tp) / per) - floor((t - tp) / per))
        return @. k * cos(m + w)
    end

    # Period must be positive
    if per <= 0
        per = 1E-6
    end
        
    # Force circular orbit if ecc is negative
    if ecc < 0
        ecc = 0
        m = @. 2 * π * (((t - tp) / per) - floor((t - tp) / per))
        return @. k * cos(m + w)
    end
    
    # Force bounded orbit if ecc almost blows up
    if ecc > 0.99
        ecc = 0.99
    end
        
    # Calculate the eccentric anomaly (ea) from the mean anomaly (ma). Requires solving kepler's eq. if ecc>0.
    ta = true_anomaly(t, tp, per, ecc)
    rv = @. k * (cos(ta + w) + ecc * cos(w))

    # Return rv
    return rv
end


"""
    solve_kepler_all_times(mas::AbstractVector{<:Real}, ecc::Real)
Wrapper to solve Kepler's equation for multiple observation times (multiple mean anomalies, `mas`) for a single value of eccentricity `ecc`.
"""
function solve_kepler_all_times(mas::AbstractVector{<:Real}, ecc::Real)
    eas = zeros(length(mas))
    for i=1:length(mas)
        eas[i] = solve_kepler(mas[i], ecc)
    end
    return eas
end

"""
    solve_kepler(ma::Float64, ecc::Float64)::Float64
Solves Kepler's equation with a higher order Newton method. This method is largely based on the Kepler's equation solver in RadVel.
"""
function solve_kepler(ma::Float64, ecc::Float64)::Float64

    CONV = 1E-10
    K = 0.85
    MAX_ITERS = 200
    
    # First guess for ea
    ea = ma + sign(sin(ma)) * K * ecc
    fi = ea - ecc * sin(ea) - ma

    # Updated ea
    ea_new = 0.0

    # Counter
    count::Int32 = 0
    
    # Break when converged
    while true & (count < MAX_ITERS)
        
        # Increase counter
        count += 1
        
        # Update ea
        fip = 1.0 - ecc * cos(ea)
        fipp = ecc * sin(ea)
        fippp = 1.0 - fip
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        ea_new = ea + d3
        
        # Check convergence
        fi = ea_new - ecc * sin(ea_new) - ma
        if fi < CONV
            break
        end
        ea = ea_new
    end

    return ea_new
end

"""
    true_anomaly(t, tp, per, ecc)
Computes the true anomaly by solving Kepler's equation.
"""
function true_anomaly(t, tp, per, ecc)
    m = @. 2 * π * (((t - tp) / per) - floor((t - tp) / per))
    ea = solve_kepler_all_times(m, ecc)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    ta = @. 2.0 * atan((n1 / n2)^0.5 * tan(ea / 2.0))
    return ta
end

"""
    tc_to_tp(tc, per, ecc, w)
Converts the time of conjunction to the time of periastron.
"""
function tc_to_tp(tc, per, ecc, w)
    
    # If ecc >= 1, no tp exists
    if ecc >= 1
        return tc
    end

    f = π / 2 - w
    ee = @. 2 * atan(tan(f / 2) * sqrt((1 - ecc) / (1 + ecc)))
    tp = @. tc - per / (2 * π) * (ee - ecc * sin(ee))

    return tp
end


"""
    tp_to_tc(tp, per, ecc, w)
Converts the time of periastron to the time of conjunction.
"""
function tp_to_tc(tp, per, ecc, w)
    
    # If ecc >= 1, no tc exists.
    if ecc >= 1
        return tp
    end

    f = π / 2 - w                                         # true anomaly during transit
    ee = @. 2 * atan(tan( f / 2) * sqrt((1 - ecc) / (1 + ecc)))  # eccentric anomaly

    tc = @. tp + per / (2 * π) * (ee - ecc * sin(ee))         # time of conjunction

    return tc
end