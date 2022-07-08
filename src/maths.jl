module Maths

export compute_mass, compute_sa, bin_phased_rvs, planet_signal

using Statistics
using Infiltrator

# CONSTANTS
const MASS_JUPITER_EARTH_UNITS = 317.82838 # mass of jupiter in earth masses
const MASS_EARTH_GRAMS = 5.972181578208704E27 # mass of earth in grams
const RADIUS_EARTH_CM = 6.371009E8 # radius of earth in cm
const YEAR_EARTH_DAYS = 365.25 # one year for earth in days
const K_JUPITER_P_EARTH = 28.4329 # the semi-amplitude of jupiter for a one year orbit
const MSUN_KG = 1.988435E30 # The mass of the sun in kg
const RSUN_M = 6.957E8 # Radius of sun in meters
const G_MKS = 6.6743E-11 # The Newtonian gravitational constant in mks units
const AU_M = 1.496E11 # 1 AU in meters

function compute_mass(per, ecc, k, mstar)
    return k * sqrt(1 - ecc^2) / K_JUPITER_P_EARTH * (per / YEAR_EARTH_DAYS)^(1 / 3) * mstar^(2 / 3) * MASS_JUPITER_EARTH_UNITS
end

function compute_sa(per, mstar)
    return (G_MKS / (4 * π^2))^(1 / 3) * (mstar * MSUN_KG)^(1 / 3) * (per * 86400)^(2 / 3) / AU_M
end

function compute_sa_deriv_mstar(per, mstar)
    return (G_MKS / (4 * π^2))^(1 / 3) * (mstar * MSUN_KG)^(-2 / 3) / 3 * (per * 86400)^(2 / 3) * (MSUN_KG / AU_M)
end

function get_phases(t, per, tc)
    ϕ = @. mod(t - tc - per / 2, per) / per
    return ϕ
end

function weighted_mean(x, w)
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    if length(good) > 0
        xx = @view x[good]
        ww = @view w[good]
        return sum(xx .* ww) / sum(ww)
    else
        return NaN
    end
end

function weighted_stddev(x, w)
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    xx = x[good]
    ww = w[good]
    ww ./= sum(ww)
    μ = weighted_mean(xx, ww)
    dev = xx .- μ
    bias_estimator = 1.0 - sum(ww.^2)
    var = sum(dev .^2 .* ww) / bias_estimator
    return sqrt(var)
end

function bin_phased_rvs(phases, rvs, unc; nbins=10)

    binned_phases = fill(NaN, nbins)
    binned_rvs = fill(NaN, nbins)
    binned_unc = fill(NaN, nbins)
    bins = collect(range(0, 1, length=nbins+1))
    for i=1:nbins
        inds = findall((phases .>= bins[i]) .&& (phases .< bins[i + 1]))
        n = length(inds)
        if n == 0
            continue
        end
        w = 1 ./ unc[inds].^2
        w ./= sum(w)
        binned_phases[i] = @views mean(phases[inds])
        binned_rvs[i] = @views weighted_mean(rvs[inds], w)
        if length(inds) == 1
            binned_unc[i] = @views rvs[inds[0]]
        else
            binned_unc[i] = @views weighted_stddev(rvs[inds], w) / sqrt(n)
        end
    end

    return binned_phases, binned_rvs, binned_unc
end

function compute_mass_deriv_mstar(per, ecc, k, mstar)
    a = k * sqrt(1 - ecc^2) / K_JUPITER_P_EARTH * (per / YEAR_EARTH_DAYS)^(1 / 3) * MASS_JUPITER_EARTH_UNITS
    dMp_dMstar = (2 / 3) * a * mstar^(-1 / 3)
    return dMp_dMstar
end

function compute_density(mplanet, rplanet)
    mplanet_grams = mplanet * MASS_EARTH_GRAMS
    rplanet_cm = rplanet * RADIUS_EARTH_CM
    rho_cgs = (3 * mplanet_grams) / (4 * π * rplanet_cm^3)
    return rho_cgs
end

function compute_density_deriv_rplanet(rplanet, mplanet)
    mplanet_grams = mplanet * MASS_EARTH_GRAMS
    rplanet_cm = rplanet * RADIUS_EARTH_CM
    d_rho_d_rplanet = (9 * mplanet_grams) / (4 * π * rplanet_cm^4)
    return d_rho_d_rplanet
end

function solve_kepler_all_times(mas, ecc)
    eas = zeros(length(mas))
    for i=1:length(mas)
        eas[i] = solve_kepler(mas[i], ecc)
    end
    return eas
end

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


function true_anomaly(t, tp, per, ecc)
    m = @. 2 * π * (((t - tp) / per) - floor((t - tp) / per))
    ea = solve_kepler_all_times(m, ecc)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    ta = @. 2.0 * atan((n1 / n2)^0.5 * tan(ea / 2.0))
    return ta
end


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

function planet_signal(t, per, tp, ecc, w, k)
    
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

function period_from_transit_duration(mstar, rstar, tdur)
    mstar_mks = mstar * MSUN_KG
    rstar_mks = rstar * RSUN_M
    tdur_mks = tdur * 86400
    per = π * G_MKS * mstar_mks * tdur_mks^3 / (4 * rstar_mks^3)
    return per / 86400
end

end