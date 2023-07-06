using DataStructures
using ForwardDiff

export planet_stats, print_pstats

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

"""
    planet_stats(post::RVPosterior, mcmc_result; radii=nothing)
For each planet, computes the quantities Mp*sini  [Earth masses], radius [Earth radii] (from C-K relation or just returns the provided radii), densities [g/cm^3],  Semi-major axis [AU]. Returns a dictionary of NamedTuples
"""
function planet_stats(post::RVPosterior, mcmc_result; mstar, radii=nothing, output_path=nothing, display=true)
    planets = first(post)[2].model.planets
    masses = compute_masses(post, mcmc_result, mstar)
    radii = compute_radii(post, mcmc_result, masses, mstar, radii)
    ρs = compute_densities(post, mcmc_result, radii, mstar)
    smas = compute_smas(post, mcmc_result, mstar)
    pstats = OrderedDict{Int, NamedTuple}()
    for planet_index ∈ keys(planets)
        pstats[planet_index] = (;mass=masses[planet_index], radius=radii[planet_index], ρ=ρs[planet_index], sma=smas[planet_index])
    end
    if !isnothing(output_path)
        jldsave("$(output_path)planet_stats.jld"; pstats)
    end
    if display
        print_pstats(pstats)
    end
    return pstats
end

function compute_masses(post, mcmc_result, mstar)
    planets = first(post)[2].model.planets
    masses = OrderedDict{Int, Any}() # In earth masses
    for planet_index ∈ keys(planets)
        mass_chain = get_mass_chain(mcmc_result, planets, planet_index, mstar)
        r = chain_uncertainty(mass_chain)
        per, tp, ecc, w, k = convert_basis(mcmc_result.pmed, planet_index, planets[planet_index], StandardOrbitBasis)
        #dM_dMstar = compute_mass_deriv_mstar(per, ecc, k, mstar[2])
        unc_low = sqrt(r.minus^2 + ForwardDiff.derivative((_mstar) -> compute_mass(per, ecc, k, _mstar), mstar.value)^2 * mstar.minus^2)
        unc_high = sqrt(r.plus^2 + ForwardDiff.derivative((_mstar) -> compute_mass(per, ecc, k, _mstar), mstar.value)^2 * mstar.plus^2)
        masses[planet_index] = (;value=r.value, minus=unc_low, plus=unc_high)
    end
    return masses
end

function get_mass_chain(mcmc_result, planets, planet_index, mstar)
    mdist = []
    pars = deepcopy(mcmc_result.pmed)
    for i=1:mcmc_result.n_steps
        for pname ∈ parameters(planets[planet_index], planet_index)
            if pars[pname].vary
                ii = index_from_pname(pars, pname, rel_vary=true)
                pars[pname].value = mcmc_result.chains[i, ii]
            end
        end
        per, tp, ecc, w, k = convert_basis(pars, planet_index, planets[planet_index], StandardOrbitBasis)
        push!(mdist, compute_mass(per, ecc, k, mstar.value))
    end
    return mdist
end

function get_radius_chain(mcmc_result, planets, planet_index, mstar)
    rdist = []
    pars = deepcopy(mcmc_result.pmed)
    for i=1:mcmc_result.n_steps
        for pname ∈ parameters(planets[planet_index], planet_index)
            if pars[pname].vary
                ii = index_from_pname(pars, pname, rel_vary=true)
                pars[pname].value = mcmc_result.chains[i, ii]
            end
        end
        per, tp, ecc, w, k = convert_basis(pars, planet_index, planets[planet_index], StandardOrbitBasis)
        m = compute_mass(per, ecc, k, mstar.value)
        push!(rdist, mass2radius_ck(m))
    end
    return rdist
end

function get_density_chain(mcmc_result, planets, planet_index, radii, mstar)
    ρdist = []
    pars = deepcopy(mcmc_result.pmed)
    for i=1:mcmc_result.n_steps
        for pname ∈ parameters(planets[planet_index], planet_index)
            if pars[pname].vary
                ii = index_from_pname(pars, pname, rel_vary=true)
                pars[pname].value = mcmc_result.chains[i, ii]
            end
        end
        per, tp, ecc, w, k = convert_basis(pars, planet_index, planets[planet_index], StandardOrbitBasis)
        push!(ρdist, compute_density(per, ecc, k, mstar.value, radii[planet_index].value))
    end
    return ρdist
end

function get_sma_chain(mcmc_result, planets, planet_index, mstar)
    smadist = []
    pars = deepcopy(mcmc_result.pmed)
    for i=1:mcmc_result.n_steps
        for pname ∈ parameters(planets[planet_index], planet_index)
            if pars[pname].vary
                ii = index_from_pname(pars, pname, rel_vary=true)
                pars[pname].value = mcmc_result.chains[i, ii]
            end
        end
        per, tp, ecc, w, k = convert_basis(pars, planet_index, planets[planet_index], StandardOrbitBasis)
        push!(smadist, compute_sma(per, mstar.value))
    end
    return smadist
end

function compute_radii(post, mcmc_result, masses, mstar, radii=nothing)
    planets = first(post)[2].model.planets
    if !isnothing(radii)
        radii = deepcopy(radii)
    else
        radii = Dict{Int, Any}()
    end
    for planet_index ∈ keys(planets)
        if planet_index ∉ keys(radii)
            radius_chain = get_radius_chain(mcmc_result, planets, planet_index, mstar)
            r = chain_uncertainty(radius_chain)
            per, tp, ecc, w, k = convert_basis(mcmc_result.pmed, planet_index, planets[planet_index], StandardOrbitBasis)
            # Need dRp_dMstar = dRp/dMp * dMp/dMstar
            dMplanet_dMstar = ForwardDiff.derivative((_mstar) -> compute_mass(per, ecc, k, _mstar), mstar.value)
            mplanet = compute_mass(per, ecc, k, mstar.value)
            dRp_dMp = ForwardDiff.derivative((_mplanet) -> mass2radius_ck(_mplanet), mplanet)
            dRp_dMstar = dRp_dMp * dMplanet_dMstar
            unc_low = sqrt(r.minus^2 + dRp_dMstar^2 * mstar.value)
            unc_high = abs(r.plus^2 + dRp_dMstar^2 * mstar.value)
            radii[planet_index] = (;value=r.value, minus=unc_low, plus=unc_high)
        end
    end
    return radii
end

# M is in Earth Masses
# Radius returned in Earth Radii
function mass2radius_ck(mass::Real)
	MJ = 317.828133 # in earth masses
	RJ = 11.209 # in earth radii
	if mass <= 2.04
		return 1.008 * mass^0.279
    elseif mass > 2.04 && mass <= 0.414*MJ
		return 0.80811 * mass^0.589
	else
		return 17.739 * mass^-0.044
    end
end

function compute_smas(post, mcmc_result, mstar)
    planets = first(post)[2].model.planets
    smas = OrderedDict{Int, Any}() # In earth masses
    for planet_index ∈ keys(planets)
        sma_chain = get_sma_chain(mcmc_result, planets, planet_index, mstar)
        r = chain_uncertainty(sma_chain)
        # Need da_dMstar
        per, tp, ecc, w, k = convert_basis(mcmc_result.pmed, planet_index, planets[planet_index], StandardOrbitBasis)
        da_dMstar = ForwardDiff.derivative((_mstar) -> compute_sma(per, _mstar), mstar.value)
        unc_low = sqrt(r.minus^2 + da_dMstar^2 * mstar.minus^2)
        unc_high = sqrt(r.plus^2 + da_dMstar^2 * mstar.plus^2)
        smas[planet_index] = (;value=r.value, minus=unc_low, plus=unc_high)
    end
    return smas
end

function compute_densities(post, mcmc_result, radii, mstar)
    planets = first(post)[2].model.planets
    ρs = OrderedDict{Int, Any}() # In g/cm^3
    for planet_index ∈ keys(planets)
        ρ_chain = get_density_chain(mcmc_result, planets, planet_index, radii, mstar)
        r = chain_uncertainty(ρ_chain)
        # Need dρ/dMstar = dρ/dMplanet * dMplanet/Mstar
        per, tp, ecc, w, k = convert_basis(mcmc_result.pmed, planet_index, planets[planet_index], StandardOrbitBasis)
        dMplanet_dMstar = ForwardDiff.derivative((_mstar) -> compute_mass(per, ecc, k, _mstar), mstar.value)
        mplanet = compute_mass(per, ecc, k, mstar.value)
        dρ_dMplanet = ForwardDiff.derivative((_mplanet) -> compute_density(_mplanet, radii[planet_index].value), mstar.value)
        dρ_dMstar = dρ_dMplanet * dMplanet_dMstar
        unc_low = sqrt(r.minus^2 + dρ_dMstar^2 * mstar.minus^2)
        unc_high = sqrt(r.plus^2 + dρ_dMstar^2 * mstar.plus^2)
        ρs[planet_index] = (;value=r.value, minus=unc_low, plus=unc_high)
    end
    return ρs
end

# per [days], ecc [unitless], k [m/s], mstar [solar masses] -> mplanet [earth masses]
function compute_mass(per::Real, ecc::Real, k::Real, mstar::Real)
    return k * sqrt(1 - ecc^2) / K_JUPITER_P_EARTH * (per / YEAR_EARTH_DAYS)^(1 / 3) * mstar^(2 / 3) * MASS_JUPITER_EARTH_UNITS
end

# per [days], mstar [solar masses] -> SMA [AU]
function compute_sma(per::Real, mstar::Real)
    return (G_MKS / (4 * π^2))^(1 / 3) * (mstar * MSUN_KG)^(1 / 3) * (per * 86400)^(2 / 3) / AU_M
end

# per [days], ecc [unitless], k [m/s], mstar [solar masses], rplanet [earth radii] -> rho [g/cm^3]
function compute_density(per::Real, ecc::Real, k::Real, mstar::Real, rplanet::Real)
    mplanet = compute_mass(per, ecc, k, mstar)
    mplanet_grams = mplanet * MASS_EARTH_GRAMS
    rplanet_cm = rplanet * RADIUS_EARTH_CM
    ρcgs = compute_density(mplanet_grams, rplanet_cm)
    return ρcgs
end

# arbitrary units
function compute_density(mplanet::Real, rplanet::Real)
    ρ = (3 * mplanet) / (4 * π * rplanet^3)
    return ρ
end

function print_pstats(pstats; nm=3, nr=3, nρ=3, nsma=6)
    println("Planet  |  Mass [Earth masses]  |  Radius [Earth radii]  |     ρ [g/cm^3]     |    SMA [AU]")
    for (i, p) ∈ zip(keys(pstats), values(pstats))
        line = "$(ALPHABET[i]) | $(stat_str(p.mass, nm)) | $(stat_str(p.radius, nr)) | $(stat_str(p.ρ, nρ)) | $(stat_str(p.sma, nsma))"
        println(line)
    end
end

function stat_str(r::NamedTuple, n)
    return "$(round(r.value, digits=n)) + $(round(r.plus, digits=n)) - $(round(r.minus, digits=n))"
end