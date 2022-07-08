using Combinatorics

using Infiltrator

export model_comparison

function model_comparison(post, p0::Parameters)
    p00 = deepcopy(p0)
    post0 = deepcopy(post)
    like0 = first(post)[2]
    model0 = like0.model
    planets0 = copy(model0.planets)
    planet_perms = get_planet_perms(planets0)
    println("Running model comparison with $(length(planet_perms)) different models ...")
    mc_results = []

    for i=1:length(planet_perms)
        postcp = deepcopy(post)
        p0cp = deepcopy(p00)
        planetscp = planet_perms[i]
        for like ∈ values(postcp)
            like.model.planets = planetscp
        end
        for planet_index ∈ keys(planets0)
            if planet_index ∉ keys(planetscp)
                disable_planet_parameters!(p0cp, planets0, planet_index)
            end
        end
        map_result = run_mapfit(postcp, p0cp)
        pbest = map_result.pbest
        lnL = map_result.lnL
        aicc = compute_aicc(postcp, pbest)
        bic = compute_bic(postcp, pbest)
        redχ2 = compute_redχ2(postcp, pbest)
        n_free = num_varied(pbest)
        result = (;planets=planetscp, lnL=lnL, pbest=pbest, redχ2=redχ2, bic=bic, aicc=aicc, n_free=n_free)
        push!(mc_results, result)
    end
    aiccs = [r.aicc for r ∈ mc_results]
    ss = sortperm(aiccs)
    mc_results = [mc_results[i] for i ∈ ss]
    aiccs = [r.aicc for r ∈ mc_results]
    bics = [r.bic for r ∈ mc_results]
    Δaiccs = aiccs .- aiccs[1]
    Δbics = bics .- bics[1]
    mc_results_out = []
    for (i, r) ∈ enumerate(mc_results)
        push!(mc_results_out, (;planets=r.planets, lnL=r.lnL, pbest=r.pbest, redχ2=r.redχ2, bic=r.bic, aicc=r.aicc, n_free=r.n_free, Δaicc=Δaiccs[i], Δbic=Δbics[i]))
    end

    return mc_results_out
end

function get_planet_perms(planets)
    n = length(planets)
    pset = powerset(1:n)
    planet_perms = OrderedDict{Int, OrbitBasis}[]
    for p ∈ pset
        d = OrderedDict{Int, OrbitBasis}()
        for pp ∈ p
            d[pp] = planets[pp]
        end
        push!(planet_perms, d)
    end
    return planet_perms
end