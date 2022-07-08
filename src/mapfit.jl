function run_mapfit(post::RVPosterior, p0::Parameters)
    println("Running MAP fit ...")
    vecs = to_vecs(p0)
    ptest = deepcopy(p0)
    pbest = deepcopy(p0)
    obj_wrapper = (x) -> begin
        set_values!(ptest, x)
        return -1 * compute_logaprob(post, ptest)
    end
    map_result = (;pbest=pbest, lnL=NaN, fcalls=0, simplex=nothing, iteration=0)
    opt_result = IterativeNelderMead.optimize(obj_wrapper, vecs.values, IterativeNelderMead.IterativeNelderMeadOptimizer(), vary=vecs.vary)
    set_values!(pbest, opt_result.pbest)
    map_result = (;pbest=pbest, lnL=-1 * opt_result.fbest, fcalls=opt_result.fcalls, simplex=opt_result.simplex, iteration=opt_result.iteration)
    println("Completed MAP fit")
    return map_result
end

