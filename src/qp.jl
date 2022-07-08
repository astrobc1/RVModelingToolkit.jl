export QuasiPeriodic

struct QuasiPeriodic <: NoiseKernel
    par_names::Vector{String}
end

function compute_cov_matrix(kernel::QuasiPeriodic, pars, x1, x2)
    par_names = kernel.par_names
    dist_matrix = compute_stationary_dist_matrix(x1, x2)

    amp = pars[par_names[1]].value
    exp_length = pars[par_names[2]].value
    per_length = pars[par_names[3]].value
    per = pars[par_names[4]].value

    decay_term = @. -0.5 * dist_matrix^2 / exp_length^2
    
    periodic_term = @. -0.5 * sin((π / per) * dist_matrix)^2 / per_length^2
    
    K = @. amp^2 * exp(decay_term + periodic_term)
    
    return K

end