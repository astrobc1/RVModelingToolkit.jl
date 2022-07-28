export QuasiPeriodic

"""
Type for a QuasiPeriodic GP kernel. `par_names` is a vector containing the names of the hyper-parameters, which must be in the following order, however can use arbitrary names.
    1. amplitude
    2. exponential length scale
    3. period length scale
    4. period
"""
struct QuasiPeriodic <: NoiseKernel
    par_names::Vector{String}
end

"""
    compute_cov_matrix(kernel::QuasiPeriodic, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real})
"""
function compute_cov_matrix(kernel::QuasiPeriodic, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real})
    par_names = kernel.par_names
    dist_matrix = compute_stationary_dist_matrix(t1, t2)

    amp = pars[par_names[1]].value
    exp_length = pars[par_names[2]].value
    per_length = pars[par_names[3]].value
    per = pars[par_names[4]].value

    decay_term = @. -0.5 * dist_matrix^2 / exp_length^2
    
    periodic_term = @. -0.5 * sin((π / per) * dist_matrix)^2 / per_length^2
    
    K = @. amp^2 * exp(decay_term + periodic_term)
    
    return K

end