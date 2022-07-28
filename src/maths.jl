using Statistics

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