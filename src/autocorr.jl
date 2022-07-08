using DSP
using Statistics

function get_chain(chains, flat=false, thin=1, discard=0)
    v = [discard + thin - 1:self.iteration:thin]
    if flat
        s = list(v.shape[2:end])
        s[1] = prod(size(v)[1:2])
        return reshape(v, s)
    end
    return v
end

function get_autocorr_times(chain; discard=0, thin=1, c=5)
    return thin .* integrated_time((@view chain[1:thin:end, :, :]), c=c)
end

function integrated_time(chain; c=5)

    n_t, n_w, n_d = size(chain)
    τ_est = zeros(n_d)
    windows = zeros(Int, n_d)

    # Loop over parameters
    for d=1:n_d
        f = zeros(n_t)
        for k=1:n_w
            f .+= @views function_1d(chain[:, k, d])
        end
        f ./= n_w
        τs = 2 .* cumsum(f) .- 1
        windows[d] = auto_window(τs, c)
        τ_est[d] = τs[windows[d]]
    end

    return τ_est
end

function auto_window(τs, c)
    m = 1:length(τs) .< c .* τs
    if any(m)
        return argmin(m)
    end
    return length(τs)
end

function function_1d(x)
    n = next_pow_two(length(x))
    f = DSP.fft(x .- nanmean(x))
    acf = @views real.(DSP.ifft(f .* conj.(f))[1:length(x)])
    acf ./= acf[1]
    return acf
end

function next_pow_two(n)
    i = 1
    while i < n
        i = i << 1
    end
    return i
end