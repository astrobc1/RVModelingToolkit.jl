using Colors
using PlotlyJS
using PyCall, PyPlot
using LaTeXStrings
using Formatting

export plot_rvs_full, plot_rvs_phased, plot_rvs_phased_all, savebokehhtml, corner_plot, update_latex_strings!, ALPHABET, COLORS_HEX_GADFLY

# alphabet ignoring a because astronomers
const ALPHABET = ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Colors
const COLORS_HEX_GADFLY = [
    "#00BEFF", "#D4CA3A", "#FF6DAE", "#67E1B5", "#EBACFA",
    "#9E9E9E", "#F1988E", "#5DB15A", "#E28544", "#52B8AA"
]

function hex2rgba(h, a=1.0)
    h = strip(h, '#')
    r, g, b = Int.(hex2bytes(h))
    s = "rgba($r,$g,$b,$a)"
    return s
end

"""
    plot_rvs_phased(post::RVPosterior, pars::Parameters, planet_index::Int; data_colors::AbstractDict, titles::Bool=true, star_name::Union  {String, Nothing}=nothing)
Plot the phased RVs for a single planet. Returns a PlotlyJS figure which can be saved to the interactive HTML file `fname` with `PlotlyJS.savefig(p, fname)`.
"""
function plot_rvs_phased(post::RVPosterior, pars::Parameters, planet_index::Int; data_colors::AbstractDict, titles::Bool=true, star_name::Union{String, Nothing}=nothing)

    # Like0
    like0 = first(post.likes)[2]

    star_name_str = replace(star_name, "_"=>" ")
    
    # Compute tc for this basis
    per, tp, ecc, ω, k = convert_basis(pars, planet_index, like0.model.planets[planet_index], StandardOrbitBasis)
    tc = tp_to_tc(tp, per, ecc, ω)

    # Figures
    if titles
        title="<b>$(star_name_str) $(ALPHABET[planet_index]), P = $(round(per, digits=6)) d, K = $(round(k, digits=2)) m/s, e = $(round(ecc, digits=3))</b>"
    else
        title=""
    end
    p1 = PlotlyJS.plot(PlotlyJS.Layout(title=title, xaxis_title="<b>Phase</b>", yaxis_title="<b>RV [m/s]</b>", font=PlotlyJS.attr(family="juliamono, Courier New", size=20), template="plotly_white", width=1400, height=800))
    p2 = PlotlyJS.plot(PlotlyJS.Layout(title="", xaxis_title="<b>Phase</b>", yaxis_title="<b>Residual RV [m/s]</b>", font=PlotlyJS.attr(family="juliamono, Courier New", size=20), template="plotly_white", width=1400, height=800))
    
    # A high res time grid
    t_hr_one_period = collect(range(tc, tc + per, length=500))
    
    # Convert grid to phases [0, 1]
    phases_hr_one_period = get_phases(t_hr_one_period, per, tc)
    
    # Build high res model for this planet
    planet_model_phased = build_planet(like0.model, pars, t_hr_one_period, planet_index)
    
    # Sort the phased model
    ss = sortperm(phases_hr_one_period)
    phases_hr_one_period .= phases_hr_one_period[ss]
    planet_model_phased .= planet_model_phased[ss]
    
    # Store the data in order to bin the phased RVs.
    phases_data_all = Float64[]
    data_rv_all1 = Float64[]
    data_rv_all2 = Float64[]
    data_rverr_all = Float64[]
    
    # Loop over likes
    for like ∈ values(post)
        
        # Compute the final residuals
        residuals = compute_residuals(like, pars)
        errors = compute_data_errors(like, pars)
        
        # Compute the noise model
        if !isnothing(like.gp)
            data_t = get_times(like.data)
            noise_components = compute_noise_components(like, pars, data_t)
            noise_labels = collect(keys(noise_components))
            for comp ∈ noise_labels
                residuals[noise_components[comp][3]] .-= noise_components[comp][1][noise_components[comp][3]]
            end
        end
        
        # Loop over instruments and plot each
        for data ∈ values(like.data)
            data_rverr = errors[like.data.indices[data.instname]]
            data_rv1 = residuals[like.data.indices[data.instname]] .+ build_planet(like.model, pars, data.t, planet_index)
            data_rv2 = residuals[like.data.indices[data.instname]]
            phases_data = get_phases(data.t, per, tc)
            PlotlyJS.add_trace!(p1, PlotlyJS.scatter(x=phases_data, y=data_rv1, error_y=PlotlyJS.attr(array=data_rverr), name="<b>$(data.instname)</b>", mode="markers", marker=PlotlyJS.attr(color=data_colors[data.instname], size=8)))
            PlotlyJS.add_trace!(p2, PlotlyJS.scatter(x=phases_data, y=data_rv2, error_y=PlotlyJS.attr(array=data_rverr), name="<b>$(data.instname)</b>", mode="markers", marker=PlotlyJS.attr(color=data_colors[data.instname], size=8)))
            phases_data_all = vcat(phases_data_all, phases_data)
            data_rv_all1 = vcat(data_rv_all1, data_rv1)
            data_rv_all2 = vcat(data_rv_all2, data_rv2)
            data_rverr_all = vcat(data_rverr_all, data_rverr)
        end
    end

    # Plot the model
    PlotlyJS.add_trace!(p1, PlotlyJS.scatter(x=phases_hr_one_period, y=planet_model_phased, name="<b>Keplerian Model</b>", line=PlotlyJS.attr(color="black", width=3)))
    
    # Plot the the binned data
    ss = sortperm(phases_data_all)
    phases_data_all, data_rv_all1, data_rv_all2, data_rverr_all = phases_data_all[ss], data_rv_all1[ss], data_rv_all2[ss],  data_rverr_all[ss]
    phases_binned, rv_binned1, rverr_binned = bin_phased_rvs(phases_data_all, data_rv_all1, data_rverr_all, nbins=10)
    PlotlyJS.add_trace!(p1, PlotlyJS.scatter(x=phases_binned, y=rv_binned1, error_y=PlotlyJS.attr(array=rverr_binned), name=nothing, showlegend=false, mode="markers", marker=PlotlyJS.attr(color="Maroon", size=12, line=PlotlyJS.attr(width=2, color="DarkSlateGrey"))))

    # Final config
    PlotlyJS.update_xaxes!(p1, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)
    PlotlyJS.update_yaxes!(p1, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)
    PlotlyJS.update_xaxes!(p2, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)
    PlotlyJS.update_yaxes!(p2, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)

    # Return figs
    return p1, p2

end

"""
    plot_rvs_phased_all(post::RVPosterior, pars::Parameters; data_colors)
Wrapper to plot all phased RVs. Returns 2 vectors of Plotly figures, which can be saved to the interactive HTML files with `PlotlyJS.savefig(p, fname)`.
"""
function plot_rvs_phased_all(post::RVPosterior, pars::Parameters; data_colors::AbstractDict, titles=true, star_name=nothing)
    figs1, figs2 = [], []
    like0 = first(post)[2]
    for planet_index ∈ keys(like0.model.planets)
        result = plot_rvs_phased(post, pars, planet_index, data_colors=data_colors, titles=titles, star_name=star_name)
        push!(figs1, result[1])
        push!(figs2, result[2])
    end
    return figs1, figs2
end


function get_smart_gp_sampling_times(data_t, Δt, δt)
    t_out = [minimum(data_t) - Δt / 2:δt:maximum(data_t) + Δt / 2;]
    n_init = length(t_out)
    bad = ones(n_init)
    for i=1:n_init
        if minimum(abs.(t_out[i] .- data_t)) > Δt
            bad[i] = 0
        end
    end
    bad = findall(bad .== 0)
    t_out = deleteat!(t_out, bad)
    return t_out
end

"""
    plot_rvs_full(post::RVPosterior, pars::Parameters; data_colors::Union{AbstractDict, Nothing}=nothing, gp_colors::Union{AbstractDict, Nothing}=nothing, time_offset::Union{<:Real, Nothing}=nothing, gp_Δt::Union{<:Real, Nothing}=nothing, gp_δt::Union{<:Real, Nothing}=nothing, n_model_pts::Int=5000)
Plots the data RVs, the Keplerian model, and the GP as a function of modified BJD, as well as the residuals (separate plots). Returns two PlotlyJS plots, which can be saved to the interactive HTML files with `PlotlyJS.savefig(p, fname)`.
"""
function plot_rvs_full(post::RVPosterior, pars::Parameters; data_colors::Union{AbstractDict, Nothing}=nothing, gp_colors::Union{AbstractDict, Nothing}=nothing, time_offset::Union{<:Real, Nothing}=nothing, gp_Δt::Union{<:Real, Nothing}=nothing, gp_δt::Union{<:Real, Nothing}=nothing, n_model_pts::Int=5000)

    # Like0
    like0 = first(post.likes)[2]

    # Time offset
    if isnothing(time_offset)
        time_offset = like0.model.t0
    end
    time_offset_str = Formatting.format(time_offset)

    # Figures
    p1 = PlotlyJS.plot(PlotlyJS.Layout(title="", xaxis_title="<b>BJD - $(time_offset_str)</b>", yaxis_title="<b>RV [m/s]</b>", font=PlotlyJS.attr(family="juliamono, Courier New", size=20), template="plotly_white", width=1700, height=700))
    p2 = PlotlyJS.plot(PlotlyJS.Layout(title="", xaxis_title="<b>BJD - $(time_offset_str)</b>", yaxis_title="<b>Residual RV [m/s]</b>", font=PlotlyJS.attr(family="juliamono, Courier New", size=20), template="plotly_white", width=1700, height=500))
    
    # Plot the high resolution Keplerian model + trend (no noise yet)
    if length(like0.model.planets) > 0 || like0.model.trend_poly_deg > 0
        
        # Generate a high resolution data grid
        t_data_all = Float64[]
        for like ∈ values(post)
            t_data_all = vcat(t_data_all, get_times(like.data))
        end
        ti, tf = maximum(t_data_all), minimum(t_data_all)
        Δt = tf - ti
        thr = collect(range(ti - Δt / 100, tf + Δt / 100, length=n_model_pts))
    
        # Generate the high res Keplerian + Trend model
        model_rv = build(like0.model, pars, thr)

        # Plot the planet model
        PlotlyJS.add_trace!(p1, PlotlyJS.scatter(x=thr .- time_offset, y=model_rv, name="<b>Keplerian Model</b>", line=PlotlyJS.attr(color="black", width=3)))
    end
    
    # Loop over likes and:
    # 1. Plot high res GP if present
    # 2. Plot data and residuals
    for like ∈ values(post)

        data_t = get_times(like.data)

        # Correlated Noise
        if !isnothing(like.gp)
            t_gp = get_smart_gp_sampling_times(data_t, gp_Δt, gp_δt)
            noise_components_temp = compute_noise_components(like, pars, t_gp[1:2])
            noise_labels = collect(keys(noise_components_temp))
            gps_hr = Dict{}(label => Float64[] for label ∈ noise_labels)
            gps_error_hr = Dict{}(label => Float64[] for label ∈ noise_labels)
            n_chunks = Int(ceil(length(t_gp) / 500))
            for i=1:n_chunks
                ss = ((i-1)*500+1):min(((i)*500), length(t_gp))
                noise_components = compute_noise_components(like, pars, t_gp[ss])
                for comp ∈ noise_labels
                    _gp_hr = noise_components[comp][1]
                    _gp_error_hr = noise_components[comp][2]
                    gps_hr[comp] = vcat(gps_hr[comp], _gp_hr)
                    gps_error_hr[comp] = vcat(gps_error_hr[comp], _gp_error_hr)
                end
            end

            for comp ∈ noise_labels

                chunks = group_times_gp(t_gp, gp_Δt)
                n_chunks = length(chunks)

                for i=1:n_chunks
                    
                    ss = chunks[i][1]:chunks[i][2]
                    tt = t_gp[ss] .- time_offset

                    # Colors
                    color_line = hex2rgba(gp_colors[comp], 0.6)
                    color_fill = hex2rgba(gp_colors[comp], 0.3)

                    # Plot GP
                    PlotlyJS.add_trace!(p1, PlotlyJS.scatter(x=t_gp[ss] .- time_offset, y=gps_hr[comp][ss], line=PlotlyJS.attr(width=0.8, color=gp_colors[comp]), name=nothing, showlegend=false))
                
                    # Plot GP error
                    gp_hr_lower, gp_hr_upper = gps_hr[comp][ss] .- gps_error_hr[comp][ss], gps_hr[comp][ss] .+ gps_error_hr[comp][ss]
                    if i == 1
                        add_trace!(p1, PlotlyJS.scatter(x=vcat(tt, reverse(tt)), y=vcat(gp_hr_upper, reverse(gp_hr_lower)), fill="toself",
                                                               line=PlotlyJS.attr(width=1, color=color_line), fillcolor=color_fill, name="<b>$(comp)</b>", showlegend=true))
                    else
                        add_trace!(p1, PlotlyJS.scatter(x=vcat(tt, reverse(tt)), y=vcat(gp_hr_upper, reverse(gp_hr_lower)), fill="toself",
                                                               line=PlotlyJS.attr(width=1, color=color_line), fillcolor=color_fill, name=nothing, showlegend=false))
                    end
                end
            end
        end
    end

    for like ∈ values(post)

        data_t = get_times(like.data)
        residuals = compute_residuals(like, pars)
        errors = compute_data_errors(like, pars)

        # Compute the noise model for the data and remove
        if !isnothing(like.gp)
            noise_components = compute_noise_components(like, pars, data_t)
            noise_labels = collect(keys(noise_components))
            for comp ∈ noise_labels
                residuals[noise_components[comp][3]] .-= noise_components[comp][1][noise_components[comp][3]]
            end
        end

        # Plot each instrument
        for data ∈ values(like.data)
                
            # Raw data - zero point
            data_rv1 = data.rv .- pars["gamma_$(data.instname)"].value
            
            # Residuals
            data_rv2 = residuals[like.data.indices[data.instname]]
            
            # Errors
            data_rverr = errors[like.data.indices[data.instname]]
            
            # Plot rvs
            marker_color = hex2rgba(data_colors[data.instname], 0.9)
            PlotlyJS.add_trace!(p1, PlotlyJS.scatter(x=data.t .- time_offset, y=data_rv1, error_y=PlotlyJS.attr(array=data_rverr), name="<b>$(data.instname)</b>", mode="markers", marker=PlotlyJS.attr(color=marker_color, size=14, line=PlotlyJS.attr(width=2, color="DarkSlateGrey"))))
            PlotlyJS.add_trace!(p2, PlotlyJS.scatter(x=data.t .- time_offset, y=data_rv2, error_y=PlotlyJS.attr(array=data_rverr), name="<b>$(data.instname)</b>", mode="markers", marker=PlotlyJS.attr(color=marker_color, size=14, line=PlotlyJS.attr(width=2, color="DarkSlateGrey"))))
        end

    end

    # Final config
    PlotlyJS.update_xaxes!(p1, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)
    PlotlyJS.update_yaxes!(p1, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)
    PlotlyJS.update_xaxes!(p2, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)
    PlotlyJS.update_yaxes!(p2, zeroline=false, tickprefix="<b>", ticksuffix ="</b><br>", automargin=true)
    
    # Return the figures
    return p1, p2

end


"""
    corner_plot(post::RVPosterior, mcmc_result::NamedTuple)
Create a corner plot with the `NamedTuple` returned from `run_mcmc`.
"""
function corner_plot(post::RVPosterior, mcmc_result::NamedTuple)
    pmedvecs = to_vecs(mcmc_result.pmed)
    vi = findall(pmedvecs.vary)
    truths = pmedvecs.values[vi]
    labels = [par.latex_str for par ∈ values(mcmc_result.pmed) if par.vary]
    corner = pyimport("corner")
    pygui(false)
    p = corner.corner(mcmc_result.chains, labels=labels, truths=truths, show_titles=true)
    return p
end

function Base.isdigit(s::String)
    try
        parse(Float64, s)
        true
    catch
        false
    end
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
            binned_unc[i] = @views rvs[inds[1]]
        else
            binned_unc[i] = @views weighted_stddev(rvs[inds], w) / sqrt(n)
        end
    end

    return binned_phases, binned_rvs, binned_unc
end


function update_latex_strings!(post, pars)
    planets = first(post)[2].model.planets
    for par ∈ values(pars)
        
        pname = par.name
    
        # Planets (per, tc, k, ecc, w, other bases added later if necessary)
        if startswith(pname, "per") && isdigit(pname[4:end])
            ii = parse(Int, pname[end])
            letter = ALPHABET[ii]
            par.latex_str = L"P_{%$letter}"
        elseif startswith(pname, "tc") && isdigit(pname[3:end])
            ii = parse(Int, pname[end])
            letter = ALPHABET[ii]
            par.latex_str = L"Tc_{%$letter}"
        elseif startswith(pname, "ecc") && isdigit(pname[4:end])
            ii = parse(Int, pname[end])
            letter = ALPHABET[ii]
            par.latex_str = L"e_{%$letter}"
        elseif startswith(pname, "w") && isdigit(pname[2:end])
            ii = parse(Int, pname[end])
            letter = ALPHABET[ii]
            par.latex_str = L"\omega_{%$letter}"
        elseif startswith(pname, "k") && isdigit(pname[2:end])
            ii = parse(Int, pname[end])
            letter = ALPHABET[ii]
            par.latex_str = L"K_{%$letter}"
            
        # Gammas
        elseif startswith(pname, "gamma") && !endswith(pname, "dot")
            instname = split(pname, "_")[end]
            par.latex_str = L"\gamma_{%$instname}"
        elseif startswith(pname, "gamma") && endswith(pname, "_dot")
            par.latex_str = L"\dot{\gamma}"
        elseif startswith(pname, "gamma") && endswith(pname, "_ddot")
            par.latex_str = L"\ddot{\gamma}"
    
        # Jitter
        elseif startswith(pname, "jitter_")
            instname = split(pname, "_")[end]
            par.latex_str = L"\sigma_{%$instname}"
        end
    end
end

function group_times_gp(t, Δt)
    n = length(t)
    prev_i = 1
    indices = Vector{Int64}[]
    if n == 1
        push!(indices, [1, 1])
    else
        for i=1:n-1
            if t[i+1] - t[i] > Δt
                push!(indices, [prev_i, i])
                prev_i = i + 1
            end
        end
        push!(indices, [prev_i, n - 1])
    end

    return indices
end