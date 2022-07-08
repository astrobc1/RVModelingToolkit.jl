using Infiltrator
using Colors
using Bokeh
using PyCall, PyPlot
using LaTeXStrings
using RVModelingToolkit

export plot_rvs_full, plot_rvs_phased, plot_rvs_phased_all, savebokehhtml, corner_plot, update_latex_strings!, ALPHABET

const ALPHABET = ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

const COLORS_HEX_GADFLY = [
    "#00BEFF", "#D4CA3A", "#FF6DAE", "#67E1B5", "#EBACFA",
    "#9E9E9E", "#F1988E", "#5DB15A", "#E28544", "#52B8AA"
]

function get_default_plot(;width, height, fontsize="16pt", xlabel, ylabel)
    p = Bokeh.figure(width=width, height=height)
    p.x_axis.axis_label = xlabel
    p.y_axis.axis_label = ylabel
    p.x_axis.axis_label_text_font = "juliamono"
    p.y_axis.axis_label_text_font = "juliamono"
    p.x_axis.major_label_text_font_size = fontsize
    p.y_axis.major_label_text_font_size = fontsize
    p.x_axis.axis_label_text_font_style = "bold"
    p.y_axis.axis_label_text_font_style = "bold"
    p.x_axis.axis_label_text_font_size = fontsize
    p.y_axis.axis_label_text_font_size = fontsize
    p.x_grid.grid_line_dash = [4, 4]
    p.y_grid.grid_line_dash = [4, 4]
    p.x_grid.grid_line_alpha = 0.8
    p.y_grid.grid_line_alpha = 0.8
    p.outline_line_color = "#000000"
    return p
end

function savebokehhtml(fname, p)
    doc = Bokeh.Document(p)
    dochtml = Bokeh.doc_standalone_html(doc)
    dochtml_split = split(dochtml, "</head>");
    dochtml_temp = String[]
    push!(dochtml_temp, dochtml_split[1])
    push!(dochtml_temp, """\n<script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.1.min.js"></script>\n""")
    push!(dochtml_temp, dochtml_split[2])
    dochtml_out = join(dochtml_temp)
    open(fname, "w") do io
        write(io, dochtml_out)
    end
end

function plot_rvs_phased(post::RVPosterior, pars::Parameters, planet_index::Int; data_colors)

    # Figures and axes
    p1 = get_default_plot(width=1200, height=600, xlabel="Phase", ylabel="RV [m/s]")
    p2 = get_default_plot(width=1200, height=350, xlabel="Phase", ylabel="Residual RV [m/s]")

    # Like0
    like0 = first(post.likes)[2]
    
    # Compute tc for this basis
    per, tp, ecc, ω, k = convert_basis(pars, planet_index, like0.model.planets[planet_index], StandardOrbitBasis)
    tc = Maths.tp_to_tc(tp, per, ecc, ω)
    
    # A high res time grid
    t_hr_one_period = collect(range(tc, tc + per, length=500))
    
    # Convert grid to phases [0, 1]
    phases_hr_one_period = Maths.get_phases(t_hr_one_period, per, tc)
    
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
            phases_data = Maths.get_phases(data.t, per, tc)
            Bokeh.plot!(p1, Bokeh.Scatter, x=phases_data, y=data_rv1, size=6, color=data_colors[data.instname], legend_label=data.instname)
            Bokeh.plot!(p2, Bokeh.Scatter, x=phases_data, y=data_rv2, size=6, color=data_colors[data.instname], legend_label=data.instname)
            for i=1:length(data_rv1)
                w1 = Bokeh.plot!(p1, Bokeh.Whisker, base=phases_data[i], upper=data_rv1[i] + data_rverr[i], lower=data_rv1[i] - data_rverr[i], line_color=data_colors[data.instname], line_width=2)
                w1.upper_head.line_color = data_colors[data.instname]
                w1.lower_head.line_color = data_colors[data.instname]
                w1.upper_head.line_width = 2
                w1.lower_head.line_width = 2
                w2 = Bokeh.plot!(p2, Bokeh.Whisker, base=phases_data[i], upper=data_rv2[i] + data_rverr[i], lower=data_rv2[i] - data_rverr[i], line_color=data_colors[data.instname], line_width=2)
                w2.upper_head.line_color = data_colors[data.instname]
                w2.lower_head.line_color = data_colors[data.instname]
                w2.upper_head.line_width = 2
                w2.lower_head.line_width = 2
            end
            
            # Store for binning at the end
            phases_data_all = vcat(phases_data_all, phases_data)
            data_rv_all1 = vcat(data_rv_all1, data_rv1)
            data_rv_all2 = vcat(data_rv_all2, data_rv2)
            data_rverr_all = vcat(data_rverr_all, data_rverr)
        end
    end

    # Plot the model
    Bokeh.plot!(p1, Bokeh.Line, x=phases_hr_one_period, y=planet_model_phased)
    
    # Plot the the binned data
    ss = sortperm(phases_data_all)
    phases_data_all, data_rv_all1, data_rv_all2, data_rverr_all = phases_data_all[ss], data_rv_all1[ss], data_rv_all2[ss],  data_rverr_all[ss]
    phases_binned, rv_binned1, rverr_binned = Maths.bin_phased_rvs(phases_data_all, data_rv_all1, data_rverr_all, nbins=10)
    phases_binned, rv_binned2, rverr_binned = Maths.bin_phased_rvs(phases_data_all, data_rv_all2, data_rverr_all, nbins=10)
    Bokeh.plot!(p1, Bokeh.Scatter, x=phases_binned, y=rv_binned1, size=14, fill_color="maroon")
    #Bokeh.plot!(p2, Bokeh.Scatter, x=phases_binned, y=rv_binned2, size=14, fill_color="maroon")
    for i=1:length(phases_binned)
        w1 = Bokeh.plot!(p1, Whisker, base=phases_binned[i], upper=rv_binned1[i] + rverr_binned[i], lower=rv_binned1[i] - rverr_binned[i], line_color="black", line_width=4)
        w1.upper_head.line_color = "black"
        w1.lower_head.line_color = "black"
        w1.upper_head.line_width = 4
        w1.lower_head.line_width = 4
        #w2 = Bokeh.plot!(p2, Whisker, base=phases_binned[i], upper=rv_binned2[i] + rverr_binned[i], lower=rv_binned2[i] - rverr_binned[i], line_color="black")
        #w2.upper_head.line_color = "black"
        #w2.lower_head.line_color = "black"
    end

    # Final config
    p1.legend.label_text_font_size = "16pt"
    p1.legend.label_text_font_style = "bold"
    p2.legend.label_text_font_size = "16pt"
    p2.legend.label_text_font_style = "bold"
    
    # Return figs
    return p1, p2

end


function plot_rvs_phased_all(post::RVPosterior, pars::Parameters; data_colors)
    figs1, figs2 = [], []
    like0 = first(post)[2]
    for planet_index ∈ keys(like0.model.planets)
        result = plot_rvs_phased(post, pars, planet_index, data_colors=data_colors)
        push!(figs1, result[1])
        push!(figs2, result[2])
    end
    return figs1, figs2
end

function get_smart_gp_sampling_times(data_t, Δt, δt)
    ti, tf = minimum(data_t), maximum(data_t)
    t_out = Float64[]
    for i=1:length(data_t)
        t_out = vcat(t_out, [data_t[i] - Δt / 2:(δt/4):data_t[i] + Δt / 2;])
    end

    # Delete points we don't need
    n_init = length(t_out)
    for _=1:n_init
        d = abs.(diff(t_out))
        k = argmin(d)
        if d[k] < δt
            deleteat!(t_out, k)
        else
            break
        end
    end
    ss = sortperm(t_out)
    t_out .= t_out[ss]

    # Chunk
    #@infiltrate
    # d = diff(t_out)
    # s = findall(d .> Δt / 2)
    # tt_out = Vector{Float64}[]
    # for i=1:length(s)-1
    #     push!(tt_out, t_out[s[i]:s[i+1]])
    # end
    return t_out
end


function plot_rvs_full(post::RVPosterior, pars::Parameters, n_model_pts=5000; kernel_sampling=nothing, kernel_window=nothing, data_colors=nothing, gp_colors=nothing, time_offset=nothing, gp_Δt=nothing, gp_δt=nothing)

    # Like0
    like0 = first(post.likes)[2]

    # Time offset
    if isnothing(time_offset)
        time_offset = like0.model.t0
    end
    if round(time_offset) == time_offset
        time_offset = Int(time_offset)
    end

    # Figure and axes
    p1 = get_default_plot(width=1400, height=600, xlabel="BJD - $(time_offset)", ylabel="RV [m/s]")
    p2 = get_default_plot(width=1400, height=400, xlabel="BJD - $(time_offset)", ylabel="Residual RV [m/s]")
    
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
        Bokeh.plot!(p1, Line; x=thr .- like0.model.t0, y=model_rv, legend_label="Keplerian Model", line_color="black")
    end
    
    # Loop over likes and:
    # 1. Plot high res GP if present
    # 2. Plot data and residuals
    for like ∈ values(post)

        data_t = get_times(like.data)
        residuals = compute_residuals(like, pars)
        errors = compute_data_errors(like, pars)

        # Correlated Noise
        if !isnothing(like.gp)
            t_gp = get_smart_gp_sampling_times(data_t, gp_Δt, gp_δt)
            noise_components_temp = compute_noise_components(like, pars, t_gp[1:2])
            noise_labels = collect(keys(noise_components_temp))
            gps_hr = Dict{}(label => Float64[] for label ∈ noise_labels)
            gps_error_hr = Dict{}(label => Float64[] for label ∈ noise_labels)
            noise_components = compute_noise_components(like, pars, t_gp)
            for comp ∈ noise_labels
                gps_hr[comp] = vcat(gps_hr[comp], noise_components[comp][1])
                gps_error_hr[comp] = vcat(gps_error_hr[comp], noise_components[comp][2])
            end
                
            # Plot the GPs
            for comp ∈ noise_labels

                # Plot GP
                Bokeh.plot!(p1, Bokeh.Line, x=t_gp .- time_offset, y=gps_hr[comp], line_color=gp_colors[comp])
            
                # Plot GP error
                gp_hr_lower, gps_hr_upper = gps_hr[comp] .- gps_error_hr[comp], gps_hr[comp] .+ gps_error_hr[comp]
                Bokeh.plot!(p1, Bokeh.VArea, x=t_gp .- time_offset, y1=gp_hr_lower, y2=gps_hr_upper, fill_color=gp_colors[comp], fill_alpha=0.5, legend_label=comp)
            end
        
            # Compute the noise model for the data and remove
            noise_components = compute_noise_components(like, pars, data_t)
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
            Bokeh.plot!(p1, Bokeh.Scatter, x=data.t .- time_offset, y=data_rv1, size=12, color=data_colors[data.instname], line_color="#42464a", legend_label=data.instname)
            Bokeh.plot!(p2, Bokeh.Scatter, x=data.t .- time_offset, y=data_rv2, size=10, color=data_colors[data.instname], line_color="#42464a")
            for i=1:length(data_rv1)
                w1 = Bokeh.plot!(p1, Bokeh.Whisker, base=data.t[i] - time_offset, upper=data_rv1[i] + data_rverr[i], lower=data_rv1[i] - data_rverr[i], line_color=data_colors[data.instname], line_width=2)
                w1.upper_head.line_color = data_colors[data.instname]
                w1.lower_head.line_color = data_colors[data.instname]
                w1.upper_head.line_width = 2
                w1.lower_head.line_width = 2
                w2 = Bokeh.plot!(p2, Bokeh.Whisker, base=data.t[i] - time_offset, upper=data_rv2[i] + data_rverr[i], lower=data_rv2[i] - data_rverr[i], line_color=data_colors[data.instname], line_width=2)
                w2.upper_head.line_color = data_colors[data.instname]
                w2.lower_head.line_color = data_colors[data.instname]
                w2.upper_head.line_width = 2
                w2.lower_head.line_width = 2
            end
        end
    end
    
    # Return the figures
    return p1, p2

end


        
function corner_plot(post::RVPosterior, mcmc_result)
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
            par.latex_str = L"\sigma_{%$instname}$"
        end
    end
end