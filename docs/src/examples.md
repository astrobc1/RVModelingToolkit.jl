Examples
========

# Example 1: KELT-24, a massive hot Jupiter, no GP

Download the RVs for KELT-24 from [here](https://drive.google.com/file/d/1A94iHcSua-p-cQcL8_JRMdPcpAq8-hvr/view?usp=sharing).

```julia

# Imports
using RVModelingToolkit
using Statistics
using PlotlyJS

# Load in data (csv file with first row time, mnvel, errvel,tel)
data = read_radvel_file("kelt24_rvs_20220708_example.txt")

# Init parameters and planets dictionary
pars = Parameters()
planets = Dict{Int, OrbitBasis}()

# Planet 1 (b)
# per, tc, ecc, w, k;
# period [days]
# time of conjunction (BJD)
# eccentricity ∈ [0, 1]
# Arg of periastron of star's orbit [radians]
# RV semi-amplitude [m/s]
planets[1] = TCOrbitBasis()
per1 = 5.5514926
tc1 =  2457147.0529
ecc1 = 0.077
ecc1_unc = 0.04
w1 = 55 * π / 180
w1_unc = 15 * π / 180
pars["per1"] = Parameter(value=per1, vary=false)
pars["tc1"] = Parameter(value=tc1, vary=false)
pars["ecc1"] = Parameter(value=ecc1, vary=true)
add_prior!(pars["ecc1"], Priors.Uniform(0, 1))
add_prior!(pars["ecc1"], Priors.Gaussian(ecc1, ecc1_unc))
pars["w1"] = Parameter(value=w1, vary=true)
add_prior!(pars["w1"], Priors.Gaussian(w1, w1_unc))
pars["k1"] = Parameter(value=450, vary=true)
add_prior!(pars["k1"], Priors.Positive())

# Zero points of each spectrograph
for instname ∈ keys(data)
    pars["gamma_$instname"] = Parameter(value=median(data[instname].rv) + π / 100, vary=true)
end

# Linear trend parameter
pars["gamma_dot"] = Parameter(value=0.1, vary=true)
add_prior!(pars["gamma_dot"], Priors.Uniform(-1, 1))

# "Jitter" (uncorrelated noise; adds a correction term (must be positive) in-quadrature to the posterior distribution, unique to each spectrograph)
for instname ∈ keys(data)
    pname = "jitter_$instname"
    pars[pname] = Parameter(value=10, vary=true)
    add_prior!(pars[pname], Priors.JeffreysGD(1E-10, 200))
end

# TRES jitter tends to be zero, so we overwrite it here
pars["jitter_TRES"] = Parameter(value=0, vary=false)

# Store each like in a dictionary
likes = Dict{String, RVLikelihood}()

# Planets, trend polynomial degree (gammas), trend zero point for terms > 0, also used in plots
model = RVModel(planets, 1, t0=round(median(get_times(data))))

# Construct single likelihood and posterior objects. Here the key "rvs_all" is arbitrary.
likes["rvs_all"] = RVLikelihood(data, model)
post = RVPosterior(likes)
update_latex_strings!(post, pars)

# maximize lnL with NelderMead (returns NamedTuple)
map_result = run_mapfit(post, pars)
pbest = map_result.pbest
println(pbest)

# Interactive RV plots with PlotlyJS
star_name = "KELT-24"
data_colors = Dict("TRES" => COLORS_HEX_GADFLY[1], "SONG" => COLORS_HEX_GADFLY[2])
fig_full, fig_res = plot_rvs_full(post, pbest, data_colors=data_colors)
PlotlyJS.savefig(fig_full, "$(star_name)_rvs.html")
PlotlyJS.savefig(fig_res, "$(star_name)_rvs_residuals.html")
figs_phased, figs_phased_res = plot_rvs_phased_all(post, pbest, data_colors=data_colors, star_name=star_name)
for i=1:length(figs_phased)
    PlotlyJS.savefig(figs_phased[i], "$(star_name)_$(ALPHABET[i])_rvs_phased.html")
    PlotlyJS.savefig(figs_phased_res[i], "$(star_name)_$(ALPHABET[i])_rvs_phased_residuals.html")
end

# MCMC (returns NamedTuple)
mcmc_result = run_mcmc(post, pbest, n_burn_steps=500, n_τs_thresh=40, n_min_steps=1000, n_max_steps=50_000)

# Corner plot
fig = corner_plot(post, mcmc_result)
fig.savefig("$(star_name)_corner.png")

# Model comparison (Vector of NamedTuples sorted by the delta AICc)
mc_result = model_comparison(post, pbest)
```


# Example 2: AU Mic, with disjoint GPs.

Download the RVs for AU Mic from [here](https://drive.google.com/file/d/193QMqe3hFamTdVJQlkYSiW7xYAKn0YWs/view?usp=sharing).

```julia
using RVModelingToolkit
using Statistics
using DataStructures
using PlotlyJS

# Data
fname = "aumic_rvs_20220710_example.txt"
data = read_radvel_file(fname)

# Init parameters and planets dictionary
pars = Parameters()
planets = OrderedDict{Int, OrbitBasis}()

# Planet 1 (detected at a few sigma here)
planets[1] = TCOrbitBasis()
per1 = 8.4629991
tc1 = 2458330.39046
ecc1 = 0.189
w1 = 1.5449655 + π

pars["per1"] = Parameter(value=per1, vary=false)
pars["tc1"] = Parameter(value=tc1, vary=false)
pars["ecc1"] = Parameter(value=ecc1, vary=false)
pars["w1"] = Parameter(value=w1, vary=false)
pars["k1"] = Parameter(value=8, vary=true)
add_prior!(pars["k1"], Priors.Positive())

# Planet c (not detected through this analysis)
planets[2] = TCOrbitBasis()
per2 = 18.858991
tc2 = 2458342.2243

pars["per2"] = Parameter(value=per2, vary=false)
pars["tc2"] = Parameter(value=tc2, vary=false)
pars["ecc2"] = Parameter(value=0, vary=false)
pars["w2"] = Parameter(value=π, vary=false)
pars["k2"] = Parameter(value=5, vary=true)
add_prior!(pars["k2"], Priors.Positive())

# Gamma offsets, don't include gamma dot or ddot yet
for instname ∈ keys(data)
    pars["gamma_$instname"] = Parameter(value=median(data[instname].rv) + π / 100, vary=true)
end

# Fix jitter to zero by default
for instname ∈ keys(data)
    pname = "jitter_$instname"
    pars[pname] = Parameter(value=0, vary=false)
end

# Increase HIRES error bars
pars["jitter_HIRES"] = Parameter(value=5, vary=false)

# GP amplitudes for each spectrogrpah
for (i, instname) ∈ enumerate(keys(data))
    pname = "gp_amp_$instname"
    pars[pname] = Parameter(value=std(data[instname].rv), vary=true)
    add_prior!(pars[pname], Priors.JeffreysGD(1, 600))
    pars[pname].latex_str = L"\eta_{\sigma,%$instname}"
end

# GP decay time (days)
pars["gp_decay"] = Parameter(value=100, vary=false)
add_prior!(pars["gp_decay"], Priors.Uniform(20, 2_000))
pars["gp_decay"].latex_str = L"\eta_{\tau}"

# GP smoothness (unitless)
pars["gp_per_length"] = Parameter(value=0.28, vary=false)
pars["gp_per_length"].latex_str = "\eta_{\ell}"

# GP period (days)
pars["gp_per"] = Parameter(value=4.836, vary=true)
add_prior!(pars["gp_per"], Priors.Gaussian(pars["gp_per"].value, 0.001))
pars["gp_per"].latex_str = L"\eta_{P}"

# noise, model, like, post
# Each spectrograph gets its own like here
likes = OrderedDict{String, RVLikelihood}()
for _data ∈ values(data)
    data_view = get_view(data, [_data.instname])
    model = RVModel(planets, 0, t0=2450000.0)
    gp = GaussianProcess(QuasiPeriodic(["gp_amp_$(_data.instname)", "gp_decay", "gp_per_length", "gp_per"]))
    likes[_data.instname] = RVLikelihood(data_view, model, gp)
end

post = RVPosterior(likes)
update_latex_strings!(post, pars)

# Map fit
map_result = run_mapfit(post, pars)

# Alias best fit params
pbest = map_result.pbest
println(pbest)

# RV plots, also color GPs to match spectrographs
star_name = "AU_Mic"
data_colors = Dict("HIRES" => COLORS_HEX_GADFLY[1], "TRES" => COLORS_HEX_GADFLY[2], "SPIRou" => COLORS_HEX_GADFLY[3], "iSHELL" => COLORS_HEX_GADFLY[4])
gp_colors = Dict("GP [HIRES]" => COLORS_HEX_GADFLY[1], "GP [TRES]" => COLORS_HEX_GADFLY[2], "GP [SPIRou]" => COLORS_HEX_GADFLY[3], "GP [iSHELL]" => COLORS_HEX_GADFLY[4])
fig_full, fig_res = plot_rvs_full(post, pbest, data_colors=data_colors, gp_colors=gp_colors, gp_Δt=50, gp_δt=0.02)
PlotlyJS.savefig(fig_full, "$(star_name)_rvs.html")
PlotlyJS.savefig(fig_res, "$(star_name)_rvs_residuals.html")
figs_phased, figs_phased_res = plot_rvs_phased_all(post, pbest, data_colors=data_colors, titles=true, star_name="AU Mic")
for i=1:length(figs_phased)
    PlotlyJS.savefig(figs_phased[i], "$(star_name)_$(ALPHABET[i])_rvs_phased.html")
    PlotlyJS.savefig(figs_phased_res[i], "$(star_name)_$(ALPHABET[i])_rvs_phased_residuals.html")
end

# MCMC
mcmc_result = run_mcmc(post, pbest, n_burn_steps=500, n_τs_thresh=40, n_min_steps=1000, n_max_steps=80_000)

# Corner plot
fig = corner_plot(post, mcmc_result)
fig.savefig("$(star_name)_corner.png", dpi=200)
```

# Example 3: AU Mic, with J1 kernel.

```julia
using RVModelingToolkit
using Statistics
using DataStructures
using PlotlyJS

# Data
fname = "aumic_rvs_20220710_example.txt"
data = read_radvel_file(fname)

# Init parameters and planets dictionary
pars = Parameters()
planets = OrderedDict{Int, OrbitBasis}()

# Planet 1 (detected at a few sigma here)
planets[1] = TCOrbitBasis()
per1 = 8.4629991
tc1 = 2458330.39046
ecc1 = 0.189
w1 = 1.5449655 + π

pars["per1"] = Parameter(value=per1, vary=false)
pars["tc1"] = Parameter(value=tc1, vary=false)
pars["ecc1"] = Parameter(value=ecc1, vary=false)
pars["w1"] = Parameter(value=w1, vary=false)
pars["k1"] = Parameter(value=8, vary=true)
add_prior!(pars["k1"], Priors.Positive())

# Planet c (not detected through this analysis)
planets[2] = TCOrbitBasis()
per2 = 18.858991
tc2 = 2458342.2243

pars["per2"] = Parameter(value=per2, vary=false)
pars["tc2"] = Parameter(value=tc2, vary=false)
pars["ecc2"] = Parameter(value=0, vary=false)
pars["w2"] = Parameter(value=π, vary=false)
pars["k2"] = Parameter(value=5, vary=true)
add_prior!(pars["k2"], Priors.Positive())

# Gamma offsets, don't include gamma dot or ddot yet
for instname ∈ keys(data)
    pars["gamma_$instname"] = Parameter(value=median(data[instname].rv) + π / 100, vary=true)
end

# Fix jitter to zero by default
for instname ∈ keys(data)
    pname = "jitter_$instname"
    pars[pname] = Parameter(value=0, vary=false)
end

# Increase HIRES error bars
pars["jitter_HIRES"] = Parameter(value=5, vary=false)

# # GP amps
gp_par_names = String[]
for (i, instname) ∈ enumerate(keys(data))
    pname = "gp_amp_" * instname
    push!(gp_par_names, pname)
    pars[pname] = Parameter(value=std(data[instname].rv), vary=true)
    add_prior!(pars[pname], Priors.JeffreysGD(1, 600))
    pars[pname].latex_str = L"\eta_{\sigma,%$instname}"
end

# GP decay time
pars["gp_decay"] = Parameter(value=100, vary=false)
add_prior!(pars["gp_decay"], Priors.Uniform(20, 2_000))
pars["gp_decay"].latex_str = L"\eta_{\\tau}"
push!(gp_par_names, "gp_decay")

# GP smoothness
pars["gp_per_length"] = Parameter(value=0.28, vary=false)
pars["gp_per_length"].latex_str = L"\eta_{\ell}"
push!(gp_par_names, "gp_per_length")

# GP period
pars["gp_per"] = Parameter(value=4.836, vary=true)
add_prior!(pars["gp_per"], Priors.Gaussian(pars["gp_per"].value, 0.001))
pars["gp_per"].latex_str = L"\eta_{P}"
push!(gp_par_names, "gp_per")

# noise, model, like, post
# Each spectrograph gets its own like here
likes = Dict{String, RVLikelihood}()
model = RVModel(planets, 0, t0=2450000.0)
gp = GaussianProcess(ChromaticKernelJ1(gp_par_names))
likes["rvs_j1"] = RVLikelihood(data, model, gp)

post = RVPosterior(likes)
update_latex_strings!(post, pars)

# Map fit
map_result = run_mapfit(post, pars)

# Alias best fit params
pbest = map_result.pbest
println(pbest)

# RV plots, also color GPs to match spectrographs
star_name = "AU_Mic"
data_colors = Dict("HIRES" => COLORS_HEX_GADFLY[1], "TRES" => COLORS_HEX_GADFLY[2], "SPIRou" => COLORS_HEX_GADFLY[3], "iSHELL" => COLORS_HEX_GADFLY[4])
gp_colors = Dict("GP HIRES" => COLORS_HEX_GADFLY[1], "GP TRES" => COLORS_HEX_GADFLY[2], "GP SPIRou" => COLORS_HEX_GADFLY[3], "GP iSHELL" => COLORS_HEX_GADFLY[4])
fig_full, fig_res = plot_rvs_full(post, pbest, data_colors=data_colors, gp_colors=gp_colors, gp_Δt=50, gp_δt=0.02)
PlotlyJS.savefig(fig_full, "$(star_name)_rvs.html")
PlotlyJS.savefig(fig_res, "$(star_name)_rvs_residuals.html")
figs_phased, figs_phased_res = plot_rvs_phased_all(post, pbest, data_colors=data_colors, titles=true, star_name="AU Mic")
for i=1:length(figs_phased)
    PlotlyJS.savefig(figs_phased[i], "$(star_name)_$(ALPHABET[i])_rvs_phased.html")
    PlotlyJS.savefig(figs_phased_res[i], "$(star_name)_$(ALPHABET[i])_rvs_phased_residuals.html")
end

# MCMC
mcmc_result = run_mcmc(post, pbest, n_burn_steps=500, n_τs_thresh=40, n_min_steps=1000, n_max_steps=80_000)

# Corner plot
fig = corner_plot(post, mcmc_result)
fig.savefig("$(star_name)_corner.png", dpi=200)
```

# Example 4: AU Mic, with J2 kernel.

```julia
using RVModelingToolkit
using Statistics
using DataStructures
using PlotlyJS

# Data
fname = "aumic_rvs_20220710_example.txt"
data = read_radvel_file(fname)
data["HIRES"].λ = 550.0
data["TRES"].λ = 650
data["SPIRou"].λ = 1650.0
data["iSHELL"].λ = 2350.0

# Init parameters and planets dictionary
pars = Parameters()
planets = OrderedDict{Int, OrbitBasis}()

# Planet 1 (detected at a few sigma here)
planets[1] = TCOrbitBasis()
per1 = 8.4629991
tc1 = 2458330.39046
ecc1 = 0.189
w1 = 1.5449655 + π

pars["per1"] = Parameter(value=per1, vary=false)
pars["tc1"] = Parameter(value=tc1, vary=false)
pars["ecc1"] = Parameter(value=ecc1, vary=false)
pars["w1"] = Parameter(value=w1, vary=false)
pars["k1"] = Parameter(value=8, vary=true)
add_prior!(pars["k1"], Priors.Positive())

# Planet c (not detected through this analysis)
planets[2] = TCOrbitBasis()
per2 = 18.858991
tc2 = 2458342.2243

pars["per2"] = Parameter(value=per2, vary=false)
pars["tc2"] = Parameter(value=tc2, vary=false)
pars["ecc2"] = Parameter(value=0, vary=false)
pars["w2"] = Parameter(value=π, vary=false)
pars["k2"] = Parameter(value=5, vary=true)
add_prior!(pars["k2"], Priors.Positive())

# Gamma offsets, don't include gamma dot or ddot yet
for instname ∈ keys(data)
    pars["gamma_$instname"] = Parameter(value=median(data[instname].rv) + π / 100, vary=true)
end

# Fix jitter to zero by default
for instname ∈ keys(data)
    pname = "jitter_$instname"
    pars[pname] = Parameter(value=0, vary=false)
end

# Increase HIRES error bars
pars["jitter_HIRES"] = Parameter(value=5, vary=false)

# GP
gp_par_names = String[]
push!(gp_par_names, "gp_amp_0")
pars["gp_amp_0"] = Parameter(value=150, vary=true)
add_prior!(pars["gp_amp_0"], Priors.JeffreysGD(1, 600))
pars["gp_amp_0"].latex_str = L"\eta_{\sigma,0}"

# GP
push!(gp_par_names, "gp_amp_pl")
pars["gp_amp_pl"] = Parameter(value=1.5, vary=true)
add_prior!(pars["gp_amp_pl"], Priors.Uniform(-2, 4))
pars["gp_amp_pl"].latex_str = L"\eta_{\lambda}"

# GP decay time
pars["gp_decay"] = Parameter(value=100, vary=false)
add_prior!(pars["gp_decay"], Priors.Uniform(20, 2_000))
pars["gp_decay"].latex_str = L"\eta_{\\tau}"
push!(gp_par_names, "gp_decay")

# GP smoothness
pars["gp_per_length"] = Parameter(value=0.28, vary=false)
pars["gp_per_length"].latex_str = L"\eta_{\ell}"
push!(gp_par_names, "gp_per_length")

# GP period
pars["gp_per"] = Parameter(value=4.836, vary=true)
add_prior!(pars["gp_per"], Priors.Gaussian(pars["gp_per"].value, 0.001))
pars["gp_per"].latex_str = L"\eta_{P}"
push!(gp_par_names, "gp_per")

# noise, model, like, post
likes = OrderedDict{String, RVLikelihood}()
model = RVModel(planets, 0, t0=2450000.0)
gp = GaussianProcess(ChromaticKernelJ2(gp_par_names, 550.0))
likes["rvs_j2"] = RVLikelihood(data, model, gp)

post = RVPosterior(likes)
update_latex_strings!(post, pars)

# Map fit
map_result = run_mapfit(post, pars)

# Alias best fit params
pbest = map_result.pbest
println(pbest)

# RV plots, also color GPs to match spectrographs
star_name = "AU_Mic"
data_colors = Dict("HIRES" => COLORS_HEX_GADFLY[1], "TRES" => COLORS_HEX_GADFLY[2], "SPIRou" => COLORS_HEX_GADFLY[3], "iSHELL" => COLORS_HEX_GADFLY[4])
gp_colors = Dict("GP 550" => COLORS_HEX_GADFLY[1], "GP 650" => COLORS_HEX_GADFLY[2], "GP 1650" => COLORS_HEX_GADFLY[3], "GP 2350" => COLORS_HEX_GADFLY[4])
fig_full, fig_res = plot_rvs_full(post, pbest, data_colors=data_colors, gp_colors=gp_colors, gp_Δt=50, gp_δt=0.02)
PlotlyJS.savefig(fig_full, "$(star_name)_rvs.html")
PlotlyJS.savefig(fig_res, "$(star_name)_rvs_residuals.html")
figs_phased, figs_phased_res = plot_rvs_phased_all(post, pbest, data_colors=data_colors, titles=true, star_name="AU Mic")
for i=1:length(figs_phased)
    PlotlyJS.savefig(figs_phased[i], "$(star_name)_$(ALPHABET[i])_rvs_phased.html")
    PlotlyJS.savefig(figs_phased_res[i], "$(star_name)_$(ALPHABET[i])_rvs_phased_residuals.html")
end

# MCMC
mcmc_result = run_mcmc(post, pbest, n_burn_steps=500, n_τs_thresh=40, n_min_steps=1000, n_max_steps=80_000)

# Corner plot
fig = corner_plot(post, mcmc_result)
fig.savefig("$(star_name)_corner.png", dpi=200)
```