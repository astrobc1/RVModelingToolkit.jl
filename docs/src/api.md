API
===

### RV Data

```@docs
RVData
CompositeRVData
get_times
get_rvs
get_rverrs
get_instnames
get_λs
read_radvel_file
```

### Parameters

```@docs
Parameter
Parameters
```

#### Priors

```@docs
Prior
Priors.Gaussian
Priors.Uniform
Priors.JeffreysGD
Priors.Positive
logprob
```

### The Keplerian Model

```@docs
RVModel
build
build_planet
build_planets
build_trend_zero
build_global_trend
true_anomaly
solve_kepler
tc_to_tp
tp_to_tc
disable_planet_parameters!
solve_kepler_all_times
```

#### Orbit Bases

```@docs
OrbitBasis
StandardOrbitBasis
TCOrbitBasis
parameters
convert_basis
```

### Gaussian Processeses

```@docs
NoiseKernel
GaussianProcess
QuasiPeriodic
ChromaticKernelJ1
ChromaticKernelJ2
compute_cov_matrix
predict
compute_stationary_dist_matrix
```

### Posteriors, Likelihoods

```@docs
RVLikelihood
RVPosterior
compute_logL
compute_logaprob
compute_prior_logprob
compute_residuals
compute_data_errors
compute_noise_components
compute_redχ2
compute_bic
compute_aicc
```


### Fitting

```@docs
run_mapfit
run_mcmc
model_comparison
```

### Plotting

```@docs
plot_rvs_phased
plot_rvs_phased_all
plot_rvs_full
corner_plot
```

### Brute Force Periodogram
```@docs
brute_force_periodogram
```

### Planetary and Orbital Measurements (Coming soon)

#### Mass, Radius (C-K), Density, SMA ...