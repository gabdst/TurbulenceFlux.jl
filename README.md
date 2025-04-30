


# TurbulenceFlux.jl  [![DOI](https://zenodo.org/badge/733581341.svg)](https://doi.org/10.5281/zenodo.15310755)

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

TurbulenceFlux.jl is a Julia package designed for high-resolution turbulence analysis and flux estimation using flux tower measurements of wind speeds and concentrations. This package is currently in its alpha stage, meaning it is under active development and significant modifications are expected.

## Important Notice

As an alpha release, this package may contain bugs, lack features, and undergo substantial changes in its API and functionality. We welcome feedback and contributions to help improve and stabilize the package.

## Usage

Below is a quick usage guide to get you started.

```julia
using TurbulenceFlux

# Assuming u, v, w are the three wind speed components (m/s),
# P (Pa) and T (K) are pressure and temperature signals,
# C (umol/mol) is the CO2 concentration

work_dim = length(u) # size of signals
fs = 20 # Sampling frequency
ref_dist = 10 # Distance of reference (m), e.g., tower flux height - displacement height

# Defining time decomposition parameters
Δt = 60 * fs # 1min time sampling corresponding to a 1min time flux
kernel_dim = work_dim
σ = 10 * 60 * fs / kernel_dim # 10min averaging length

# Gaussian averaging kernel
kernel_dim = work_dim # max time support
time_params = (
    kernel_type = :gaussian, # Gaussian type kernel
    kernel_dim,
    kernel_params = [4 * Δt / kernel_dim], # Ratio of max time support.
    Δt,       # 1min time sampling
)
time_params = (kernel_type = :gaussian, kernel_dim = kernel_dim, kernel_params = [σ], Δt = Δt)

# Defining scale decomposition parameters
wave_dim = work_dim
scale_params = (
    β = 2, # First wavelet shape parameter
    γ = 3, # Second wavelet shape parameter
    J = floor(Int, log2(wave_dim)), # Number of octaves
    Q = 4, # Number of inter-octaves
    fmin = 2 * fs / (wave_dim), # Minimum frequency peak
    fmax = fs / 2, # Maximum Frequency peak
    fs = fs,
    wave_dim, # max time support
)

# Prepare density and mean wind amplitude
# (for simplicity here using the previous time-decomposition)
mean_wind = compute_wind_amplitude([u v w], time_params)
density = compute_density(P, T, time_params)

# time and normalized frequency coordinates
to_eta(i_t, j_ξ) = log10((ref_dist * freq_peak[j_ξ]) / mean_wind[i_t]) # ∼ log(ref_dist/eddy_dim)
S = (length(time_sampling), length(freq_peak)) # Dimension
CI = CartesianIndices(S)
t = map(c -> time_h[time_sampling[c[1]]], CI) # get the time values
eta = map(c -> to_eta(c[1], c[2]), CI)

# time-frequency decomposition of CO2 flux
time_sampling, (freq_peak, σ_t), decomp_FC = timescale_flux_decomp(w, C, time_params, scale_params)
decomp_FC = decomp_FC .* density # Convert to flux units

# Computation of the amplitude of Reynold's tensor vertical component τ_w
_, _, τ_w = amplitude_reynolds_w(u, v, w, time_params_turbu, scale_params)

# Turbulence extraction based on the laplacian of log(τ_w)
(masks, Δτ, itp) = turbu_extract_laplacian(t, eta, log10.(τ_w), δ_Δτ = 1, δ_τ = 1e-3)

# Scale integration of the flux given the turbulence mask
FC = time_integrate_flux(decomp_FC, masks.turbulence)

FC # CO2 flux
```

## Contributing

We encourage contributions from the community to help improve TurbulenceFlux.jl. If you have any suggestions, bug reports, or would like to contribute code, please feel free to open an issue or submit a pull request.
