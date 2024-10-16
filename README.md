# TurbulenceFlux.jl
A julia package for high resolution turbulence analysis and flux estimation given flux tower measurements of wind speeds and concentrations.

## Usage

We give below a quick usage guide. For more details, a notebook is available under `nb/` with a link to download data samples.

```julia
using TurbulenceFlux
# Assuming u,v,w the three wind speed components (m/s),
# P (Pa) and T (K) pressure and temperature signals,
# C (umol/mol) the CO2 concentration

work_dim=length(u) # size of signals
fs=20 # Sampling frequency
ref_dist=10 # Distance of reference (m), e.g. tower flux height - displacement height

# Defining time decomposition parameters
Δt=60*fs # 1min time sampling corresponding to a 1min time flux
kernel_dim=6*60*60*fs # 6h of max time support
σ=10*60*fs/kernel_dim # 10min averaging length 
# Gaussian averaging kernel reaching 0.6 at 10min
time_params=(kernel_type=:gaussian,kernel_dim=kernel_dim,kernel_params=[σ],Δt=Δt);

# Defining scale decomposition parameters
wave_dim=6*60*60*fs # 6h of max time support 
scale_params=(β=2,γ=3,J=25,Q=4,wmin=2*pi/wave_dim,wmax=pi,fs=fs,wave_dim=wave_dim) # Wavelet parameters and minimum and maximum frequency peaks

# Time-scale decomposition of CO2 flux
time_sampling,freq_peak,timescale_CO2=timescale_flux_decomp(w,C,time_params,scale_params)

# Prepare density and mean wind amplitude 
# (for simplicity here using the previous time-decomposition)
mean_wind = compute_wind_amplitude(u,v,w,time_params)
density = compute_density(P,T,time_params)

# Computation of the amplitude of Reynold's tensor vertical component τ_w
_,_,τ_w = amplitude_reynolds_w(U,V,W,time_params_turbu,scale_params)

# Turbulence extraction based on the laplacian of log(τ_w)
(masks,tau_rey)=turbu_extract_laplacian(τ_w; time_sampling=time_sampling, freq_peak=freq_peak, ref_dist=ref_dist, mean_wind=mean_wind)
Δτ=tau_rey[4]# laplacian

# Scale integration of the flux given the turbulence mask
F_CO2,units_CO2=time_integrate_flux(timescale_CO2,masks[3],density,:CO2)

F_CO2 # CO2 flux
```
