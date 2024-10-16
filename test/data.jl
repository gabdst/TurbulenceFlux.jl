using JLD2
using DataFrames
push!(LOAD_PATH,"../")
using TurbulenceFlux
file=jldopen( "../data/data_sample.jld2")
days=collect(keys(file))
d=days[6]
data=file[d]
close(file)

fs=20 # Sampling Frequency (Hz)
displacement_height = 14.7 # Displacement Height (m)
instrument_height = 30.75 # Measurement Height
canopy_height = 21.5 # Height of canopy forest
roughness_length = 4.05

work_dim=length(data["Signals"].Date)

(;U,V,W,T,CO2,H2O,P)=data["Signals"];
# some conversions
T=T .+ 274.15 # °C TO K
P=1000*P; # kPa to Pa

Δt=1*60*fs # 1 min Sampling Step
kernel_dim = work_dim # max time support
time_params = (
    kernel_type = :gaussian, # Gaussian type kernel
    kernel_dim, 
    kernel_params = [4*Δt/kernel_dim], # Ratio of max time support.
    Δt,       # 1min time sampling
    );

wave_dim = work_dim
scale_params = (
    β = 2, # First wavelet shape parameter
    γ = 3, # Second wavelet shape parameter
    J = floor(Int,log2(wave_dim)), # Number of octaves
    Q = 4, # Number of inter-octaves
    fmin = 2*fs/(wave_dim), # Minimum frequency peak
    fmax = fs/2, # Maximum Frequency peak
    fs = fs,
    wave_dim, # max time support
    );


gmw_params=TurbulenceFlux.gmw_grid(scale_params.β,scale_params.γ,scale_params.J,scale_params.Q,scale_params.fmin*2pi/fs,scale_params.fmax*2pi/fs)
gmw_frame = TurbulenceFlux.GMWFrame(work_dim,gmw_params)
σ_waves = gmw_frame.σ_waves
averaging_kernel = TurbulenceFlux._gausskernel(work_dim,time_params.kernel_params[1],0)


using BenchmarkTools
out=[]
b_1= @benchmark begin
    decomp_T= timescale_flux_decomp(W,T,time_params,scale_params;with_info=true)
    push!(out,decomp_T[end])
end 
    
b_2 = @benchmark begin 
    decomp_T_cs=TurbulenceFlux.cross_scattering(W,T,Δt,gmw_frame.gmw_frame,averaging_kernel)
    push!(out,decomp_T_cs)
end 
