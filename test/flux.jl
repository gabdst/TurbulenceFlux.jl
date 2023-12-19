using Test
push!(LOAD_PATH,"../")
import TurbulenceFlux: _init_parameters_wave,_init_mappings_wave,_phase_kernel,WaveletConv,GaussConv
import TurbulenceFlux: timescale_flux_decomp
using LinearAlgebra
using FFTW

work_dim=8192
wave_dim=1024
kernel_dim=513
β=1
γ=3
J=8
Q=8
wmin=2pi/wave_dim
wmax=pi
Δt=4

if rem(work_dim,Δt) !== 0
  throw(error("The time interval Δt should divide input signal size"))
end

# Checking convolution kernels are working
## Self-dual property of Wavelet Convolution Kernel
wave_params=_init_parameters_wave(;β=β,γ=γ,J=J,Q=Q,wmin=5*wmin,wmax=wmax)
r=(1:work_dim) .+ (wave_dim-1)
phase_output=[_phase_kernel(wave_dim) for _ in 1:2]
### With low pass
wave_params_map=_init_mappings_wave(wave_dim,wave_params,with_low_pass=true)
WaveC=WaveletConv((work_dim,1),wave_dim,wave_params_map,phase_output=phase_output,r=r)

# Check constant fourier spectrum
waves=map(ξ->abs2.(rfft(WaveC.kernel_func(WaveC,WaveC.kernel_params_map[1](ξ))[:])),wave_params)
@test isapprox(sum(waves),ones(length(waves[1])),rtol=0.1)

δ=vcat(1,zeros(work_dim-1))
δ_ξ = [ WaveC(WaveC(δ,ξ),ξ) for ξ in wave_params]
@test isapprox(δ,sum(δ_ξ),rtol=0.1) # OK if less than 0.1% relative error on norm(x-y)/norm(x)
### No low pass
wave_params_map=_init_mappings_wave(wave_dim,wave_params,with_low_pass=false)
WaveC=WaveletConv((work_dim,1),wave_dim,wave_params_map,phase_output=phase_output,r=r)

# Check constant fourier spectrum
waves=map(ξ->abs2.(rfft(WaveC.kernel_func(WaveC,WaveC.kernel_params_map[1](ξ))[:])),wave_params)
@test isapprox(sum(waves),ones(length(waves[1])),rtol=0.1)

δ=vcat(1,zeros(work_dim-1))
δ_ξ = [ WaveC(WaveC(δ,ξ),ξ) for ξ in wave_params]
@test isapprox(δ,sum(δ_ξ),rtol=0.1) # OK if less than 0.1% relative error on norm(x-y)/norm(x)

## Consistent time-averaging property of Averaging Convolution Kernel
r=(1:work_dim) .+ (kernel_dim-1)
time_sampling=1:Δt:work_dim
phase_output=[_phase_kernel(kernel_dim) for _ in 1:2]
KernelC=GaussConv((work_dim,1),kernel_dim,r=r,phase_output=phase_output,kernel_params_map=(identity,identity),kernel_init_params=()->nothing)
δ_train=zeros(work_dim)
δ_train[time_sampling].=1 # Flux of one every Δt => should get a constant average flux of 1/Δt
σ=1/128
δ_train_KC=KernelC(δ_train,[σ]) # The gaussian reaches 0.6 at time index u=kernel_dim*sigma, if u/Δt is too low, it should break the consistency
@test all(isapprox.(δ_train_KC,1/Δt,rtol=0.1))

# Final check, putting together Wavelet and Averaging Kernels
## Checking that we recover the mean flux over the entire period for the dirac train above
δ_train_ξ = [ WaveC(δ_train,ξ) .^2 for ξ in wave_params] # Square 
δ_train_ξ = [ KernelC(x,[σ]) for x in δ_train_ξ] # Average it
δ_train_ξ = [ x[time_sampling] for x in δ_train_ξ] # Sample it
flux_value = sum(sum(δ_train_ξ))/length(time_sampling) #  Sum it, should be 1/Δt, the average flux over the entire period is 1/Δt for the dirac_train
@test isapprox(flux_value,1/Δt,rtol=0.1)

σ=1/(16)
scale_params=(wave_dim=wave_dim,β=β,γ=γ,J=J,Q=Q,wmin=wmin,wmax=wmax)
time_params=(kernel_dim=kernel_dim,kernel_type=:gaussian,kernel_params=[σ],Δt=Δt)

time_sampling,freq_peak,flux_TS=timescale_flux_decomp(δ_train,δ_train,time_params,scale_params)
flux_value=sum(flux_TS)/size(flux_TS,1)
@test isapprox(flux_value,1/Δt,rtol=0.1)
