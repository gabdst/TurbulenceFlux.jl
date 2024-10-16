# Testing differentiability feature
import TurbulenceFlux
import TurbulenceFlux:_init_parameters_wave,GaussConv,WaveletConv
using ForwardDiff

work_dim = 1024
kernel_dim = 512
wave_dim = 512

# Dirac Test Signal
δ = vcat(1,zeros(work_dim-1))

GaussC = GaussConv((work_dim,1),(kernel_dim,1))
σ = [10/kernel_dim,0]

# Just checking that it executes without errors
@test begin
  d = ForwardDiff.jacobian(σ->GaussC(δ,σ),σ)
  true
end

β = 1
γ = 3
J = floor(Int,log2(wave_dim))
Q = 4
wmin = 2pi / wave_dim
wmax = pi

wave_params = _init_parameters_wave(; β = β, γ = γ, J = J, Q = Q, wmin = wmin, wmax = wmax)

WaveC = WaveletConv(
    (work_dim, 1),
    (wave_dim,size(wave_params,2))
)

@test begin
  d = ForwardDiff.jacobian(wave_params->WaveC.kernel_func(WaveC,wave_params),wave_params)
  d = ForwardDiff.jacobian(wave_params->WaveC(δ,wave_params),wave_params)
  true
end
