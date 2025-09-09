import TurbulenceFlux:
    _default_phase_kernel, next2pow_padding, mean_wind_amplitude, mean_density
work_dim = 64
kernel_dim = 16
kernel_type = :rect
kernel_params = [5]
padding = next2pow_padding(kernel_dim, 0)
tp = TimeParams(kernel_dim, kernel_type, kernel_params; padding)
phase = _default_phase_kernel(kernel_dim)

# Mean Wind Amplitude
ws = [circshift(vcat(1, zeros(work_dim - 1)), phase) for _ = 1:3]
target = sqrt.((averaging_kernel(tp) .^ 2) * 3)
mwa = mean_wind_amplitude(ws, tp)
@test isapprox(mwa[1:kernel_dim], target)

# Mean Density
P = circshift(vcat(TurbulenceFlux.R, zeros(work_dim - 1)), phase)
T = ones(work_dim)
target = averaging_kernel(tp)
md = mean_density(P, T, tp)
@test isapprox(md[1:kernel_dim], target)
