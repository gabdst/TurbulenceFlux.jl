using Test
import TurbulenceFlux

work_dim = 1024
wave_dim = work_dim
kernel_dim = work_dim
x = vcat(1, zeros(work_dim - 1))
y = circshift(x, 10)
b = 2
g = 3
J = floor(Int, log2(wave_dim))
Q = 4
wmin = 2pi / wave_dim
wmax = pi
deltat = 4
time_sampling = 1:deltat:work_dim

gmw_params = TurbulenceFlux.init_wavelet_parameters(b, g, J, Q, wmin, wmax)
gmw_frame = TurbulenceFlux.GMWFrame(wave_dim, gmw_params)
σ_waves = gmw_frame.σ_waves
σ_min = minimum(σ_waves)
many_averaging_kernel =
    [TurbulenceFlux._gausskernel(kernel_dim, 10 * σ / kernel_dim, 0) for σ in σ_waves]
averaging_kernel = TurbulenceFlux._gausskernel(kernel_dim, 10 * σ_min / kernel_dim, 0)

out0 = TurbulenceFlux.cross_scattering(x, x, deltat, gmw_frame.gmw_frame, averaging_kernel)
out1 = TurbulenceFlux.cross_scattering(x, y, deltat, gmw_frame, averaging_kernel)

out2 = TurbulenceFlux.cross_scattering(
    [x, y],
    [(1, 1), (1, 2)],
    deltat,
    gmw_frame,
    many_averaging_kernel,
)
