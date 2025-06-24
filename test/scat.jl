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
sigma_waves = gmw_frame.sigma_waves
sigma_min = minimum(sigma_waves)
many_averaging_kernel = [
    TurbulenceFlux.gausskernel(kernel_dim, 10 * sigma / kernel_dim, 0) for
    sigma in sigma_waves
]
averaging_kernel = TurbulenceFlux.gausskernel(kernel_dim, 10 * sigma_min / kernel_dim, 0)

out0 = TurbulenceFlux.cross_scalogram(x, x, deltat, gmw_frame.gmw_frame, averaging_kernel)
out1 = TurbulenceFlux.cross_scalogram(x, y, deltat, gmw_frame, averaging_kernel)

out2 = TurbulenceFlux.cross_scalogram(
    [x, y],
    [(1, 1), (1, 2)],
    deltat,
    gmw_frame,
    many_averaging_kernel,
)
