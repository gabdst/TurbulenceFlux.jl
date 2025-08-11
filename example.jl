using JLD2
using DataFrames
using TurbulenceFlux
using GLMakie
using Statistics


# Some data available at https://drive.proton.me/urls/VM51CC7Y6G#q9ykofXnuqys
data = jldopen("data_sample.jld2")
dates = keys(data)
d = first(dates)
signals = data[d]["Signals"]
u = signals.U
v = signals.V
w = signals.W
P = signals.P * 1e3 # To Pa
T = signals.T .+ 273.15 # To K
C = signals.CO2 # umol/mol

# Assuming u, v, w are the three wind speed components (m/s),
# P (Pa) and T (K) are pressure and temperature signals,
# C (umol/mol) is the CO2 concentration

work_dim = length(u) # size of signals (24h)
fs = 20 # Sampling frequency
ref_dist = 10 # Distance of reference (m), e.g., tower flux height - displacement height

# Defining time decomposition parameters
dt = 60 * fs # 1min time sampling corresponding to a 1min time flux
kernel_dim = work_dim
σ = 10 * 60 * fs / kernel_dim # 10min averaging length

# Gaussian averaging kernel
kernel_dim = work_dim # max time support
time_params = (
    kernel_type = :gaussian, # Gaussian type kernel
    kernel_dim,
    kernel_params = [30 * dt / kernel_dim], # Ratio of max time support.
    Δt = dt,       # 1min time sampling
)

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

# time-frequency decomposition of CO2 flux
time_sampling, (freq_peak, sigmat), decomp_FC =
    timescale_flux_decomp(w, C, time_params, scale_params, with_info = true)
decomp_FC = decomp_FC .* density # Convert to flux units

# Computation of the amplitude of Reynold's tensor vertical component τ_w
_, _, tauw = amplitude_reynolds_w(u, v, w, time_params, scale_params)

# time and normalized frequency coordinates
to_eta(i, j) = log10((ref_dist * freq_peak[j]) / mean_wind[i]) # ∼ log(ref_dist/eddy_dim)
S = (length(time_sampling), length(freq_peak)) # Dimension
CI = CartesianIndices(S)
time_h = (0:(work_dim-1)) / (fs * 3600)
t = map(c -> time_h[time_sampling[c[1]]], CI) # get the time values
eta = map(c -> to_eta(c[1], c[2]), CI)

# Turbulence extraction based on the laplacian of log(τ_w)
(masks, deltatau, itp) = turbu_extract_laplacian(t, eta, log10.(tauw), δ_Δτ = 1, δ_τ = 1e-3)

# Scale integration of the flux given the turbulence mask
FC = time_integrate_flux(decomp_FC, masks.turbulence)

cmap_flux = Makie.Reverse(:bam)
cmap_tau = Makie.Reverse(:roma)
function plot_contour(
    x,
    y,
    z;
    g = nothing,
    xlabel = "",
    ylabel = "",
    zlabel = "",
    mask = nothing,
    mask_alpha = 0.25,
    vmin = -5 * std(z),
    vmax = 5 * std(z),
    length = 20,
    levels = range(vmin, vmax, length = length),
    colormap = cmap_tau,
)
    ax = Axis(g[1, 1]; xlabel, ylabel)
    co = contourf!(ax, x, y, z; levels, colormap)
    if !isnothing(mask)
        contourf!(ax, x, y, mask, colormap = [(:black, mask_alpha), (:white, 0)])
    end
    Colorbar(g[1, 2], co, label = zlabel)
    return g, ax
end

function plot_contour_line(args...; kwargs...)
    g, ax = plot_contour(args...; kwargs...)
    ax_line = Axis(g[2, 1], ylabel = kwargs[:zlabel], xlabel = kwargs[:xlabel]) # Prepare line Axis below
    return g, ax, ax_line
end

h(d) = d[:, 1:end-1]
etaref = h(eta)[div(size(eta, 1), 2), :]
tref = t[:, 1]
fig = Figure(size = (1000, 800))
g1 = GridLayout(fig[1, 1])
g2 = GridLayout(fig[2, 1])
g3 = GridLayout(fig[1:2, 2])
xlabel = "Time [h]"
ylabel = L"\eta"
g1, ax1 = plot_contour(
    tref,
    etaref,
    h(tauw);
    mask = h(masks.turbulence),
    g = g1,
    xlabel,
    ylabel,
    zlabel = L"\tau_w\,\mathrm{[m^2\,s^{-2}]}",
)
g2, ax2 = plot_contour(
    tref,
    etaref,
    h(deltatau);
    g = g2,
    xlabel,
    ylabel,
    zlabel = L"\Delta \tau_w\,\mathrm{[m^2\,s^{-2}]}",
)
g3, ax3, ax_line = plot_contour_line(
    tref,
    etaref,
    h(decomp_FC);
    mask = h(masks.turbulence),
    g = g3,
    xlabel,
    ylabel,
    zlabel = L"FC\,\mathrm{[umol\,m^{-2}\,s^{-1}]}",
    colormap = cmap_flux,
)
lines!(ax_line, tref, FC)
linkxaxes!(ax1, ax2, ax3, ax_line)

