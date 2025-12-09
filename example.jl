push!(LOAD_PATH, ".")
using TurbulenceFlux

using Pkg
Pkg.activate()
Pkg.add(["JLD2", "WGLMakie", "Statistics"])

using JLD2
using WGLMakie
using Statistics
## Some Plotting Recipes
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

########## DATA SAMPLES #############
# Some data available at https://drive.proton.me/urls/0NHRADNYTC#5q9CLk4V0sbA
data = jldopen("data_sample.hdf5")
dates = keys(data)
d = first(dates)
signals = data[d]["Signals"]
# Converting to Dictionnary
signals = Dict(Symbol(k) => signals[k] for k in keys(signals))

### FLUX ESTIMATION PARAMETERS ######
work_dim = length(signals[:W])
# Auxilliary variables
fs = 20 # Hz, sampling frequency
z_d = 10  # m, displacement height
aux = AuxVars(; fs, z_d)
cp = CorrectionParams()
# Wavelet Parameters
wave_dim = 24 * 60 * 60 * fs # max wavelet duration of 10 hours
b = 1
g = 3
J = floor(Int, log2(wave_dim))
Q = 2
fmin = (2 * fs / wave_dim) / fs # The lowest fourier frequency is fs/wave_dim
fmax = (fs / 2) / fs
sp = ScaleParams(b, g, J, Q, fmin, fmax, wave_dim)

# Time-Averaging Parameters
kernel_dim = 24 * 60 * 60 * fs # Max averaging length of 10 hours
sigma = 30 * 60 * fs # 30 min averaging
avg_kernel = GaussAvg(kernel_dim, sigma)
dt = fs * 60 # 1 minute time sampling
tp = TimeParams(avg_kernel; dt)
dp = DecompParams(sp, tp)
# Using the same Time-Averaging parameters for auxilliary estimates ( V_SIGMA,W_SIGMA,USTAR,RHO, etc...)
tp_aux = tp

# Method Parameters
method = TurbuLaplacian(; tr_tau = 1.0e-3, tr_dtau = 1, dp, tp_aux)

######## FLUX ESTIMATION #######
results = estimate_flux(signals, aux, cp, method)
tauw = results.estimate.TAUW_TF
dtauw = results.estimate.DTAUW_TF
FC = vec(results.estimate.FC)
FC_TF = results.estimate.FC_TF
tauw_m = results.estimate.TAUW_TF_M

h(d) = d[:, 1:(end - 1)] # Removal of lowest frequency band (mean)
eta = log10.(results.estimate.ETA) # Time-varying normalized frequency
etaref = h(eta)[div(size(eta, 1), 2), :] # Normalized frequency estimated at noon
tref = (0:dt:(work_dim - 1)) * 1 / fs / (60 * 60)
fig = Figure(size = (1000, 600));
g1 = GridLayout(fig[1, 1])
g2 = GridLayout(fig[2, 1])
g3 = GridLayout(fig[1:2, 2])
xlabel = "Time [h]"
ylabel = L"\log_{\mathrm{10}}\, \eta"
g1, ax1 = plot_contour(
    tref,
    etaref,
    h(tauw);
    mask = h(tauw_m),
    g = g1,
    xlabel,
    ylabel,
    zlabel = L"\tau_w\,\mathrm{[m^2\,s^{-2}]}",
)
g2, ax2 = plot_contour(
    tref,
    etaref,
    h(dtauw);
    g = g2,
    xlabel,
    ylabel,
    zlabel = L"\Delta \tau_w\,\mathrm{[m^2\,s^{-2}]}",
)
g3, ax3, ax_line = plot_contour_line(
    tref,
    etaref,
    h(FC_TF);
    mask = h(tauw_m),
    g = g3,
    xlabel,
    ylabel,
    zlabel = L"FC\,\mathrm{[umol\,m^{-2}\,s^{-1}]}",
    colormap = cmap_flux,
)
lines!(ax_line, tref, FC)
linkxaxes!(ax1, ax2, ax3, ax_line)
fig
