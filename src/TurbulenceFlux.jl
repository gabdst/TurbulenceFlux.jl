module TurbulenceFlux
using LinearAlgebra,
    NaNStatistics,
    Statistics,
    Rotations,
    Random,
    StatsFuns,
    FFTW,
    LoopVectorization,
    PhysicalConstants,
    Unitful,
    DataInterpolations,
    DataFrames
import GeneralizedMorseWavelets as GMW
import Loess

const LAMBDA = 40660 / 1000 # "J.mmol^-1" latent heat of evaporation of water
const C_p = 29.07 # Molar Heat Capacity at constant pressure J.mol^-1.K^-1
const R = ustrip(PhysicalConstants.CODATA2018.R)

include("utils.jl")
include("conv.jl")
include("graph.jl")
include("corrections.jl")
include("flux.jl")
include("diff.jl")

export DecompParams,
    ScaleParams,
    TimeParams,
    GMWFrame,
    cross_scalogram,
    averaging_kernel,
    RepeatedMedianRegressor,
    flag_spikes,
    flag_nan,
    interpolate_errors!,
    optim_timelag,
    planar_fit,
    mean_wind,
    mean_density,
    flux_scale_integral,
    flux_scalogram,
    reynolds_w_scalogram
end

# export timescale_flux_decomp,
#     init_averaging_conv_kernel,
#     time_integrate_flux,
#     turbulence_mask_extraction,
#     turbu_extract_threshold,
#     turbu_extract_diffusion,
#     turbu_extract_laplacian,
#     compute_density,
#     compute_wind_amplitude,
#     amplitude_reynolds_w,
#     get_timescale_mask,
#     find_nan_regions,
#     optim_timelag,
#     tofluxunits
# end
