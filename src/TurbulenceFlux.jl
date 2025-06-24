module TurbulenceFlux
include("conv.jl")
include("graph.jl")
include("flux.jl")
include("diff.jl")

export timescale_flux_decomp,
    init_averaging_conv_kernel,
    time_integrate_flux,
    turbulence_mask_extraction,
    turbu_extract_threshold,
    turbu_extract_diffusion,
    turbu_extract_laplacian,
    compute_density,
    compute_wind_amplitude,
    amplitude_reynolds_w,
    get_timescale_mask,
    find_nan_regions,
    optim_timelag,
    tofluxunits
end
