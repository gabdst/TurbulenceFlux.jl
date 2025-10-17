module TurbulenceFlux
using LinearAlgebra,
    NaNStatistics,
    Statistics,
    Rotations,
    Random,
    StatsFuns,
    FFTW,
    LoopVectorization,
    SparseArrays,
    PhysicalConstants,
    Unitful,
    DataInterpolations
import GeneralizedMorseWavelets as GMW
import Loess

const LAMBDA = 40660 / 1000 # "J.mmol^-1" latent heat of evaporation of water
const C_p = 29.07 # Molar Heat Capacity at constant pressure J.mol^-1.K^-1
const R = ustrip(PhysicalConstants.CODATA2018.R)

const CORRECTIONS = (:planar_fit, :despiking, :optim_timelag)
const mandatory_temp_variables = (; TA = u"°C", T_SONIC = u"°C")
const mandatory_variables = (;
    TIMESTAMP = NoUnits,
    U = u"m/s",
    V = u"m/s",
    W = u"m/s",
    PA = u"kPa",
    mandatory_temp_variables...,
)

const gas_variables = (; CO2 = u"μmol/mol", H2O = u"mmol/mol")
const output_variables = (;
    TIMESTAMP = NoUnits,
    ETA = NoUnits,
    RHO = u"mol/m^3",
    FC = u"μmol/m^2/s",
    FC_DT = u"μmol/m^2/s^2",
    FC_DSIGMA = u"μmol/m^2/s^2",
    FC_TF = u"μmol/m^2/s",
    FC_DT_TF = u"μmol/m^2/s^2",
    FC_DSIGMA_TF = u"μmol/m^2/s^2",
    WS = u"m/s",
    USTAR = u"m^2/s^2",
    H = u"W/m^2",
    H_DT = u"W/m^2/s",
    H_DSIGMA = u"W/m^2/s",
    H_TF = u"W/m^2",
    H_DT_TF = u"W/m^2/s",
    H_DSIGMA_TF = u"W/m^2/s",
    LE = u"W/m^2",
    LE_DT = u"W/m^2/s",
    LE_DSIGMA = u"W/m^2/s",
    LE_TF = u"W/m^2",
    LE_DT_TF = u"W/m^2/s",
    LE_DSIGMA_TF = u"W/m^2/s",
    TAUW = u"m^2/s^2",
    TAUW_DT = u"m^2/s^3",
    TAUW_DSIGMA = u"m^2/s^3",
    TAUW_TF = u"m^2/s^2",
    TAUW_DT_TF = u"m^2/s^3",
    TAUW_DSIGMA_TF = u"m^2/s^3",
    TAUW_TF_M = NoUnits,
    U_SIGMA = u"m/s",
    V_SIGMA = u"m/s",
    W_SIGMA = u"m/s",
)

struct ErrorVariableMissing <: Exception
    msg::String
end
function ErrorVariableMissing(var::Symbol)
    msg = """
    Mandatory variable $var is missing. Please check dataframe.
    """
    ErrorVariableMissing(msg)
end
function ErrorVariableMissing(var::Tuple{Vararg{Symbol}})
    msg = """
    At least one mandatory variable in $var is missing. Please check dataframe.
    """
    ErrorVariableMissing(msg)
end
function check_variables(df::Dict)
    var_names = keys(df)
    for v in keys(mandatory_variables)
        if v isa Symbol
            if v in keys(mandatory_temp_variables)
                !(isdisjoint(keys(mandatory_temp_variables), var_names)) ||
                    throw(ErrorVariableMissing(keys(mandatory_temp_variables)))
            else
                v in var_names || throw(ErrorVariableMissing(v))
            end
        else
            throw(error("Unexpected variable-name type"))
        end
    end
    return true
end

const InputSignals = Dict{Symbol,AbstractArray}

function get_var_names(df::Dict)
    var_names = collect(keys(df))
    popat!(var_names, findfirst(==(:TIMESTAMP), var_names))
    return var_names
end

@kwdef struct AuxVars
    fs::Integer # Sampling Frequency
    z_d::Float64 # Displacement Height
end


@kwdef mutable struct CorrectionParams
    timelag_max::Integer = 0
    fc_timelag::Real = 0.1 # Cutting frequency for timelag optimization, 0.1Hz by default
    rot_matrix::AbstractMatrix{<:Real} = zeros(Float64, 3, 3) # Rotation matrix used, zeros by default
    timelags::Dict{Symbol,Int} = Dict{Symbol,Int}()# timelags in number of samples by gas names, :CO2 => 4
    window_size_despiking::Integer = 200
    corrections::Vector{Symbol} = [CORRECTIONS...]
end
const QualityControl = Dict{Symbol,AbstractSparseArray{Bool,Int}}
get_qc(qc::QualityControl, var::Symbol) = get(qc, var, false)
get_qc(qc::QualityControl, var::Tuple{Vararg{Symbol}}) =
    reduce((a, b) -> a .|| b, map(a -> get_qc(qc, a), var))
function update_quality_control!(
    qc::QualityControl,
    var::Tuple{Vararg{Symbol}},
    mask::AbstractArray{Bool},
)
    for v in var
        update_quality_control!(qc, v, mask)
    end
end

function update_quality_control!(qc::QualityControl, var::Symbol, mask::AbstractArray{Bool})
    if var in keys(qc)
        qc[var] .= mask # In place modification
    else
        qc[var] = sparse(mask) # Add new
    end
end

include("utils.jl")
include("conv.jl")
include("graph.jl")
include("corrections.jl")
include("flux.jl")

export DecompParams,
    ScaleParams,
    TimeParams,
    GMWFrame,
    AuxVars,
    CorrectionParams,
    cross_scalogram,
    dt_cross_scalogram,
    dp_cross_scalogram,
    average,
    dt_average,
    dp_average,
    averaging_kernel,
    dt_averaging_kernel,
    dp_averaging_kernel,
    RepeatedMedianRegressor,
    flag_spikes,
    flag_nan,
    getparams,
    interpolate_errors!,
    optim_timelag,
    planar_fit,
    normalized_frequency,
    sigmas_wind,
    ustar,
    mean_wind,
    mean_density,
    flux_scale_integral,
    flux_scalogram,
    reynolds_w_scalogram,
    FluxEstimationMethod,
    ReynoldsEstimation,
    TurbuThreshold,
    TurbuLaplacian,
    turbulence_mask,
    estimate_flux,
    default_phase_kernel,
    next2pow_padding,
    mandatory_variables,
    mandatory_temp_variables,
    gas_variables,
    output_variables
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
