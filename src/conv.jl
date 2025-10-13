function next2pow_padding(N, init_padding)
    M = N + init_padding
    return nextpow(2, M) - M
end

default_phase_kernel(kernel_dim) =
    iseven(kernel_dim) ? div(kernel_dim, 2) - 1 : div(kernel_dim, 2)
_copy_real!(x::AbstractArray{Float64}, y::AbstractArray{ComplexF64}) = map!(real, x, y)

mutable struct CConv{T<:Union{Float64,ComplexF64}}
    const input_dim::Int64
    const output_dim::Int64
    const padding::Int64
    const crop_range::Tuple{Int64,Int64}
    # Internals
    const buff_in::Vector{ComplexF64}
    const pfft_in::AbstractFFTs.Plan{ComplexF64}
    const buff_ker::Vector{ComplexF64}
    const pfft_ker::AbstractFFTs.Plan{ComplexF64}
    const pifft_in::AbstractFFTs.Plan{ComplexF64}
    const buff_out::Vector{T}
end

function CConv(
    ::Type{T},
    input_dim::Int64;
    padding = 0,
    crop_range = (1, input_dim),
) where {T<:Union{Float64,ComplexF64}}
    conv_dim = (input_dim + padding)
    # Setting up internals in keywords 
    buff_in = Vector{ComplexF64}(undef, conv_dim)
    pfft_in = plan_fft!(buff_in)
    buff_ker = Vector{ComplexF64}(undef, conv_dim)
    pfft_ker = plan_fft!(buff_ker)
    pifft_in = plan_ifft!(buff_in)
    output_dim = crop_range[2] - crop_range[1] + 1
    buff_out = Array{T}(undef, output_dim)
    return CConv{T}(
        input_dim,
        output_dim,
        padding,
        crop_range,
        buff_in,
        pfft_in,
        buff_ker,
        pfft_ker,
        pifft_in,
        buff_out,
    )
end

function _load_sig!(C::CConv, X::AbstractVector{<:Real})
    fill!(C.buff_in, 0.0) # Prepare zero padding
    copy!(view(C.buff_in, axes(X, 1)), X)
    # Signal loaded in buff_in
    C.pfft_in * C.buff_in # buff_in in place fft of input X
    return C.buff_in
end

function _load_kernel!(C::CConv, kernel::AbstractVector{<:Real}, flip_kernel, phase)
    fill!(C.buff_ker, 0.0) # Prepare zero padding
    copy!(view(C.buff_ker, axes(kernel, 1)), kernel)
    if phase != 0 # Correct Phase
        circshift!(C.buff_ker, -phase)
    end
    # Kernel Loaded in buff_ker
    C.pfft_ker * C.buff_ker # buff_ker in place fft of kernel
    if flip_kernel
        conj!(C.buff_ker)
    end
    return C.buff_ker
end

function _conv!(C::CConv)
    # In place multiplication in buff_in
    C.buff_in .*= C.buff_ker
    C.pifft_in * C.buff_in# buff_in place ifft
    _crop_out!(C) # Crop in buff_out!
end

function _crop_out!(C::CConv)
    # Crop then correct the phase and not the other way
    Y_v = view(C.buff_in, C.crop_range[1]:C.crop_range[2])
    if eltype(C.buff_out) <: Real
        _copy_real!(C.buff_out, Y_v)
    else
        copyto!(C.buff_out, Y_v)
    end
    return C.buff_out
end

function (C::CConv)(
    X::AbstractArray{<:Real},
    kernel::AbstractVector{<:Union{Real,Complex}};
    flip_kernel = false,
    load_kernel = true,
    phase = default_phase_kernel(length(kernel)),
)
    _load_sig!(C, X)
    if load_kernel
        _load_kernel!(C, kernel, flip_kernel, phase)
    end
    _conv!(C)
    return copy(C.buff_out)
end

struct GMWFrame
    wave_dim::Integer # N
    params::AbstractVector{<:AbstractVector{<:Real}} # a,u,β,γ, Px4
    frame::AbstractVector{<:AbstractVector{<:Union{Real,Complex}}} # (P+1)xN
    freq_peaks::Vector{Float64} # P+1
    sigmas::Array{Float64} # P+1
    selfdual::Bool
    analytic::Bool
end

mutable struct ScaleParams
    const b::Real
    const g::Real
    const J::Int
    const Q::Int
    const wmin::Real
    const wmax::Real
    const wave_dim::Int
    const work_dim::Int
    const analytic::Bool
    const padding::Int
    frame::Ref{GMWFrame}
end

ScaleParams(
    b,
    g,
    J,
    Q,
    wmin,
    wmax,
    wave_dim;
    work_dim = wave_dim,
    analytic = false,
    padding = 0,
) = ScaleParams(
    b,
    g,
    J,
    Q,
    wmin,
    wmax,
    wave_dim,
    work_dim,
    analytic,
    padding,
    Ref{GMWFrame}(),
)

struct TimeParams
    kernel_dim::Int
    kernel_type::Symbol
    kernel_params::AbstractArray{<:Real}
    work_dim::Int
    dt::Int
    padding::Int
end

TimeParams(
    kernel_dim::Int,
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real};
    work_dim::Int = kernel_dim,
    dt::Int = 1,
    padding::Int = 0,
) = TimeParams(kernel_dim, kernel_type, kernel_params, work_dim, dt, padding)

struct DecompParams
    sp::ScaleParams
    tp::TimeParams
    work_dim::Int
end

ScaleParams(dp::DecompParams) = dp.sp
TimeParams(dp::DecompParams) = dp.tp

function DecompParams(;
    b::Real,
    g::Real,
    J::Int,
    Q::Int,
    wmin::Real,
    wmax::Real,
    wave_dim::Int,
    kernel_dim::Int,
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real},
    work_dim::Int,
    dt::Int = 1,
    analytic::Bool = false,
    padding::Union{Symbol,Int} = :next2pow,
)
    padding = padding == :next2pow ? next2pow_padding(work_dim, 0) : padding
    sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim; work_dim, analytic, padding)
    tp = TimeParams(kernel_dim, kernel_type, kernel_params; work_dim, dt, padding)
    return DecompParams(sp, tp, work_dim)
end

function wavelet_parameters(b, g, J, Q, wmin, wmax)
    return GMW.gmw_grid(b, g, J, Q, wmin, wmax, 0)
end
wavelet_parameters(; b, g, J, Q, wmin, wmax) = wavelet_parameters(b, g, J, Q, wmin, wmax)
wavelet_parameters(sp::ScaleParams) =
    wavelet_parameters(sp.b, sp.g, sp.J, sp.Q, sp.wmin, sp.wmax)
wavelet_parameters(dp::DecompParams) = wavelet_parameters(ScaleParams(dp.sp))

frequency_peaks(dp::DecompParams) = frequency_peaks(ScaleParams(dp))
frequency_peaks(sp::ScaleParams) = map(p -> GMW.peak_n(p, 1), wavelet_parameters(sp))

_gauss(t, p) = exp(-0.5 * (t / p) .^ 2)
_dtgauss(t, p) = -(t / p^2) * _gauss(t, p)
_dpgauss(t, p) = -(t^2 / p^3) * _gauss(t, p)
function gausskernel(kernel_dim, kernel_params)
    p = kernel_params[1]
    t = LinRange(-kernel_dim / 2, kernel_dim / 2, kernel_dim)
    g = _gauss.(t, p)
    g = g / sum(g)
    return g
end

function dt_gausskernel(kernel_dim, kernel_params)
    p = kernel_params[1]
    t = LinRange(-kernel_dim / 2, kernel_dim / 2, kernel_dim)
    g = _gauss.(t, p)
    dg = _dtgauss.(t, p)
    g = dg / sum(g)
end

function dp_gausskernel(kernel_dim, kernel_params)
    p = kernel_params[1]
    t = LinRange(-kernel_dim / 2, kernel_dim / 2, kernel_dim)
    g = _gauss.(t, p)
    dpg = _dpgauss.(t, p)
    s = sum(g)
    ds = sum(dpg)
    g = (s * dpg - ds * g) / s^2
end

function gauss_expo_kernel(kernel_dim, kernel_params)
    s, alpha, n = kernel_params
    sigmas = exp.([log(s) + i * log(alpha) for i = 0:(n-1)])
    return [gausskernel(kernel_dim, sigma) for sigma in sigmas]
end

function dt_gauss_expo_kernel(kernel_dim, kernel_params)
    s, alpha, n = kernel_params
    sigmas = exp.([log(s) + i * log(alpha) for i = 0:(n-1)])
    return [dt_gausskernel(kernel_dim, sigma) for sigma in sigmas]
end

function dp_gauss_expo_kernel(kernel_dim, kernel_params)
    s, alpha, n = kernel_params
    sigmas = exp.([log(s) + i * log(alpha) for i = 0:(n-1)])
    return [dp_gausskernel(kernel_dim, sigma) for sigma in sigmas]
end

function rectkernel(kernel_dim, kernel_params)
    n = kernel_dim
    t = LinRange(-kernel_dim / 2, kernel_dim / 2, kernel_dim)
    T = kernel_params[1] / 2
    g = zeros(Float64, n)
    g[abs.(t).<=T] .= 1
    g = g / sum(g)
    return g
end

averaging_kernel(tp::TimeParams) =
    averaging_kernel(tp.kernel_type, tp.kernel_params, tp.kernel_dim)
averaging_kernel(dp::DecompParams) =
    averaging_kernel(dp.tp.kernel_type, dp.tp.kernel_params, dp.tp.kernel_dim)

function averaging_kernel(
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real},
    kernel_dim::Integer,
)
    if kernel_type == :gaussian
        avg_kernel = gausskernel(kernel_dim, kernel_params)
    elseif kernel_type == :gaussian_exponential
        avg_kernel = gauss_expo_kernel(kernel_dim, kernel_params)
    elseif kernel_type == :rect
        avg_kernel = rectkernel(kernel_dim, kernel_params)
    else
        avg_kernel = []
        throw(ArgumentError("Kernel type $(kernel_type) not implemented"))
    end
    return avg_kernel
end

function dt_fft(x::AbstractArray{<:Real})
    N = length(x)
    if iseven(N)
        Nh = div(N, 2)
        k = collect(Complex(0, 2pi / N) * ((-Nh+1):Nh))
        k[end] = 0
        circshift!(k, -Nh + 1)
    else
        Nh = div(N, 2)
        k = collect(Complex(0, 2pi / N) * ((-Nh):Nh))
        circshift!(k, -Nh)
    end
    return real(ifft(fft(x) .* k))
end

dt_averaging_kernel(tp::TimeParams) =
    dt_averaging_kernel(tp.kernel_type, tp.kernel_params, tp.kernel_dim)
dt_averaging_kernel(dp::DecompParams) =
    dt_averaging_kernel(dp.tp.kernel_type, dp.tp.kernel_params, dp.tp.kernel_dim)

function dt_averaging_kernel(
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real},
    kernel_dim::Integer,
)
    if kernel_type == :gaussian
        avg_kernel = dt_gausskernel(kernel_dim, kernel_params)
    elseif kernel_type == :gaussian_exponential
        avg_kernel = dt_gauss_expo_kernel(kernel_dim, kernel_params)
    else
        avg_kernel = []
        throw(ArgumentError("dt Kernel type $(kernel_type) not implemented"))
    end
    return avg_kernel
end

dp_averaging_kernel(tp::TimeParams) =
    dp_averaging_kernel(tp.kernel_type, tp.kernel_params, tp.kernel_dim)
dp_averaging_kernel(dp::DecompParams) =
    dp_averaging_kernel(dp.tp.kernel_type, dp.tp.kernel_params, dp.tp.kernel_dim)

function dp_averaging_kernel(
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real},
    kernel_dim::Integer,
)
    if kernel_type == :gaussian
        avg_kernel = dp_gausskernel(kernel_dim, kernel_params)
    elseif kernel_type == :gaussian_exponential
        avg_kernel = dp_gauss_expo_kernel(kernel_dim, kernel_params)
    else
        avg_kernel = []
        throw(ArgumentError("dt Kernel type $(kernel_type) not implemented"))
    end
    return avg_kernel
end

function GMWFrame(sp::ScaleParams)
    (; wave_dim, analytic) = sp
    params = wavelet_parameters(sp)
    if isassigned(sp.frame)
        frame = sp.frame[]
    else
        frame = GMWFrame(wave_dim, params; analytic)
        sp.frame = Ref(frame)
    end
    return frame
end

GMWFrame(dp::DecompParams) = GMWFrame(ScaleParams(dp))

function GMWFrame(
    N::Integer,
    params::AbstractArray{<:AbstractVector{<:Real}};
    selfdual = true,
    analytic = false,
)
    L = div(N, 2) + 1 # Analytical fft size

    # Init frame 
    frame = [GMW.gmw(0, p[1], 0, p[3], p[4], N, :peak) for p in params]
    freq_peaks = map(p -> GMW.peak_n(p, 1), params)
    # Build low-pass 
    w = (0:(L-1)) / N
    w0 = minimum(freq_peaks)
    low_pass = exp.(-(3 * log(10) / 20) * (w / w0) .^ 2) # -3dB at w=w0
    frame = [frame..., low_pass]
    freq_peaks = vcat(freq_peaks, 0.0)

    # Self-Dual normalisation
    if selfdual
        ψ_norm = sqrt.(sum(x -> abs2.(x), frame))
        if analytic
            if iseven(N)
                ψ_norm[2:(L-1)] /= sqrt(2)
            else
                ψ_norm[2:L] /= sqrt(2)
            end
        end
        frame = [g ./ ψ_norm for g in frame]
    end

    # To time now
    # First some padding
    frame = [vcat(g, zeros(N - L)) for g in frame]
    if !analytic
        for g in frame
            g[1] /= 2
            g[L] /= iseven(N) ? 2 : 1
        end
        frame = [2 * real(ifft(g)) for g in frame]
    else
        frame = [ifft(g) for g in frame]
    end
    # time centering of the filters
    frame = [circshift(g, default_phase_kernel(N)) for g in frame]

    # time deviations
    # filters are symmetrical in time so we compute the time deviation only on the right side 
    t = LinRange(-1, 1, N) * div(N, 2)
    sigmas = [sqrt(sum(abs2, t .* g) / sum(abs2, g)) for g in frame]
    return GMWFrame(N, params, frame, freq_peaks, sigmas, selfdual, analytic)
end


function cross_scalogram(
    L::AbstractDict{Symbol,<:AbstractArray{<:Real}},
    C::AbstractDict{Tuple{Symbol,Symbol},Symbol},
    dp::DecompParams,
    frame::Union{AbstractArray{<:AbstractArray{T}},GMWFrame},
    avg_kernel::Union{AbstractVector{<:Real},AbstractVector{<:AbstractVector{<:Real}}},
) where {T<:Union{Real,Complex}}

    (; work_dim, tp, sp) = dp
    (; dt) = tp
    (; padding, analytic) = sp
    allequal(length, values(L)) || throw(error("Wrong size"))
    frame = frame isa GMWFrame ? frame.frame : frame
    work_dim == length(first(values(L))) ||
        throw(error("work_dim miss specified in DecompParams struct"))
    time_sampling = 1:dt:work_dim
    if avg_kernel isa Vector{Float64}
        many_kernels = false
    else
        many_kernels = true
        length(frame) == length(avg_kernel) ||
            throw(error("Number of filters and averaging kernels are not equal"))
    end
    return_type = analytic ? ComplexF64 : Float64

    WaveC = CConv(return_type, work_dim; padding)
    KernelC = CConv(return_type, work_dim; padding)

    out = Dict(c => Dict() for c in keys(C))
    for (i, gmw) in enumerate(frame)
        mem = Dict()
        avg_kernel = many_kernels ? avg_kernel[i] : avg_kernel
        load_kernel = true
        for c in keys(C)
            (n, m) = c
            x = L[n]
            y = L[m]
            x_i = n in keys(mem) ? mem[n] : mem[n] = WaveC(x, gmw; load_kernel)
            y_i = m in keys(mem) ? mem[m] : mem[m] = WaveC(y, gmw; load_kernel)
            f = KernelC(x_i .* y_i, avg_kernel; load_kernel)
            f = f[time_sampling]
            out[c][i] = f
            if load_kernel
                load_kernel = false
            end
        end
    end
    # Output matrices
    out_hcat = Dict()
    for c in keys(C)
        cs = [out[c][i] for i = 1:length(frame)]
        out_hcat[C[c]] = stack(cs)
    end
    return out_hcat
end

cross_scalogram(
    L::AbstractDict{Symbol,<:AbstractArray{<:Real}},
    C::AbstractDict{Tuple{Symbol,Symbol},Symbol},
    dp::DecompParams,
) = cross_scalogram(L, C, dp, GMWFrame(dp), averaging_kernel(dp))

_xy_dict(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}) =
    (Dict(:x => x, :y => y), Dict((:x, :y) => :xy))

cross_scalogram(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, args...) =
    cross_scalogram(_xy_dict(x, y)..., args...)[:xy]

dt_cross_scalogram(
    L::AbstractDict{Symbol,<:AbstractArray{<:Real}},
    C::AbstractDict{Tuple{Symbol,Symbol},Symbol},
    dp,
) = cross_scalogram(L, C, dp, GMWFrame(dp), dt_averaging_kernel(dp))
dt_cross_scalogram(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, args...) =
    dt_cross_scalogram(_xy_dict(x, y)..., args...)[:xy]

dp_cross_scalogram(
    L::AbstractDict{Symbol,<:AbstractArray{<:Real}},
    C::AbstractDict{Tuple{Symbol,Symbol},Symbol},
    dp,
) = cross_scalogram(L, C, dp, GMWFrame(dp), dp_averaging_kernel(dp))
dp_cross_scalogram(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, args...) =
    dp_cross_scalogram(_xy_dict(x, y)..., args...)[:xy]


average(x::AbstractArray{<:Real}, tp::TimeParams, subsampling::Bool = true) =
    average([x], tp, subsampling)[1]
function average(
    L::AbstractArray{<:AbstractArray{<:Real}},
    tp::TimeParams,
    subsampling::Bool = true,
)
    (; dt, padding) = tp
    allequal(length, L) || throw(error("Wrong size"))
    work_dim = length(L[1])
    time_sampling = 1:dt:work_dim
    KernelC = CConv(Float64, work_dim; padding)
    avg_kernel = averaging_kernel(tp)
    out = []
    load_kernel = true
    for x in L
        x_avg = KernelC(x, avg_kernel; load_kernel)
        if subsampling
            x_avg = x_avg[time_sampling]
        end
        push!(out, x_avg)
        if load_kernel
            load_kernel = false
        end
    end
    return out
end

dt_average(x::AbstractArray{<:Real}, tp::TimeParams, subsampling::Bool = true) =
    dt_average([x], tp, subsampling)[1]
function dt_average(
    L::AbstractArray{<:AbstractArray{<:Real}},
    tp::TimeParams,
    subsampling::Bool = true,
)
    (; dt, padding) = tp
    allequal(length, L) || throw(error("Wrong size"))
    work_dim = length(L[1])
    time_sampling = 1:dt:work_dim
    KernelC = CConv(Float64, work_dim; padding)
    avg_kernel = dt_averaging_kernel(tp)
    out = []
    load_kernel = true
    for x in L
        x_avg = KernelC(x, avg_kernel; load_kernel)
        if subsampling
            x_avg = x_avg[time_sampling]
        end
        push!(out, x_avg)
        if load_kernel
            load_kernel = false
        end
    end
    return out
end

dp_average(x::AbstractArray{<:Real}, tp::TimeParams, subsampling::Bool = true) =
    dp_average([x], tp, subsampling)[1]
function dp_average(
    L::AbstractArray{<:AbstractArray{<:Real}},
    tp::TimeParams,
    subsampling::Bool = true,
)
    (; dt, padding) = tp
    allequal(length, L) || throw(error("Wrong size"))
    work_dim = length(L[1])
    time_sampling = 1:dt:work_dim
    KernelC = CConv(Float64, work_dim; padding)
    avg_kernel = dp_averaging_kernel(tp)
    out = []
    load_kernel = true
    for x in L
        x_avg = KernelC(x, avg_kernel; load_kernel)
        if subsampling
            x_avg = x_avg[time_sampling]
        end
        push!(out, x_avg)
        if load_kernel
            load_kernel = false
        end
    end
    return out
end

# function scalogram_error_mask(work_dim::Int, gmw::GMWFrame; factor = 3, max_sigma = false)
#     sigmas = gmw.sigmas
#     mask = falses(work_dim, length(sigmas))
#     max_sigma_val = ceil(Int, maximum(sigmas) * factor[1])
#     for (i, s) in enumerate(sigmas)
#         if max_sigma
#             s = max_sigma_val
#         else
#             s = ceil(Int, s * factor[1])
#         end
#         mask[1:s, i] .= true
#         mask[(end-s+1):end, i] .= true
#     end
#     return mask
# end

function error_mask(dp::DecompParams, work_dim::Int)
    (; tp, sp) = dp
    mask = vcat(1, falses(work_dim - 2), 1)
    frame = GMWFrame(sp)
    avg_kernel = averaging_kernel(tp)
    out = cross_scalogram(mask, mask, dp, frame, avg_kernel)
    mask = abs.(out) .> 1e-6
    return mask
end

function error_mask(dp::DecompParams, mask::AbstractArray{Bool})
    (; tp, sp) = dp
    mask = copy(mask)
    mask[1] = 1
    mask[end] = 1
    frame = GMWFrame(sp)
    avg_kernel = averaging_kernel(tp)
    out = cross_scalogram(mask, mask, dp, frame, avg_kernel)
    mask = abs.(out) .> 1e-6
    return mask
end

error_mask(tp::TimeParams) = error_mask(tp, tp.work_dim)
function error_mask(tp::TimeParams, work_dim::Int)
    mask = vcat(1, falses(work_dim - 2), 1)
    mask = average(mask, tp) .> 1e-6
    return mask
end
function error_mask(tp::TimeParams, mask::AbstractArray{Bool})
    mask = copy(mask)
    mask[1] = 1
    mask[end] = 1
    mask = average(mask, tp) .> 1e-6
    return mask
end
