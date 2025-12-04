function next2pow_padding(N, init_padding)
    M = N + init_padding
    return nextpow(2, M) - M
end

default_phase_kernel(kernel_dim) =
    iseven(kernel_dim) ? div(kernel_dim, 2) - 1 : div(kernel_dim, 2)
_copy_real!(x::AbstractArray{Float64}, y::AbstractArray{ComplexF64}) = map!(real, x, y)

mutable struct CConv{T <: Union{Float64, ComplexF64}}
    const input_dim::Int64
    const output_dim::Int64
    const padding::Int64
    const crop_range::Tuple{Int64, Int64}
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
    ) where {T <: Union{Float64, ComplexF64}}
    conv_dim = (input_dim + padding)
    # Setting up internals in keywords
    buff_in = zeros(ComplexF64, conv_dim)
    pfft_in = plan_fft!(buff_in)
    buff_ker = zeros(ComplexF64, conv_dim)
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
    C.pifft_in * C.buff_in # buff_in place ifft
    return _crop_out!(C) # Crop in buff_out!
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
        kernel::AbstractVector{<:Union{Real, Complex}};
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
    frame::AbstractVector{<:AbstractVector{<:Union{Real, Complex}}} # (P+1)xN
    freq_peaks::Vector{Float64} # P+1
    sigmas::Array{Float64} # P+1
    selfdual::Bool
    analytic::Bool
end

"""
    ScaleParams(b, g, J, Q, wmin, wmax, wave_dim, analytic, padding, frame)

A struct to hold parameters for time-frequency scale decomposition.

# Fields
- `b::Real`: First parameter of the Generalized Morse Wavelet (e.g. `1`)
- `g::Real`: Second parameter of the Generalized Morse Wavelet (e.g. `3`)
- `J::Int`: Number of octaves (e.g. `floor(Int(log2(wave_dim)))`)
- `Q::Int`: Number of inter-octaves (e.g. `3`)
- `wmin::Real`: Minimum frequency pulsation (e.g. `2*pi/wave_dim`)
- `wmax::Real`: Maximum frequency pulsation (e.g. `pi`)
- `wave_dim::Int`: Size of the wavelet transform. 
- `analytic::Bool`: If `true`, uses analytic wavelets.
- `padding::Int`: Zero-padding length for the transform.
- `frame::Ref{GMWFrame}`: Reference to the `GMWFrame` object used for the decomposition.
"""
mutable struct ScaleParams
    const b::Real
    const g::Real
    const J::Int
    const Q::Int
    const wmin::Real
    const wmax::Real
    const wave_dim::Int
    const analytic::Bool
    const padding::Int
    frame::Union{Nothing, GMWFrame}
end

ScaleParams(
    b,
    g,
    J,
    Q,
    wmin,
    wmax,
    wave_dim;
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
    analytic,
    padding,
    nothing
)


"""
    TimeParams(kernel_dim, kernel_type, kernel_params, dt, padding)

A struct to hold parameters for time-domain averaging and sampling.

# Fields
- `kernel_dim::Int`: Size of the averaging kernel. 
- `kernel_type::Symbol`: Type of kernel (e.g., `:gaussian`, `:rect`).
- `kernel_params::AbstractArray{<:Real}`: Parameters defining the kernel (e.g., standard deviation for Gaussian).
- `dt::Int`: Sampling time step.
- `padding::Int`: Zero-padding length for the time-domain averaging.
"""
struct TimeParams
    kernel_dim::Int
    kernel_type::Symbol
    kernel_params::AbstractArray{<:Real}
    dt::Int
    padding::Int
end

TimeParams(
    kernel_dim::Int,
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real};
    dt::Int = 1,
    padding::Int = 0,
) = TimeParams(kernel_dim, kernel_type, kernel_params, dt, padding)


"""
    DecompParams(sp, tp)

A struct to hold parameters for time-frequency decomposition and time averaging.

# Fields
- `sp::ScaleParams`: Parameters for scale (time-frequency) decomposition.
- `tp::TimeParams`: Parameters for time-domain averaging.
"""
struct DecompParams
    sp::ScaleParams
    tp::TimeParams
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
        dt::Int = 1,
        analytic::Bool = false,
        padding::Int = 0,
    )
    sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim; analytic, padding)
    tp = TimeParams(kernel_dim, kernel_type, kernel_params; dt, padding)
    return DecompParams(sp, tp)
end

function wavelet_parameters(b, g, J, Q, wmin, wmax)
    return GMW.gmw_grid(b, g, J, Q, wmin, wmax, 0)
end
wavelet_parameters(; b, g, J, Q, wmin, wmax) = wavelet_parameters(b, g, J, Q, wmin, wmax)
wavelet_parameters(sp::ScaleParams) =
    wavelet_parameters(sp.b, sp.g, sp.J, sp.Q, sp.wmin, sp.wmax)
wavelet_parameters(dp::DecompParams) = wavelet_parameters(ScaleParams(dp.sp))

frequency_peaks(dp::DecompParams) = frequency_peaks(ScaleParams(dp))
frequency_peaks(sp::ScaleParams) =
    vcat(map(p -> GMW.peak_n(p, 1), wavelet_parameters(sp)), 0.0)

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
    return g = dg / sum(g)
end

function dp_gausskernel(kernel_dim, kernel_params)
    p = kernel_params[1]
    t = LinRange(-kernel_dim / 2, kernel_dim / 2, kernel_dim)
    g = _gauss.(t, p)
    dpg = _dpgauss.(t, p)
    s = sum(g)
    ds = sum(dpg)
    return g = (s * dpg - ds * g) / s^2
end

function gauss_expo_kernel(kernel_dim, kernel_params)
    s, alpha, n = kernel_params
    sigmas = exp.([log(s) + i * log(alpha) for i in 0:(n - 1)])
    return [gausskernel(kernel_dim, sigma) for sigma in sigmas]
end

function dt_gauss_expo_kernel(kernel_dim, kernel_params)
    s, alpha, n = kernel_params
    sigmas = exp.([log(s) + i * log(alpha) for i in 0:(n - 1)])
    return [dt_gausskernel(kernel_dim, sigma) for sigma in sigmas]
end

function dp_gauss_expo_kernel(kernel_dim, kernel_params)
    s, alpha, n = kernel_params
    sigmas = exp.([log(s) + i * log(alpha) for i in 0:(n - 1)])
    return [dp_gausskernel(kernel_dim, sigma) for sigma in sigmas]
end

function rectkernel(kernel_dim, kernel_params)
    n = kernel_dim
    t = LinRange(-kernel_dim / 2, kernel_dim / 2, kernel_dim)
    T = kernel_params[1] / 2
    g = zeros(Float64, n)
    g[abs.(t) .<= T] .= 1
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
        k = collect(Complex(0, 2pi / N) * ((-Nh + 1):Nh))
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
    if isnothing(sp.frame)
        setframe!(sp)
    end
    return sp.frame
end

setframe!(sp::ScaleParams) = begin
    (; wave_dim, analytic) = sp
    params = wavelet_parameters(sp)
    frame = GMWFrame(wave_dim, params; analytic)
    sp.frame = frame
    return sp
end

freeframe!(dp::DecompParams) = freeframe!(dp.sp)
freeframe!(sp::ScaleParams) = sp.frame = nothing

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
    w = (0:(L - 1)) / N
    w0 = minimum(freq_peaks)
    low_pass = exp.(-(3 * log(10) / 20) * (w / w0) .^ 2) # -3dB at w=w0
    frame = [frame..., low_pass]
    freq_peaks = vcat(freq_peaks, 0.0)

    # Self-Dual normalisation
    if selfdual
        ψ_norm = sqrt.(sum(x -> abs2.(x), frame))
        if analytic
            if iseven(N)
                ψ_norm[2:(L - 1)] /= sqrt(2)
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

function make_mem(C)
    ckf = collect(Iterators.flatten(keys(C)))
    ccount = map(k -> (k, count(==(k), ckf)), unique(ckf))
    filter!(x -> x[2] > 1, ccount)
    return mem = Dict{Symbol, Any}((k => nothing for (k, v) in ccount))
end

in_mem(mem, k) = k in keys(mem) && !isnothing(mem[k])
store!(mem, k, v) = k in keys(mem) && isnothing(mem[k]) ? mem[k] = v : nothing

function cross_scalogram(
        L::AbstractDict{Symbol, <:AbstractArray{<:Real}},
        C::AbstractDict{Tuple{Symbol, Symbol}, Symbol},
        dp::DecompParams,
        frame::Union{AbstractArray{<:AbstractArray{T}}, GMWFrame},
        avg_kernel::Union{AbstractVector{<:Real}, AbstractVector{<:AbstractVector{<:Real}}};
        K = 2,
        subsampling = true
    ) where {T <: Union{Real, Complex}}

    (; tp, sp) = dp
    (; dt) = tp
    (; analytic) = sp
    pad_tp = tp.padding
    pad_sp = sp.padding
    allequal(length, values(L)) || throw(error("Wrong size"))
    frame = frame isa GMWFrame ? frame.frame : frame
    work_dim = length(first(values(L)))
    time_sampling = 1:dt:work_dim
    if avg_kernel isa Vector{Float64}
        many_kernels = false
    else
        many_kernels = true
        length(frame) == length(avg_kernel) ||
            throw(error("Number of filters and averaging kernels are not equal"))
    end
    return_type = analytic ? ComplexF64 : Float64

    WaveC = CConv(return_type, work_dim; padding = pad_sp)
    KernelC = CConv(return_type, work_dim; padding = pad_tp)

    out = Dict(c => Dict() for c in keys(C))
    for C_p in Iterators.partition(C, K)
        for (i, gmw) in enumerate(frame)
            mem = make_mem(C_p)
            avgk = many_kernels ? avg_kernel[i] : avg_kernel
            load_kernel = true
            for (c, _) in C_p
                (n, m) = c
                x = L[n]
                y = L[m]
                x_i = in_mem(mem, n) ? mem[n] : WaveC(x, gmw; load_kernel)
                store!(mem, n, x_i)
                y_i = in_mem(mem, m) ? mem[m] : WaveC(y, gmw; load_kernel)
                store!(mem, m, y_i)
                f = KernelC(x_i .* y_i, avgk; load_kernel)
                if subsampling
                    f = f[time_sampling]
                end
                out[c][i] = f
                load_kernel = false
            end
        end
    end
    # Output matrices
    out_hcat = Dict()
    for c in keys(C)
        cs = [out[c][i] for i in 1:length(frame)]
        out_hcat[C[c]] = stack(cs)
    end
    return out_hcat
end

cross_scalogram(
    L::AbstractDict{Symbol, <:AbstractArray{<:Real}},
    C::AbstractDict{Tuple{Symbol, Symbol}, Symbol},
    dp::DecompParams;
    kwargs...
) = cross_scalogram(L, C, dp, GMWFrame(dp), averaging_kernel(dp); kwargs...)

_xy_dict(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}) =
    (Dict(:x => x, :y => y), Dict((:x, :y) => :xy))

cross_scalogram(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, args...; kwargs...) =
    cross_scalogram(_xy_dict(x, y)..., args...; kwargs...)[:xy]

dt_cross_scalogram(
    L::AbstractDict{Symbol, <:AbstractArray{<:Real}},
    C::AbstractDict{Tuple{Symbol, Symbol}, Symbol},
    dp; kwargs ...
) = cross_scalogram(L, C, dp, GMWFrame(dp), dt_averaging_kernel(dp); kwargs...)
dt_cross_scalogram(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, args...; kwargs...) =
    dt_cross_scalogram(_xy_dict(x, y)..., args...; kwargs...)[:xy]

dp_cross_scalogram(
    L::AbstractDict{Symbol, <:AbstractArray{<:Real}},
    C::AbstractDict{Tuple{Symbol, Symbol}, Symbol},
    dp; kwargs...
) = cross_scalogram(L, C, dp, GMWFrame(dp), dp_averaging_kernel(dp); kwargs...)
dp_cross_scalogram(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, args...; kwargs...) =
    dp_cross_scalogram(_xy_dict(x, y)..., args...; kwargs...)[:xy]


function cross_correlation_rey(
        L::AbstractDict{Symbol, <:AbstractArray{<:Real}},
        C::AbstractDict{Tuple{Symbol, Symbol}, Symbol},
        tp::TimeParams; subsampling = true
    )
    (; dt, padding) = tp
    allequal(length, L) || throw(error("Wrong size"))
    work_dim = length(first(values(L)))
    time_sampling = 1:dt:work_dim
    KernelC = CConv(Float64, work_dim; padding)
    avg_kernel = averaging_kernel(tp)
    out = Dict()
    load_kernel = true
    mem = make_mem(C)
    for c in keys(C)
        (n, m) = c
        x = L[n]
        y = L[m]
        x_p = in_mem(mem, n) ? mem[n] : x - KernelC(x, avg_kernel; load_kernel)
        store!(mem, n, x_p)
        load_kernel = false
        y_p = in_mem(mem, m) ? mem[m] : y - KernelC(y, avg_kernel; load_kernel)
        store!(mem, m, y_p)
        f = KernelC(x_p .* y_p, avg_kernel; load_kernel)
        if subsampling
            f = f[time_sampling]
        end
        out[C[c]] = f
    end
    return out
end


average(x, tp; subsampling) = average(x, tp, subsampling)
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

function converror_mask(dp::DecompParams, work_dim::Int; subsampling = true)
    (; tp, sp) = dp
    # Fake error on border
    x = vcat(true, falses(work_dim - 2), true)
    frame = GMWFrame(sp)
    avg_kernel = averaging_kernel(tp)
    out = cross_scalogram(x, x, dp, frame, avg_kernel; subsampling)
    out ./= out[1, :]'
    return out .> 0.01
end

function converror_mask(tp::TimeParams, work_dim::Int; subsampling = true)
    # Fake error on border
    mask = vcat(true, falses(work_dim - 2), true)
    mask = average(mask, tp; subsampling)
    mask ./= mask[1]
    return mask .> 0.01
end


function error_mask(tp::TimeParams, mask::AbstractArray{Bool}; subsampling = true)
    N = length(mask)
    mask = average(mask, tp; subsampling)
    mask = mask .> 0.01
    convmask = converror_mask(tp, N)
    mask = mask .|| convmask
    return mask
end
