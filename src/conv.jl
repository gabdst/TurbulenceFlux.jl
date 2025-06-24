using Random, StatsFuns, FFTW
import GeneralizedMorseWavelets as GMW

struct IIR
    a::Vector{Float64}
    b::Vector{Float64}
    y::Vector{Float64}
    function IIR(a, b)
        y = zeros(Float64, length(a))
        return new(a, b, y)
    end
end

function (iir::IIR)(x)
    y_new = iir.a' * (iir.y) + iir.b' * x
    circshift!(iir.y, 1)
    iir.y[1] = y_new
    return y_new
end

function reset!(iir::IIR)
    fill!(iir.y, 0)
    return nothing
end

function applyfilter(iir::IIR, x::Vector{Float64})
    p = length(iir.b)
    n = length(x)
    out = zeros(n)
    for i = 1:n
        v_x = view(x, max(1, i - p + 1):i)
        if length(v_x) < p
            v_x = vcat(zeros(p - length(v_x)), v_x)
        end
        out[i] = iir(v_x)
    end
    return out
end

function lowpass(fc, fs)
    alpha = 2pi * (fc / fs) / (2pi * fc / fs + 1)
    a = [1 - alpha]
    b = [alpha]
    return IIR(a, b)
end

_default_phase_kernel(kernel_dim) =
    iseven(kernel_dim) ? div(kernel_dim, 2) - 1 : div(kernel_dim, 2)
_copy_real!(x::AbstractArray{Float64}, y::AbstractArray{ComplexF64}) = map!(real, x, y)

struct CConv{T<:Union{Float64,ComplexF64}} # 1D Convolution with constrained kernel
    input_dim::Tuple{Integer,Integer}
    output_dim::Tuple{Integer,Integer,Integer}
    kernel_dim::Tuple{Integer,Integer}
    padding::Integer
    kernel_init_params::Function # (Optional) Function to initialize parameters
    kernel_func::Function # (Optional) Function that generates the kernel given the parameters
    crop_phase_output!::Function # Function that prepares the output in buff_out

    # Internals
    ## In-Place FFT plans with associated buffers
    buff_in::Array{ComplexF64}
    pfft_in::Any
    buff_work::Array{ComplexF64}
    pfft_work::Any
    pifft_work::Any
    buff_out::Array{T}
end

"""
1D convolutional kernel

It the input is NxM and the kernel function is NxP then the output is NxPxM. Each columns of the kernel is a different kernel applied on each column of the input resulting in PxM 1D convolutions.
"""
function CConv(
    ::Type{T},
    input_dim::Tuple{Integer,Integer},
    kernel_dim::Tuple{Integer,Integer};
    phase_output = nothing,
    padding = 0,
    crop_range = (1:input_dim[1]),
    conv_dim = (input_dim[1] + padding, 1, input_dim[2]),
    # Setting up internals in keywords 
    buff_in = Array{ComplexF64}(undef, conv_dim[1], 1, conv_dim[3]),
    pfft_in = plan_fft!(buff_in, 1),
    buff_work = Array{ComplexF64}(undef, conv_dim[1], kernel_dim[2], input_dim[2]),
    pfft_work = plan_fft!(buff_work, 1),
    pifft_work = plan_ifft!(buff_work, 1),
) where {T<:Union{Float64,ComplexF64}}
    if isnothing(phase_output) # Assuming centered kernel, the default phase is div(kernel_dim,2)
        default_phase = -_default_phase_kernel(kernel_dim[1]) # compensate the default phase introduced in default kernel func
        phase_output = [default_phase for _ = 1:input_dim[2]]
    end

    output_dim = (length(crop_range), kernel_dim[2], input_dim[2])
    buff_out = Array{T}(undef, output_dim...)
    crop_phase_output! = _make_crop_func(crop_range, phase_output, conv_dim, buff_out)

    return CConv{T}(
        input_dim,
        output_dim,
        kernel_dim,
        padding,
        crop_phase_output!,
        buff_in,
        pfft_in,
        buff_work,
        pfft_work,
        pifft_work,
        buff_out,
    )
end

function _load_sig!(C::CConv, X::AbstractArray{<:Real})
    X = reshape(X, promote_shape(C.input_dim, size(X)))
    fill!(C.buff_in, 0.0) # Prepare zero padding
    copy!(view(C.buff_in, axes(X, 1), 1, axes(X, 2)), X) # Copy signal on first and third axis
    return C.buff_in
end

function _conv!(C::CConv, X::AbstractArray, kernel::AbstractArray, flip_kernel::Bool)
    C.buff_in === X || copy!(C.buff_in, X) # input is buff_in, nothing to copy
    C.pfft_in * C.buff_in # buff_in in place fft of input X
    # kernel has been copied to buff_work
    if C.buff_work !== kernel
        C.buff_work[:, :, 1] .= kernel
        for k in axes(C.buff_work, 3)[2:end]
            C.buff_work[:, :, k] .= view(C.buff_work, :, :, 1)
        end
    end
    C.pfft_work * C.buff_work # buff_work in place fft of kernel
    if flip_kernel
        conj!(C.buff_work)
    end
    C.buff_work .*= C.buff_in
    C.pifft_work * C.buff_work # buff_work in place ifft
    C.crop_phase_output!(C.buff_work) # update buff_out
end

function (C::CConv)(X::AbstractArray{<:Real}, kernel, flip_kernel = false)
    X = _load_sig!(C, X)
    _conv!(C, X, kernel, flip_kernel)
    return copy(C.buff_out)
end

# Croping and phasing function operating on the output buffer buff_out
function _make_crop_func(crop_range, phase_output, output_dim, buff_out::AbstractArray)
    function crop_phase_func!(Y)
        for j = 1:output_dim[2], k = 1:output_dim[3]
            Y_v = view(Y, :, j, k)
            circshift!(Y_v, phase_output[k])
            Y_v = view(Y_v, crop_range)
            buff_out_v = view(buff_out, :, j, k)
            if eltype(buff_out_v) <: Real
                _copy_real!(buff_out_v, Y_v)
            else
                copyto!(buff_out_v, Y_v)
            end
        end
        return buff_out
    end
    return crop_phase_func!
end

struct ScaleParams
    b::Integer
    g::Integer
    J::Integer
    Q::Integer
    wmin::Float64
    wmax::Float64
    wave_dim::Integer
end

struct TimeParams
    kernel_dim::Integer
    kernel_type::Symbol
    kernel_params::AbstractArray{<:Real}
    dt::Integer
end

function wavelet_parameters(rng = nothing; b = 1, g = 3, J = 32, Q = 2, wmin = 0, wmax = pi)
    params = GMW.gmw_grid(b, g, J, Q, wmin, wmax, 0)
    return params
end
wavelet_parameters(b, g, J, Q, wmin, wmax) = wavelet_parameters(; b, g, J, Q, wmin, wmax)

function gausskernel(kernel_dim, params, padding)
    s = params[1]
    u = length(params) == 2 ? params[2] : 0
    t = LinRange(0, 1, kernel_dim)
    g = exp.(-0.5 * ((t .- 0.5 .+ u) / s) .^ 2)
    g = vcat(g, zeros(padding))
    g = g / sum(g)
    return g
end

function gauss_expo_kernel(kernel_dim, kernel_params)
    sigma0, alpha, n = kernel_params
    sigmas = exp.([log(sigma0) + i * log(alpha) for i = 0:(n-1)])
    return [gausskernel(kernel_dim, sigma, 0) for sigma in sigmas]
end

function rectkernel(kernel_dim, kernel_params)
    n = kernel_dim
    m = kernel_params[1]
    x = zeros(Float64, n)
    x[1:m] .= 1
    l = n - m
    l = iseven(l) ? div(l, 2) : div(l, 2) + 1
    circshift(x, l)
end

"""
    averaging_kernel(work_dim;kernel_type,kernel_params,dt,kernel_dim,with_sigma=false)

TODO
Initiliaze an averaging convolutional kernel.

It returns the range for sampling in time, the original parameter mapped in (-∞,+∞) and the kernel.

# Arguments
  - `input_dim::Union{Integer,Tuple{Integer,Integer}}`: Input dimension
  - `kernel_dim::Integer`: Maximum time support for the averaging kernel
  - `kernel_type::Symbol`: Type of kernel used for the averaging kernel
  - `kernel_params::Vector`: Vector of parameters for the averaging kernel
"""
averaging_kernel(tp::TimeParams) =
    averaging_kernel(tp.kernel_type, tp.kernel_params, tp.kernel_dim)
function averaging_kernel(
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real},
    kernel_dim::Integer,
)
    if kernel_type == :gaussian
        avg_kernel = gausskernel(kernel_dim, kernel_params, 0)
    elseif kernel_type == :gaussian_exponential
        avg_kernel = []
    elseif kernel_type == :rect
        avg_kernel = rectkernel(kernel_dim, kernel_params)
    else
        avg_kernel = []
        throw(error("Kernel type $(kernel_type) not implemented"))
    end
    return avg_kernel
end

struct GMWFrame
    wave_dim::Integer # N
    gmw_params::AbstractArray{<:AbstractVector{<:Real}} # a,u,b,g, Px4
    gmw_frame::AbstractArray{Vector{ComplexF64}} # (P+1)xN
    freq_peaks::AbstractArray{Float64} # P+1
    sigma_waves::Array{Float64} # P+1
    selfdual::Bool
    analytic::Bool
end

function GMWFrame(scale_params::ScaleParams)
    (; b, g, J, Q, wmin, wmax, wave_dim) = scale_params
    gmw_params = TurbulenceFlux.wavelet_parameters(b, g, J, Q, wmin, wmax)
    return GMWFrame(wave_dim, gmw_params)
end

function GMWFrame(
    N::Integer,
    gmw_params::AbstractArray{<:AbstractVector{<:Real}},
    selfdual = true,
    analytic = false,
)
    L = div(N, 2) + 1 # Analytical fft size

    # Init frame 
    gmw_frame = [GMW.gmw(0, p[1], 0, p[3], p[4], N, :peak) for p in gmw_params]
    freq_peaks = map(p -> GMW.peak_n(p, 1), gmw_params)
    # Build low-pass 
    w = (0:(L-1)) / N
    w0 = minimum(freq_peaks)
    low_pass = exp.(-(3 * log(10) / 20) * (w / w0) .^ 2) # -3dB at w=w0
    gmw_frame = [gmw_frame..., low_pass]
    freq_peaks = vcat(freq_peaks, 0.0)

    # Self-Dual normalisation
    if selfdual
        ψ_norm = sqrt.(sum(x -> abs2.(x), gmw_frame))
        if analytic
            if iseven(N)
                ψ_norm[2:(L-1)] /= sqrt(2)
            else
                ψ_norm[2:L] /= sqrt(2)
            end
        end
        gmw_frame = [g ./ ψ_norm for g in gmw_frame]
    end

    # To time now
    # First some padding
    gmw_frame = [vcat(g, zeros(N - L)) for g in gmw_frame]
    if !analytic
        for g in gmw_frame
            g[1] /= 2
            g[L] /= iseven(N) ? 2 : 1
        end
        gmw_frame = [2 * real(ifft(g)) for g in gmw_frame]
    else
        gmw_frame = [ifft(g) for g in gmw_frame]
    end
    # time centering of the filters
    gmw_frame = [circshift(g, _default_phase_kernel(N)) for g in gmw_frame]

    # time deviations
    # filters are symmetrical in time so we compute the time deviation only on the right side 
    phase = _default_phase_kernel(N)
    r = range(start = phase + 1, length = (N - phase - 1))
    t = (0:(length(r)-1))
    sigma_waves = [sqrt(sum(abs2, t .* g[r]) / sum(abs2, g[r])) for g in gmw_frame]
    return GMWFrame(N, gmw_params, gmw_frame, freq_peaks, sigma_waves, selfdual, analytic)
end

cross_scalogram(
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    deltat::Int64,
    gmw_frame::GMWFrame,
    averaging_kernel::Union{Vector{Float64},Vector{<:AbstractArray{Float64}}},
    analytic = false,
) = cross_scalogram([x, y], [(1, 2)], deltat, gmw_frame, averaging_kernel, analytic)

function cross_scalogram(
    L::Vector{<:AbstractArray{<:Real}},
    C::Vector{Tuple{Int64,Int64}},
    deltat::Int64,
    gmw_frame::GMWFrame,
    averaging_kernel::Union{Vector{Float64},Vector{<:AbstractArray{Float64}}},
    analytic = false,
)

    allequal(length, L) || throw(error("Wrong size"))
    gmw_frame = gmw_frame.gmw_frame
    work_dim = length(L[1])
    time_sampling = 1:deltat:work_dim
    if averaging_kernel isa Vector{Float64}
        many_kernels = false
    else
        many_kernels = true
        length(gmw_frame) == length(averaging_kernel) ||
            throw(error("Number of filters and averaging kernels are not equal"))
    end
    input_dim = (work_dim, 1)
    wave_dim = (length(gmw_frame[1]), 1)
    kernel_dim =
        many_kernels ? (length(averaging_kernel[1]), 1) : (length(averaging_kernel), 1)
    return_type = analytic ? ComplexF64 : Float64

    WaveC = CConv(return_type, input_dim, wave_dim)
    KernelC = CConv(return_type, input_dim, kernel_dim)

    out = Dict(c => Dict() for c in C)
    for (i, gmw) in enumerate(gmw_frame)
        mem = Dict()
        avg_kernel = many_kernels ? averaging_kernel[i] : averaging_kernel
        for c in C
            (n, m) = c
            x = L[n]
            y = L[m]
            x_i = n in keys(mem) ? mem[n] : mem[n] = WaveC(x, gmw)
            y_i = m in keys(mem) ? mem[m] : mem[m] = WaveC(y, gmw)
            x_i = vec(x_i)
            y_i = vec(y_i)
            f = KernelC(x_i .* y_i, avg_kernel)
            f = f[time_sampling]
            out[c][i] = f
        end
    end
    return out
end
