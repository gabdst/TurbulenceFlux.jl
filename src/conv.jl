_default_phase_kernel(kernel_dim) =
    iseven(kernel_dim) ? div(kernel_dim, 2) - 1 : div(kernel_dim, 2)
_copy_real!(x::AbstractArray{Float64}, y::AbstractArray{ComplexF64}) = map!(real, x, y)

struct CConv{T<:Union{Float64,ComplexF64}}
    input_dim::Int64
    output_dim::Int64
    kernel_dim::Int64
    padding::Int64
    crop_range::Tuple{Int64,Int64}
    # Internals
    phase::Int64
    buff_in::Vector{ComplexF64}
    pfft_in::AbstractFFTs.Plan{ComplexF64}
    buff_ker::Vector{ComplexF64}
    pfft_ker::AbstractFFTs.Plan{ComplexF64}
    pifft_in::AbstractFFTs.Plan{ComplexF64}
    buff_out::Vector{T}
end

function CConv(
    ::Type{T},
    input_dim::Int64,
    kernel_dim::Int64;
    padding = 0,
    crop_range = (1, input_dim),
    phase = _default_phase_kernel(kernel_dim),
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
        kernel_dim,
        padding,
        crop_range,
        phase,
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

function _load_kernel!(C::CConv, kernel::AbstractVector{<:Real}, flip_kernel)
    fill!(C.buff_ker, 0.0) # Prepare zero padding
    copy!(view(C.buff_ker, axes(kernel, 1)), kernel)
    if C.phase != 0 # Correct Phase
        circshift!(C.buff_ker, -C.phase)
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
)
    _load_sig!(C, X)
    if load_kernel
        _load_kernel!(C, kernel, flip_kernel)
    end
    _conv!(C)
    return copy(C.buff_out)
end

struct ScaleParams
    b::Real
    g::Real
    J::Int64
    Q::Int64
    wmin::Float64
    wmax::Float64
    wave_dim::Int64
end

struct TimeParams
    kernel_dim::Int64
    kernel_type::Symbol
    kernel_params::AbstractArray{<:Real}
end

function wavelet_parameters(; b, g, J, Q, wmin, wmax)
    wavelet_parameters(b, g, J, Q, wmin, wmax)
end

function wavelet_parameters(b, g, J, Q, wmin, wmax)
    return GMW.gmw_grid(b, g, J, Q, wmin, wmax, 0)
end

function gausskernel(kernel_dim, kernel_params)
    s = kernel_params[1]
    t = LinRange(0, 1, kernel_dim)
    g = exp.(-0.5 * ((t .- 0.5) / s) .^ 2)
    g = g / sum(g)
    return g
end

function gauss_expo_kernel(kernel_dim, kernel_params)
    s, alpha, n = kernel_params
    sigmas = exp.([log(s) * i * alpha for i = 0:(n-1)])
    return [gausskernel(kernel_dim, sigma) for sigma in sigmas]
end

function rectkernel(kernel_dim, kernel_params)
    n = kernel_dim
    t = LinRange(0, 1, kernel_dim)
    T = (kernel_params[1] / n) / 2
    g = zeros(Float64, n)
    g[abs.(t .- 0.5).<=T] .= 1
    g = g / sum(g)
    return g
end

function averaging_kernel(tp::TimeParams)
    averaging_kernel(tp.kernel_type, tp.kernel_params, tp.kernel_dim)
end

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

struct GMWFrame
    wave_dim::Integer # N
    gmw_params::AbstractVector{<:AbstractVector{<:Real}} # a,u,β,γ, Px4
    gmw_frame::AbstractVector{<:AbstractVector{<:Union{Real,Complex}}} # (P+1)xN
    freq_peaks::Vector{Float64} # P+1
    sigmas::Array{Float64} # P+1
    selfdual::Bool
    analytic::Bool
end

function GMWFrame(scale_params::ScaleParams; analytic = false, selfdual = true)
    (; b, g, J, Q, wmin, wmax, wave_dim) = scale_params
    gmw_params = wavelet_parameters(b, g, J, Q, wmin, wmax)
    return GMWFrame(wave_dim, gmw_params; analytic, selfdual)
end

function GMWFrame(
    N::Integer,
    gmw_params::AbstractArray{<:AbstractVector{<:Real}};
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

function cross_scalogram(
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    dt::Int64,
    gmw_frame::Union{Vector{<:AbstractArray{T}},GMWFrame},
    averaging_kernel::Union{AbstractVector{<:Real},Vector{<:AbstractVector{<:Real}}},
    analytic = false,
) where {T<:Union{Real,Complex}}
    L = [x, y]
    C = [(1, 2)]
    out = cross_scalogram(L, C, dt, gmw_frame, averaging_kernel, analytic)
    return out[C[1]]
end

function cross_scalogram(
    L::Vector{<:AbstractArray{<:Real}},
    C::Vector{Tuple{Int64,Int64}},
    dt::Int64,
    gmw_frame::Union{Vector{<:AbstractArray{T}},GMWFrame},
    averaging_kernel::Union{AbstractVector{<:Real},Vector{<:AbstractVector{<:Real}}},
    analytic = false,
) where {T<:Union{Real,Complex}}

    allequal(length, L) || throw(error("Wrong size"))
    gmw_frame = gmw_frame.gmw_frame
    work_dim = length(L[1])
    input_dim = work_dim
    wave_dim = length(gmw_frame[1])
    time_sampling = 1:dt:work_dim
    if averaging_kernel isa Vector{Float64}
        many_kernels = false
    else
        many_kernels = true
        length(gmw_frame) == length(averaging_kernel) ||
            throw(error("Number of filters and averaging kernels are not equal"))
    end
    kernel_dim = many_kernels ? length(averaging_kernel[1]) : length(averaging_kernel)

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
            f = KernelC(x_i .* y_i, avg_kernel)
            f = f[time_sampling]
            out[c][i] = f
        end
    end
    out_hcat = Dict()
    for c in C
        cs = [out[c][i] for i = 1:length(gmw_frame)]
        out_hcat[c] = stack(cs)
    end
    return out_hcat
end
