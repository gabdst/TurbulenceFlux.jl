struct GMWFrame
    wave_dim::Integer # N
    gmw_params::AbstractArray{<:AbstractVector{<:Real}} # a,u,β,γ, Px4
    gmw_frame::AbstractArray{Vector{ComplexF64}} # (P+1)xN
    freq_peaks::AbstractArray{Float64} # P+1
    σ_waves::Array{Float64} # P+1
    selfdual::Bool
    analytic::Bool
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
    σ_waves = [sqrt(sum(abs2, t .* g[r]) / sum(abs2, g[r])) for g in gmw_frame]
    return GMWFrame(N, gmw_params, gmw_frame, freq_peaks, σ_waves, selfdual, analytic)
end

function cross_scattering(
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    deltat::Int64,
    gmw_frame::Union{Vector{<:AbstractArray{T}},GMWFrame},
    averaging_kernel::Union{Vector{Float64},Vector{<:AbstractArray{Float64}}},
    analytic = false,
) where {T<:Union{Float64,ComplexF64}}

    length(x) == length(y) || throw(error("Wrong size"))

    gmw_frame = gmw_frame isa GMWFrame ? gmw_frame.gmw_frame : gmw_frame
    work_dim = length(x)
    time_sampling = 1:deltat:work_dim
    if averaging_kernel isa Vector{Float64}
        many_kernels = false
    else
        many_kernels = true
        length(gmw_frame) == length(averaging_kernel) ||
            throw(error("Number of filters and averaging kernels are not equal"))
    end
    input_dim = (length(x), 1)
    wave_dim = (length(gmw_frame[1]), 1)
    kernel_dim =
        many_kernels ? (length(averaging_kernel[1]), 1) : (length(averaging_kernel), 1)
    return_type = analytic ? ComplexF64 : Float64

    WaveC = CConv(return_type, input_dim, wave_dim)
    KernelC = CConv(return_type, input_dim, kernel_dim)

    out = Dict()
    for (i, gmw) in enumerate(gmw_frame)
        x_i = WaveC(x, gmw)
        y_i = WaveC(y, gmw)
        x_i = reshape(x_i, length(x_i))
        y_i = reshape(y_i, length(y_i))
        if many_kernels
            f = KernelC(x_i .* y_i, averaging_kernel[i])
        else
            f = KernelC(x_i .* y_i, averaging_kernel)
        end
        f = f[time_sampling]
        out[i] = f
    end
    return out
end

function cross_scattering(
    L::Vector{<:AbstractArray{<:Real}},
    C::Vector{Tuple{Int64,Int64}},
    deltat::Int64,
    gmw_frame::Union{Vector{<:AbstractArray{T}},GMWFrame},
    averaging_kernel::Union{Vector{Float64},Vector{<:AbstractArray{Float64}}},
    analytic = false,
) where {T<:Union{Float64,ComplexF64}}

    allequal(length, L) || throw(error("Wrong size"))
    gmw_frame = gmw_frame isa GMWFrame ? gmw_frame.gmw_frame : gmw_frame
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
    gmw_frame = gmw_frame isa GMWFrame ? gmw_frame.gmw_frame : gmw_frame

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
