using Random, StatsFuns, FFTW
using GeneralizedMorseWavelets

_map = StatsFuns.softplus
_invmap = StatsFuns.invsoftplus
_default_phase_kernel(kernel_dim) =
    iseven(kernel_dim) ? div(kernel_dim, 2) - 1 : div(kernel_dim, 2)
_copy_real!(x::AbstractArray{Float64}, y::AbstractArray{ComplexF64}) = map!(real, x, y)

struct CConv{T<:Union{Float64,ComplexF64}} # 1D Convolution with constrained kernel
    input_dim::Tuple{Integer,Integer}
    output_dim::Tuple{Integer,Integer,Integer}
    kernel_dim::Tuple{Integer,Integer}
    padding::Integer
    kernel_init_params::Function # Function to initialize parameters
    kernel_func::Function
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
    kernel_init_params= ()->nothing,
    kernel_func=(C,x)->identity(x),
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
        kernel_init_params,
        kernel_func,
        crop_phase_output!,
        buff_in,
        pfft_in,
        buff_work,
        pfft_work,
        pifft_work,
        buff_out,
    )
end

function initialparameters(rng::AbstractRNG, C::CConv)
    C.kernel_init_params(rng) # inverse mapping of parameters
end

function getkernel(C::CConv, params::AbstractArray)
    C.kernel_func(C, params) # parameters to kernel
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
    # TODO implement 2D fast fourier
    C.buff_work .*= C.buff_in
    C.pifft_work * C.buff_work # buff_work in place ifft
    C.crop_phase_output!(C.buff_work) # update buff_out
end

function (C::CConv)(X::AbstractArray{<:Real}, params, flip_kernel = false)
    X = _load_sig!(C, X)
    kernel = getkernel(C, params)
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
              copyto!(buff_out_v,Y_v)
            end
        end
        return buff_out
    end
    return crop_phase_func!
end

## wavelet kernel init from a finite number of fixed parameters
function _init_parameters_wave(
    rng = nothing;
    β = 1,
    γ = 3,
    J = 32,
    Q = 2,
    wmin = 0,
    wmax = pi,
) # rng not used, kept however
    params = gmw_grid(β, γ, J, Q, wmin, wmax, 0)
    return hcat(params...)
end

function _wf_get_gmfs!(buff_work,params,L,N)
    # Fill buff_work with zeros (does non-analytical part zero padding + time zero padding)
    fill!(buff_work, 0.0)

    # Fill buff_work of analytical wavelets
    for j in axes(params, 2)
        a = params[1, j]
        β = params[3, j]
        γ = params[4, j]
        b_v = view(buff_work, 1:L, j, 1)
        gmw!(b_v, 0, a, 0, β, γ, N, :peak)
    end
    return buff_work
end

function _wf_get_lp!(buff_work,params,L,N,K)
    # Create the Low-Pass
    b_v = view(buff_work, 1:L, K, 1)
    w = (0:(L - 1)) / N
    freq_peaks = mapslices(p -> peak_n(p, 1), params, dims = 1)
    w0 = minimum(freq_peaks)
    ϕ = exp.(-(3 * log(10) / 20) * (w / w0) .^ 2) # -3dB at w=w0
    b_v .= ϕ
    return b_v
end

function _wf_make_selfdual!(buff_work,N,L,selfdual,analytic)
    b_v = view(buff_work, 1:L, :, 1)
    if selfdual
      # Normalization
      ψ_norm = sqrt.(sum(abs2, b_v, dims = 2))
      if analytic
        if iseven(N)
          ψ_norm[2:(L-1)] /=sqrt(2)
        else
          ψ_norm[2:L] /=sqrt(2)
        end
      end
      b_v ./= ψ_norm
    end
    return b_v
end

function _wf_to_time!(buff_work,buff_work_v,pifft_wave,N,L,analytic)
    # Real part of inverse discrete fourier transform of analytical signal: need half coeffs at k=0 and k=N/2 (only if N is even)
    # We need to do this as we are not using irfft functions only plan_fft and plan_ifft, if we take the real part of the inverse discrete fourier transform we have to halve some coeffs
    if !analytic
      buff_work[1, :, 1] /= 2
      buff_work_v[L, :, 1] /= iseven(N) ? 2 : 1
    end

    # All good: copy first slice along third dimension
    for i in axes(buff_work, 3)[2:end]
        buff_work[:, :, i] .= buff_work[:, :, 1]
    end

    pifft_wave * buff_work_v # In place inverse fourier transform
    if !analytic
      map!(x -> 2 * real(x), buff_work_v, buff_work_v) # Take 2*Real(ψ) but with properly halved coefficients at k=0 and k=N/2
    end
    for j in axes(buff_work, 2), k in axes(buff_work, 3)
        b_v = view(buff_work, 1:N, j, k)
        circshift!(b_v, _default_phase_kernel(N))
    end
    return buff_work
end

function _wavelet_frame(
    C::CConv,
    params::AbstractArray{<:Real},
    kernel_dim::Tuple{Integer,Integer},
    buff_work::AbstractArray{<:Complex},
    buff_work_v::AbstractArray{<:Complex},
    pifft_wave,
    freq_corr::Union{Real,AbstractArray{<:Complex}} = 1,
    selfdual=true,
    analytic=false,
)
    K = size(params, 2) + 1  # Number of wavelets + 1 low pass
    N = kernel_dim[1] # Wavelet size
    L = div(N, 2) + 1 # Analytical fft size
    (K - 1) == kernel_dim[2] || throw(error("Wrong size"))

    _wf_get_gmfs!(buff_work,params,L,N)

    _wf_get_lp!(buff_work,params,L,N,K)

    _wf_make_selfdual!(buff_work,N,L,selfdual,analytic)

    # Frequency Correction
    b_v = view(buff_work, 1:L, :, 1)
    b_v .*= freq_corr

    _wf_to_time!(buff_work,buff_work_v,pifft_wave,N,L,analytic)

    return buff_work
end

function WaveletConv(input_dim, kernel_dim; padding = 0, freq_corr = 1, selfdual=true, analytic=false,conv_kwargs...)
    conv_dim = (input_dim[1] + padding, kernel_dim[2] + 1, input_dim[2])
    buff_work = Array{ComplexF64}(undef, conv_dim)
    buff_work_v = view(buff_work, 1:kernel_dim[1], :, :)
    pifft_wave = plan_ifft!(buff_work_v, 1)
    #kernel_func(C::CConv,wave_fft) = _wavefft_to_wavetime!(C,wave_fft,buff_work,buff_work_v,pifft_wave)
    kernel_func(C::CConv, params) =
        _wavelet_frame(C, params, kernel_dim, buff_work, buff_work_v, pifft_wave, freq_corr, selfdual, analytic)
    return_type = analytic ? ComplexF64 : Float64
    CConv(
        return_type,
        input_dim,
        (kernel_dim[1], kernel_dim[2] + 1);
        kernel_init_params = () -> nothing,
        kernel_func = kernel_func,
        buff_work = buff_work,
        padding = padding,
        conv_dim = conv_dim,
        conv_kwargs...,
    )
end

function _gausskernel(kernel_dim, params, padding)
    σ = params[1]
    u = length(params) == 2 ? params[2] : 0
    t = LinRange(0, 1, kernel_dim)
    g = exp.(-0.5 * ((t .- 0.5 .+ u) / σ) .^ 2)
    g = vcat(g, zeros(padding))
    g = g / sum(g)
    return g
end

function GaussConv(input_dim, kernel_dim; padding = 0, conv_kwargs...)
    conv_dim = (input_dim[1] + padding, 1, input_dim[2])
    kernel_func(C::CConv, params) = reshape(
        _gausskernel(kernel_dim[1], params, conv_dim[1] - kernel_dim[1]),
        conv_dim[1],
        1,
        1,
    )
    CConv(
        Float64,
        input_dim,
        kernel_dim;
        kernel_init_params = () -> nothing,
        kernel_func = kernel_func,
        conv_dim = conv_dim,
        padding = padding,
        conv_kwargs...,
    )
end

#function _rectkernel(C::CConv, params)
#    dim = C.kernel_dim
#    padding = params[1]
#    zero_padding = zeros(padding)
#    N = dim - padding * 2
#    if N <= 0
#        throw(
#            error(
#                "The dimension of the kernel must be higher than the total amount of padding, i.e. kernel_dim > 2*padding",
#            ),
#        )
#    end
#    kernel = vcat(zero_padding, ones(N), zero_padding, zeros(C.input_dim[1] - dim)) / N
#    return kernel
#end
#_rectkernel_init_params(rng::AbstractRNG) = [0]
#
#function RectConv(input_dim, kernel_dim; conv_kwargs...)
#    CConv(
#        input_dim,
#        kernel_dim;
#        kernel_init_params = _rectkernel_init_params,
#        kernel_func! = _rectkernel,
#        conv_kwargs...,
#    )
#end

"""
    init_wave_conv_kernel(work_dim;β,γ,J,Q,wmin,wmax,wave_dim,fs,with_sigma=false)

Initialize a wavelet convolutional kernel. With padding and same dimension as input.

# Arguments
  - `work_dim::Union{Integer,Tuple{Integer,Integer}}`: Input dimension
  - `wave_dim::Integer`: maximum time support in time index of wavelets
  - `β::Float64`: First shape parameter of Generalized Morse Wavelets
  - `γ::Float64`: Second shape parameter
  - `J::Integer`: Number of frequency octaves
  - `Q::Integer`: Number of frequency inter-octaves
  - `fmin::Float64`: Minimum frequency peak allowed
  - `fmax::Float64`: Maximum and starting frequency peak allowed
  - `fs`: Sampling frequency
  - `with_sigma=false`: Return wavelet filters time deviation
"""
function init_wave_conv_kernel(
    work_dim::Union{Integer,Tuple{Integer,Integer}};
    β::Real,
    γ::Real,
    J::Integer,
    Q::Integer,
    fmin::Real,
    fmax::Real,
    wave_dim::Integer,
    fs::Real,
    with_sigma = false,
)
    wmin = fmin * 2pi / fs
    wmax = fmax * 2pi / fs

    wave_params =
        _init_parameters_wave(; β = β, γ = γ, J = J, Q = Q, wmin = wmin, wmax = wmax)
    K = length(wave_params)
    freq_peaks = mapslices(p -> peak_n(p, 1) * fs, wave_params, dims = 1)
    freq_peaks = vcat(freq_peaks[:], 0.0)
    work_dim = work_dim isa Integer ? (work_dim, 1) : work_dim
    WaveC = WaveletConv(work_dim, (wave_dim, size(wave_params, 2)))
    if with_sigma
      σ_t = get_time_deviation(WaveC,wave_params,wave_dim)
    else
      σ_t = nothing
    end
    return ((freq_peaks, σ_t), wave_params, WaveC)
end

function get_time_deviation(WaveC,wave_params,wave_dim)
  waves = WaveC.kernel_func(WaveC, wave_params)
  phase = _default_phase_kernel(wave_dim)
  r = range(start = phase+1, length = (wave_dim-phase-1))
  t = (0:(length(r) - 1))
  waves_v = abs2.(view(waves, r, :, 1))
  waves_v ./= sum(waves_v, dims = 1)
  waves_v .*= abs2.(t)
  σ_t = dropdims(sqrt.(sum(waves_v, dims = 1)), dims = 1)
  return σ_t
end

function get_frequency_peaks(wave_params,fs=1)
    freq_peaks = mapslices(p -> peak_n(p, 1) * fs, wave_params, dims = 1)
    freq_peaks = vcat(freq_peaks[:], 0.0)
    return freq_peaks
end

"""
    init_averaging_conv_kernel(work_dim;kernel_type,kernel_params,Δt,kernel_dim,with_sigma=false)

Initiliaze an averaging convolutional kernel.

It returns the range for sampling in time, the original parameter mapped in (-∞,+∞) and the kernel.

# Arguments
  - `work_dim::Union{Integer,Tuple{Integer,Integer}}`: Input dimension
  - `kernel_dim::Integer`: Maximum time support for the averaging kernel
  - `kernel_type::Symbol`: Type of kernel used for the averaging kernel
  - `kernel_params::Vector`: Vector of parameters for the averaging kernel
  - `Δt::Integer`: Time Sampling step of the output flux.
  - `with_sigma=false`: Return time deviation
"""
function init_averaging_conv_kernel(
    work_dim::Union{Integer,Tuple{Integer,Integer}};
    kernel_type::Symbol,
    kernel_params::AbstractArray{<:Real},
    Δt::Integer,
    kernel_dim::Integer,
    with_sigma = false,
)
    time_sampling = 1:Δt:work_dim[1]
    work_dim = work_dim isa Integer ? (work_dim, 1) : work_dim
    if kernel_type == :gaussian
      KernelC = GaussConv(work_dim, (kernel_dim, 1))
        
      if with_sigma
          σ_t = KernelC.kernel_dim[1] * kernel_params[1]
      else
          σ_t = nothing
      end
    else
      throw(error("Kernel type $(kernel_type) not implemented"))
    end

    return ((time_sampling, σ_t), kernel_params, KernelC)
end
