import ForwardDiff
import ForwardDiff: Dual
import ChainRulesCore
import ForwardDiffChainRules


# WIP
#function ChainRulesCore.frule(deltas,C::typeof(CConv),X::AbstractArray{<:Real},params::AbstractArray{<:Real},flip_kernel=false)
#  _,_,Δp,_ = deltas
#    X = _prepare_sig(C, X)
#    # Prepare dual values
#    out = Array{Float64} Dual{T,V,N}}(undef, C.output_dim...)
#    Δout = 
#    kernel = getkernel(C, params) 
#    out = _conv!(C, X, C.buff_work, flip_kernel)
#    partials = map(1:N) do i
#        # Copy kernel partials to C.buff_work
#        map!(d -> ForwardDiff.partials(d)[i], C.buff_work, kernel)
#        return _conv!(C, X, C.buff_work, flip_kernel)
#    end
#    # Copy values and partials to out_dual
#    for i in LinearIndices(out_dual)
#        p = ForwardDiff.Partials{N,V}(([p[i] for p in partials]...,))
#        out_dual[i] = Dual{T,V,N}(out[i], p)
#    end
#    return out_dual
#end

# ForwardDiff Overload Function to allow Automatic Differentiation
function (C::CConv)(
  X::AbstractArray{<:Real},
  params::Array{Dual{T,V,N}},
  flip_kernel=false,
) where {T,V,N}
  _load_sig!(C, X)
  # Prepare dual values
  out_dual = Array{Dual{T,V,N}}(undef, C.output_dim...)
  kernel = getkernel(C, params)
  # Convolution with kernel_values
  ## COPYING
  map!(ForwardDiff.value, view(C.buff_work, :, :, 1), kernel)
  for k in axes(C.buff_work, 3)[2:end]
    C.buff_work[:, :, k] .= view(C.buff_work, :, :, 1)
  end

  out_value = copy(_conv!(C, C.buff_in, C.buff_work, flip_kernel))

  out_partials = map(1:N) do i
    # Reload sig for each partial 
    _load_sig!(C, X)
    # Copy kernel partials to C.buff_work
    map!(d -> ForwardDiff.partials(d)[i], view(C.buff_work, :, :, 1), kernel)
    for k in axes(C.buff_work, 3)[2:end]
      C.buff_work[:, :, k] .= view(C.buff_work, :, :, 1)
    end
    return copy(_conv!(C, C.buff_in, C.buff_work, flip_kernel))
  end
  # Copy values and partials to out_dual
  for i in LinearIndices(out_dual)
    p = ForwardDiff.Partials{N,V}(([p[i] for p in out_partials]...,))
    out_dual[i] = Dual{T,V,N}(out_value[i], p)
  end
  return out_dual
end

function _wavelet_frame(
  C::CConv,
  params::Array{Dual{T,V,M}},
  kernel_dim::Tuple{Integer,Integer},
  buff_work::AbstractArray{<:Complex},
  buff_work_v::AbstractArray{<:Complex},
  pifft_wave,
  freq_corr::Union{Real,AbstractArray{<:Complex}}=1,
  selfdual=true,
  analytic=false,
) where {T,V,M}
  K = size(params, 2) + 1  # Number of wavelets + 1 low pass
  N = kernel_dim[1] # Wavelet size
  L = div(N, 2) + 1 # Analytical fft size
  (K - 1) == kernel_dim[2] || throw(error("Wrong size"))

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

  # Create the Low-Pass
  b_v = view(buff_work, 1:L, K, 1)
  w = (0:(L-1)) / N
  freq_peaks = mapslices(p -> peak_n(p, 1), params, dims=1)
  w0 = minimum(freq_peaks)
  ϕ = exp.(-(3 * log(10) / 20) * (w / w0) .^ 2) # -3dB at w=w0
  b_v .= ϕ

  b_v = view(buff_work, 1:L, :, 1)
  if selfdual
    # Normalization
    ψ_norm = sqrt.(sum(abs2, b_v, dims=2))
    if analytic
      if iseven(N)
        ψ_norm[2:(L-1)] /= sqrt(2)
      else
        ψ_norm[2:L] /= sqrt(2)
      end
    end
    b_v ./= ψ_norm
  end

  # Frequency Correction
  b_v .*= freq_corr

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


