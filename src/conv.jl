using Random,StatsFuns,FFTW
using GeneralizedMorseWavelets
# Require GMW lib
# Inspired by the framework of Lux.jl, not totally implementing theirs for now

_map=StatsFuns.softplus
_invmap=StatsFuns.invsoftplus
_phase_kernel(kernel_dim)=iseven(kernel_dim) ? div(kernel_dim,2)-1 : div(kernel_dim,2)

struct CConv # Convolution with a Constrained Kernel
  input_dim::Tuple{Integer,Integer}
  output_dim::Tuple{Integer,Integer}
  kernel_dim::Integer
  kernel_init_params::Function # Function to initialize parameters
  kernel_params_map::Tuple{Function,Function} # Function to map parameters to true parameters used by kernel_func
  kernel_func::Function # Functions taking parameters and outputing the kernel
  phase_output::Function
end

function initialparameters(rng::AbstractRNG,C::CConv)
  C.kernel_params_map[2](C.kernel_init_params(rng)) # inverse mapping of parameters
end

function getkernel(C::CConv,params)
  params=C.kernel_params_map[1](params) # parameters to some mapped parameters
  C.kernel_func(C,params) # parameters to kernel
end

function (C::CConv)(X::AbstractArray,params,flip_kernel=false)
  size(X,1) == C.input_dim[1] || throw(error("Wrong input size. Required $(C.input_dim)"))
  kernel=getkernel(C,params)
  fft_X=rfft(X,1) # TODO: use plan_rfft/plan_irfft to speed up
  fft_kernel=flip_kernel ? conj(rfft(kernel,1)) : rfft(kernel,1)
  Y=irfft(fft_X .* fft_kernel,C.input_dim[1],1)
  C.phase_output(Y) # Apply phase if required
end

function _make_phase_output_func(r,phase_output,input_dim,output_dim)
  rs = map(1:input_dim[2]) do j
    if iszero(phase_output[j])
      return map(i->mod(i,1:input_dim[1]),r) .+ (j-1)*input_dim[1]
    else
      return map(i->mod(i,1:input_dim[1]),r .- phase_output[j]) .+ (j-1)*input_dim[1]
    end
  end
  r = vcat(rs...)
  phase_output=Y-> reshape(Y[r],(output_dim...)) # Y seen here as a contiguous array of data with a stride of input_dim
end

## wavelet kernel init from a finite number of fixed parameters
function _init_parameters_wave(rng=nothing;β=1,γ=3,J=32,Q=2,wmin=0,wmax=pi) # rng not used, kept however
  params=gmw_grid(β,γ,J,Q,wmin,wmax,0) 
  params=vcat(params,:anti) # Add filter for the remaining spectrum not spanned by the frame of wavelets
  return params
end

function _init_mappings_wave(kernel_dim,params;with_low_pass=true,selfdual=true)
  # params is a fixed list of wavelet parameters 
  d=Dict()
  for p in params
    if p == :anti
      continue
    end
    a=p[1]
    β=p[3]
    γ=p[4]
    ψ_p=gmw(0,a,0,β,γ,kernel_dim,:peak)
    d[p]=ψ_p
  end
  L=div(kernel_dim,2)+1
  w=(0:L-1)/kernel_dim
  if with_low_pass
    w0=minimum(map(x->peak_n(x,1),filter(x->!isa(x,Symbol),params)))
    ϕ=exp.(-(3/20)*(w/w0).^2)
    d[:anti]=ϕ
  else
    ϕ=zeros(L)
    ϕ[1]=1
    d[:anti]=ϕ
  end
  if selfdual # ∑_ξ |ψ_ξ|^2 = 1 
    ψ_norm = sqrt.(sum(x->abs2.(x),values(d)))
  else #diracdual # ∑_ξ ψ_ξ = 1
    ψ_norm = sum(values(d)) 
  end

  for p in params
    ψ_p = d[p] ./ ψ_norm
    d[p]=ψ_p 
  end
  #d[:anti][2:end] .= d[:anti][2:end] ./ ψ_norm[2:end] # Nothing to do at idx=1

  _map = p -> d[p]
  _invmap = identity
  return (_map,_invmap)
end

function _fixed_wave_func(C::CConv,kernel)
  # Assuming that params is the kernel in fourrier space without padding
  g=vcat(fftshift(irfft(kernel,C.kernel_dim)),zeros(C.input_dim[1]-C.kernel_dim))
  reshape(g,(length(g),1,1))
end

function WaveletConv(input_dim,wave_dim,wave_params_map;
  wave_func=_fixed_wave_func,
  phase_output=nothing,
  r=wave_dim:(input_dim[1]+wave_dim-1))
  if isnothing(phase_output)
    default_phase=_phase_kernel(wave_dim)
    phase_output=[ default_phase for _ in 1:input_dim[2]]
  end
  wave_params_map=wave_params_map 
  output_dim=input_dim
  phase_output_func=_make_phase_output_func(r,phase_output,input_dim,output_dim)
  return CConv(input_dim,output_dim,wave_dim,()->nothing,wave_params_map,wave_func,phase_output_func)
end

function _rectkernel(C::CConv,params)
  dim=C.kernel_dim
  padding=params[1]
  zero_padding=zeros(padding)
  N=dim-padding*2
  if N <= 0
    throw(error("The dimension of the kernel must be higher than the total amount of padding, i.e. kernel_dim > 2*padding"))
  end
  kernel=vcat(zero_padding,ones(N),zero_padding,zeros(C.input_dim[1]-dim))/N
  return kernel
end
_rectkernel_params_map=(identity,identity) 
_rectkernel_init_params(rng::AbstractRNG)=[0]

function _gausskernel(C::CConv,params)
  dim=C.kernel_dim
  σ=params[1]
  t=(0:(dim-1))/(dim-1)
  g=vcat(exp.(-0.5*((t .- 0.5) / σ) .^2 ),zeros(C.input_dim[1]-C.kernel_dim)) # Not optimized but needed for pullback gradients until a proper ChainRulesCore.rrule is defined for this function.
  return reshape(g/sum(g),(C.input_dim[1],1,1))
end

_gausskernel_params_map=(x->_map.(x),x->_invmap.(x))

function _gausskernel_init_params(rng::AbstractRNG)
  μ=1
  σ=1/1024
  k=μ^2 / σ
  θ=σ / μ
  d=Gamma(k,θ)
  [rand(rng,d)]
end

#DepthWiseConv
function GaussConv(input_dim,kernel_dim;
  kernel_func=_gausskernel,
  kernel_params_map=_gausskernel_params_map,
  kernel_init_params=_gausskernel_init_params,
  phase_output=zeros(input_dim[2]),
  r=kernel_dim:input_dim[1] )
  output_dim=(length(r),input_dim[2])
  phase_output_func=_make_phase_output_func(r,phase_output,input_dim,output_dim)
  return CConv(input_dim,output_dim,kernel_dim,kernel_init_params,kernel_params_map,kernel_func,phase_output_func)
end

function RectConv(input_dim,kernel_dim;
  kernel_func=_rectkernel,
  kernel_params_map=_rectkernel_params_map,
  kernel_init_params=_rectkernel_init_params,
  phase_output=zeros(input_dim[2]),
  r=kernel_dim:input_dim[1] )
  output_dim=(length(r),input_dim[2])
  phase_output_func=_make_phase_output_func(r,phase_output,input_dim,output_dim)
  return CConv(input_dim,output_dim,kernel_dim,kernel_init_params,kernel_params_map,kernel_func,phase_output_func)
end


"""
    init_wave_conv_kernel(work_dim;β,γ,J,Q,wmin,wmax,wave_dim,fs=1)

Initialize a wavelet convolutional kernel.

# Arguments
  - `work_dim::Integer`: Input dimension
  - `wave_dim::Integer`: maximum time support in time index of wavelets
  - `β::Float64`: First shape parameter of Generalized Morse Wavelets 
  - `γ::Float64`: Second shape parameter
  - `J::Integer`: Number of frequency octaves
  - `Q::Integer`: Number of frequency inter-octaves
  - `wmin::Float64`: Minimum frequency peak allowed (in pulsation)
  - `wmax::Float64`: Maximum and starting frequency peak allowed (in pulsation)
  - `fs=1`: Sampling frequency
"""
function init_wave_conv_kernel(work_dim;β,γ,J,Q,wmin,wmax,wave_dim,fs=1)
  wave_params=_init_parameters_wave(;β=β,γ=γ,J=J,Q=Q,wmin=wmin,wmax=wmax)
  freq_peak=Float64.(map(x->x==:anti ? 0. : GeneralizedMorseWavelets.peak_n(x,1)*fs,wave_params))
  wave_params_map=_init_mappings_wave(wave_dim,wave_params)
  r=(1:work_dim) .+ (wave_dim-1)
  phase_output=[_phase_kernel(wave_dim) for _ in 1:2]
  WaveC=WaveletConv((work_dim,1),wave_dim,wave_params_map,phase_output=phase_output,r=r)
  return (freq_peak,wave_params,WaveC)
end

"""
    init_averaging_conv_kernel(work_dim;kernel_type,kernel_params,Δt,kernel_dim)

Initiliaze an averaging convolutional kernel.

# Arguments
  - `work_dim::Integer`: Input dimension
  - `kernel_dim::Integer`: Maximmum time support for the averaging kernel
  - `kernel_type::Symbol`: Type of kernel used for the averaging kernel
  - `kernel_params::Vector`: Vector of parameters for the averaging kernel
  - `Δt::Integer`: Time Sampling step of the output flux.
"""
function init_averaging_conv_kernel(work_dim;kernel_type,kernel_params,Δt,kernel_dim)
  r=(1:work_dim) .+ (kernel_dim-1)
  time_sampling=1:Δt:work_dim
  phase_output=[_phase_kernel(kernel_dim) for _ in 1:2]
  if kernel_type == :gaussian 
    KernelC=GaussConv((work_dim,1),kernel_dim,r=r,phase_output=phase_output,kernel_params_map=(identity,identity),kernel_init_params=()->nothing)
  elseif kernel_type == :rectangular
    KernelC=RectConv((work_dim,1),kernel_dim,r=r,phase_output=phase_output,kernel_params_map=(identity,identity),kernel_init_params=()->nothing)
  end
  return (time_sampling,kernel_params,KernelC)
end
