using Statistics
using StatsBase
using PhysicalConstants
using Unitful
using DataInterpolations
using RegularizationTools

const λ=40660/1000 # "J.mmol^-1" latent heat of evaporation of water
const C_p=29.07 # Molar Heat Capacity at constant pressure J.mol^-1.K^-1
const R=ustrip(PhysicalConstants.CODATA2018.R)

"Convert vector `F` to flux of type `fluxtype` given the density `density`"
function tofluxunits(F,density,fluxtype)
  if fluxtype == :SensibleHeat
    F=F.* density * C_p
    units=u"J/m^2/s"
  elseif fluxtype == :LatentHeat
    F=F.* density * λ
    units=u"J/m^2/s"
  elseif fluxtype == :CO2
    F=F.* density
    units =u"μmol" / u"m^2" / u"s"
  else
    throw(error("wrong flux type"))
  end
  return (F,units)
end

function map_idx(CI,mapping)
  X=Array{Float64}(undef,length(CI))
  Y=Array{Float64}(undef,length(CI))
  Z=Array{Float64}(undef,length(CI))
  for (i,c) in enumerate(CI)
    x,y,z=mapping(c)
    X[i]=x
    Y[i]=y
    Z[i]=z
  end
  return (X,Y,Z)
end

"""
    time_integrate_flux(decomp,mask,density,fluxtype=nothing)

Integrate along the scale domain the flux `decomp` decomposed in the time-scale domain given the integration mask `mask`and scale it using the density `density`.

Optionnaly a symbol `fluxtype`can be given to indicate the type of flux such that it is converted to common flux units, see `tofluxunits`.
"""
function time_integrate_flux(decomp,mask,density,fluxtype=nothing)
  size(mask) == size(decomp) || throw(error("Wrong size between mask and decomp"))
  length(density) == size(decomp,1) || throw(error("Wrong size along time axis between density and decomp"))
  F=Vector{Float64}(undef,size(decomp,1))
  for j in axes(F,1)
    F[j]=sum(decomp[j,mask[j,:]])
  end
  if isnothing(fluxtype)
    return F .* density
  else
    return tofluxunits(F,density,fluxtype)
  end
end


"""
    compute_wind_amplitude(u,v,w,time_params)

Compute the wind amplitude given the three wind speed components `u`,`v`,`w` with the averaging convolutional kernel parameters `time_params`.
"""
function compute_wind_amplitude(u,v,w,time_params)
  length(u) == length(v) || throw(error("Signals of different size"))
  length(u) == length(w) || throw(error("Signals of different size"))
  work_dim = length(u)
  time_sampling,kernel_params,KernelC = init_averaging_conv_kernel(work_dim;time_params...)
  u = KernelC(u.^2,kernel_params)[time_sampling]
  v = KernelC(v.^2,kernel_params)[time_sampling]
  w = KernelC(w.^2,kernel_params)[time_sampling]
  wind_amplitude = sqrt.(u + v + w)
  return wind_amplitude
end


"""
    compute_density(P,T,time_params)

Compute the density given the pressure `P` (Pa) and the temperature `T` (K) with the averaging convolutional kernel parameters `time_params`.
"""
function compute_density(P,T,time_params)
  length(P) == length(T) || throw(error("Signals of different size"))
  work_dim = length(P)
  time_sampling,kernel_params,KernelC = init_averaging_conv_kernel(work_dim;time_params...)
  density = P ./ (R*T)
  density = KernelC(density,kernel_params)[time_sampling]
  return density
end


"""
    timescale_flux_decomp(w,θ,time_params,scale_params;with_info=false)

Compute the time-scale decomposition of the flux `wθ` given averaging and wavelet kernel parameters `time_params` and `scale_params` .

`w` and `θ` are real input vectors of the same size. The flux is computed using:

```math
F_{w\\theta}(u,\\xi) = \\langle w * \\psi_\\xi , (\\theta * \\psi _xi ) \\phi_u
```
where ``\\phi_u`` is the averaging kernel centered at time ``u`` and ``\\psi_\\xi`` a wavelet with parameters ``\\xi``

# Arguments

  - `w::Vector`: First signal (e.g. the vertical wind speed)
  - `θ::Vector`: Second signal (e.g. temperature)
  - `time_params::NamedTuple`: Named Tuple of the parameters for initializing the averaging convolutional kernel `\\phi`, see `init_averaging_conv_kernel`
  - `scale_params::NamedTuple`: Named Tuple of parameters for initializing the wavelet convolutional kernel `\\psi`, see `init_wave_conv_kernel`
  - `with_info::Bool=false`: Output informations about the decomposition

# Examples

```julia
wave_dim=1024
kernel_dim=512
params=(
  wave_dim=wave_dim, β=1, γ=3, J=8, Q=8, wmin=2*pi/wave_dim, wmax=pi,
  kernel_dim=kernel_dim, kernel_type=:gaussian, kernel_params=[2.0], Δt=4)

F=timescale_flux_decomp(w,θ;params...)
```
"""
function timescale_flux_decomp(w,θ,time_params,scale_params; with_info=false)
  length(w) == length(θ) || throw(error("Signals must be of the same size."))
  work_dim=length(w)
  freq_peak,wave_params,WaveC = init_wave_conv_kernel(work_dim;scale_params...)
  time_sampling,kernel_params,KernelC = init_averaging_conv_kernel(work_dim;time_params...)
  decomp=zeros(length(time_sampling),length(wave_params))
  for (i,ξ) in enumerate(wave_params)
    w_ξ=WaveC(w,ξ)
    θ_ξ=WaveC(θ,ξ)
    f=KernelC(w_ξ .* θ_ξ,kernel_params)
    @views copyto!(decomp[:,i],f[time_sampling])
  end
  if with_info
    info = (WaveC,wave_params,KernelC,kernel_params)
    return (time_sampling,freq_peak,decomp,info)
  else
    return (time_sampling,freq_peak,decomp)
  end
end


"""
    amplitude_reynolds_w(u,v,w,time_params,scale_params)

Compute the amplitude of the vertical components of the Reynold's tensor using the three wind speed components `u`,`v` and `w` and the time-scale decomposition parameters `time_params` and `scale_params`.

# Arguments

  - `u,v,w::Vector`: wind speed components signal (e.g. the vertical wind speed)
  - `time_params::NamedTuple`: Named Tuple of the parameters for initializing the averaging convolutional kernel `\\phi`, see`init_averaging_conv_kernel`
  - `scale_params::NamedTuple`: Named Tuple of parameters for initializing the wavelet convolutional kernel `\\psi`, see `init_wave_conv_kernel`
"""
function amplitude_reynolds_w(u,v,w,time_params,scale_params)
  _,_,uv=timescale_flux_decomp(u,w,time_params,scale_params)
  _,_,vw=timescale_flux_decomp(v,w,time_params,scale_params)
  time_sampling,freq_peak,ww=timescale_flux_decomp(w,w,time_params,scale_params)
  τ_rey_w =  sqrt.(uv.^2  .+ vw.^2 + ww.^2)
  return (time_sampling,freq_peak,τ_rey_w)
end

"""
    turbulence_mask_extraction(u,v,w,time_params,scale_params,method,method_params...)

Extract a time-scale mask of the vertical turbulent transport using the three wind speed components `u`,`v` and `w` using the time-scale decomposition parameters `time_params` and `scale_params` (see `init_averaging_conv_kernel` and `init_wave_conv_kernel`) and the turbulence extraction methods `method` with parameters `method_params`.

# Arguments

  - `u,v,w::Vector`: wind speed components signal (e.g. the vertical wind speed)
  - `time_params::NamedTuple`: Named Tuple of the parameters for initializing the averaging convolutional kernel `\\phi`, see`init_averaging_conv_kernel`
  - `scale_params::NamedTuple`: Named Tuple of parameters for initializing the wavelet convolutional kernel `\\psi`, see `init_wave_conv_kernel`
  - `method::Function`: Method used to extract the turbulent transport signal, see `turbu_extract_threshold`,`turbu_extract_laplacian` and `turbu_extract_diffusion`.
"""
function turbulence_mask_extraction(u,v,w,time_params,scale_params;method::Function,method_params...)
  _,_,τ_rey_w=amplitude_reynolds_w(u,v,w,time_params,scale_params)
  return method(τ_rey_w;method_params...)
end
function turbulence_mask_extraction(τ_rey_w,;method::Function,method_params...)
  return method(τ_rey_w;method_params...)
end

function turbu_extract_threshold(τ_rey_w;threshold) # Unused params, it is ok
  mask=τ_rey_w .> threshold
  return mask
end

function _interpolate_eta(t,eta)
  perm=sortperm(t)
  #A=RegularizationSmooth(eta[perm], t[perm], 3;λ=2,alg=:fixed)
  A = BSplineApprox(eta[perm], t[perm], 3, 6, :Uniform, :Uniform); # Not a good choice, it depends on the end points
  tmin=minimum(t)
  tmax=maximum(t)
  g(t) = begin
          if tmin <= t <= tmax
              return A(t)
          elseif t < tmin
              return A(tmin)
          else 
              return A(tmax)
      end
  end
  return g
end

function turbu_extract_laplacian(τ_rey_w;time_sampling,freq_peak,
                                  δ_Δτ=1,
                                  δ_τ=-6,
                                  mean_wind=nothing,
                                  ref_dist=1,
                                  weighted_laplacian=false,
                                  weight_type=:value,
                                 )
  S=size(τ_rey_w)
  CI=CartesianIndices(S)
  if isnothing(mean_wind)
    vertex_mapping=(c::CartesianIndex) -> Float64[time_sampling[c[1]],log.(freq_peak[c[2]]*ref_dist),log(τ_rey_w[c[1],c[2]])]
  else
    vertex_mapping=(c::CartesianIndex) -> Float64[time_sampling[c[1]],log.(freq_peak[c[2]]*ref_dist/mean_wind[c[1]]),log(τ_rey_w[c[1],c[2]])]
  end
  t,eta,τ_mapped=map_idx(CI,vertex_mapping)

  # Reject low-pass filter i.e. at freq_p[end]
  mask=trues(S)
  mask[:,end].=false

  adj_mat=grid_adj_mat(S,mask) # 9-point grid adjacency matrix with removed vertices from mask
  if weighted_laplacian
    # We use a gaussian kernel parametrized with the covariance matrix of the data (time,scale,value). The idea is to penalize high displacement in some space.
    Σ=@views cov(hcat(t[mask[:]],eta[mask[:]],τ_mapped[mask[:]]))
    #μ_yz=@views [mean(Y[mask[:]]),mean(τ_mapped[mask[:]])]
    if weight_type == :time_scale_value
      weight_vertex_map = c -> vertex_mapping(c)
    elseif weight_type == :scale_value
      weight_vertex_map = c -> vertex_mapping(c)[2:3]
      Σ=Σ[2:3,2:3]
    elseif weight_type == :time_scale
      weight_vertex_map = c -> vertex_mapping(c)[1:2]
      Σ=Σ[1:2,1:2]
    elseif weight_type == :value
      weight_vertex_map = c -> vertex_mapping(c)[3:3]
      Σ=Σ[3:3,3:3] # we use a 3:3 instead of 3 to keep a matrix as Σ
    else
      throw(error("Prob"))
    end
    weight_func = make_gaussian_kernel(S,weight_vertex_map,Σ)
    weights_mat = generate_weight_mat(adj_mat,weight_func)
  else
    weights_mat = adj_mat # Using the adjacency matrix as the weight matrix amounts to compute a normal laplacian
  end
  g = MyGraph(adj_mat,weights_mat)
  L = laplacian_matrix(g)
  Δτ = L*τ_mapped # the laplacian is zero where mask is false

  # Find all important minimas, keep the highest scale
  detected = reshape(δ_Δτ .< Δτ,S) # Look at (t,eta) points with important minimas 
  for v in eachslice(detected,dims=1)
    idx = findfirst(v)
    if isnothing(idx) || idx == S[2] # Nothing to do if minimas detected at the lowest scale or no minimas detected
      continue
    else
      v[idx+1:end] .= false # Keep the highest scale, remove all other minimas at lower scale at that time
    end
  end
  itp = _interpolate_eta(t[detected[:]],eta[detected[:]]) # Bspline interpolation + smoothness regularization to get interpolated value at each time t, extrapolate with constant values on the borders
  mask_advec = itp.(t) .< eta  .&& mask[:]# Get the mask removing the advection + removing the mean value
  mask_lowcoeff = δ_τ .< τ_mapped .&& mask_advec

  masks=(detected,reshape(mask_advec,S),reshape(mask_lowcoeff,S))
  tau_rey=(reshape(t,S),reshape(eta,S),reshape(τ_mapped,S),reshape(Δτ,S),itp)
  return (masks,tau_rey)
end

function turbu_extract_diffusion(τ_w;time_sampling,freq_peak,ref_dist=1,mean_wind=nothing)
  S=size(τ_w)
  CI=CartesianIndices(S)
  # Reject low-pass filter i.e. at freq_p[end]
  mask=trues(S)
  mask[:,end].=false
  if isnothing(mean_wind)
    vertex_mapping=(c::CartesianIndex) -> Float64[time_sampling[c[1]],log.(freq_peak[c[2]]*ref_dist),log(τ_w[c[1],c[2]])]
  else
    vertex_mapping=(c::CartesianIndex) -> Float64[time_sampling[c[1]],log.(freq_peak[c[2]]*ref_dist/mean_wind[c[1]]),log(τ_w[c[1],c[2]])]
  end
  X,Y,τ_mapped=map_idx(CI,vertex_mapping)
  adj_mat=grid_adj_mat(S,mask)
  function inc_func(i::Int,j::Int) 
    if i<j
      return nothing
    else
      c_i=CI[i]
      c_j=CI[j]
      v_i=vertex_mapping(c_i)
      v_j=vertex_mapping(c_j)
      v=v_i-v_j
      return v
    end
  end
  Δv = apply_func_on_edges(adj_mat,inc_func)
  Δv = filter(!isnothing,Δv)
  Σ=cov(Δv)
  weight_func = make_gaussian_kernel(S,x->vertex_mapping(x),Σ)
  weights_mat = generate_weight_mat(adj_mat,weight_func;normalize=true)
  g = MyGraph(adj_mat,weights_mat)
  s=-0.1 .< Y .< 0.1
  s=sparse(vec(s))
  M=sum(s)
  func_acc(s,i)=begin
      x=s .+ droptol!(g.weights*s,1e-6)
      x=x*(M/sum(x))
      return x
  end
  #all_s=accumulate(func_acc,1:10,init=s);
  @warn  println("Not Fully implemented yet")
  return (g,Δv,Σ,(X,Y,τ_mapped))
end
