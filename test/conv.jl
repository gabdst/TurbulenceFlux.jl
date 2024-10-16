using Test
import TurbulenceFlux:
    init_wave_conv_kernel,
    init_averaging_conv_kernel,
    _init_parameters_wave,
    _default_phase_kernel,
    WaveletConv,
    GaussConv,
    get_time_deviation
using FFTW, LinearAlgebra

work_dim = 1024
wave_dim = work_dim
kernel_dim = work_dim
β = 2
γ = 3
J = floor(Int, log2(wave_dim))
Q = 1
wmin = 2pi / wave_dim
wmax = pi
Δt = 4
time_sampling = 1:Δt:work_dim

# Checking convolution kernels are working properly
## Self-dual property of the Wavelet Convolution Kernel
@testset "Testing Wavelet Convolutions padding=$padding wave_dim=$wave_dim analytic=$analytic" for wave_dim in
                                                                                [
        wave_dim-1,
        wave_dim,
    ],
    padding in [0, wave_dim - 1],
    analytic in [true,false]

    wave_params =
        _init_parameters_wave(; β = β, γ = γ, J = J, Q = Q, wmin = wmin, wmax = wmax)

    crop_range = 1:(work_dim + padding)

    WaveC = WaveletConv(
        (work_dim, 1),
        (wave_dim, size(wave_params, 2));
        crop_range ,
        padding,
        analytic,
    )

    # Check constant fourier spectrum
    # Get the waves
    waves = WaveC.kernel_func(WaveC, wave_params)
    waves_fft = fft(waves, 1)

    N=size(waves,1)
    @test isapprox(sum(abs2, waves), 1, rtol = 0.01)
    if analytic
      if iseven(N)
        target = vcat(1,2*ones(div(N,2)-2),1,zeros(N-div(N,2)))
      else
        target = vcat(1,2*ones(div(N,2)-1),zeros(N-div(N,2)))
      end
      @test isapprox(sum(abs2, waves_fft, dims = 2),target,rtol=0.1)
    else
      @test isapprox(sum(abs2, waves_fft, dims = 2), ones(size(waves, 1)), rtol = 0.1)
    end

    δ = vcat(1, zeros(work_dim - 1))
    waves_bis = WaveC(δ, wave_params)
    @test isapprox(waves, waves_bis)

    δ_ξ = sum(ifft(abs2.(waves_fft), 1), dims = 2)
    if !analytic
      @test isapprox(vcat(δ, zeros(padding)), δ_ξ, rtol = 0.1)
    end
end

@testset "Gaussian Averaging Kernel padding=$padding kernel_dim=$kernel_dim" for padding in
                                                                                 [
        0,
        kernel_dim - 1,
    ],
    kernel_dim in [kernel_dim-1, kernel_dim]

    crop_range = 1:(work_dim)
    time_sampling = 1:Δt:work_dim
    KernelC = GaussConv(
        (work_dim, 1),
        (kernel_dim, 1),
        crop_range = crop_range,
        padding = padding,
    )
    δ_train = zeros(work_dim)
    δ_train[time_sampling] .= 1 # Flux of one every Δt => should get a constant average flux of 1/Δt
    σ = [Δt / kernel_dim]
    δ_train_KC = KernelC(δ_train, σ) # The gaussian reaches 0.6 at time index u=kernel_dim*sigma, if u/Δt is too low, it should break the consistency
    @test isapprox(δ_train_KC, fill(1 / Δt, work_dim), rtol = 0.1)
end

@testset "Backend Flux test" begin

    # Final check, putting together Wavelet and Gaussian Averaging Kernels
    padding = wave_dim - 1
    wave_params =
        _init_parameters_wave(; β = β, γ = γ, J = J, Q = Q, wmin = 5 * wmin, wmax = wmax)
    crop_range = 1:(work_dim + padding)
    WaveC = WaveletConv(
        (work_dim, 1),
        (wave_dim, size(wave_params, 2)),
        crop_range = crop_range,
        padding = padding,
    )

    crop_range = (1:work_dim)
    KernelC = GaussConv(
        (work_dim + padding, size(wave_params, 2) + 1),
        (kernel_dim, 1),
        crop_range = crop_range,
        padding = 0,
    )
    σ = [Δt / 128]

    time_sampling = 1:Δt:work_dim
    const_KC = 1 # KernelC(vcat(ones(work_dim), zeros(padding)), σ)[time_sampling]

    δ_train = zeros(work_dim)
    δ_train[time_sampling] .= 1 # Flux of one every Δt => should get a constant average flux of 1/Δt
    δ_train_ξ = WaveC(δ_train, wave_params) .^ 2  # Square it
    δ_train_ξ = dropdims(δ_train_ξ, dims = 3)
    δ_train_ξ = KernelC(δ_train_ξ, σ) # Average it
    δ_train_ξ = δ_train_ξ[time_sampling, 1, :] # Sample it
    flux_value = sum(δ_train_ξ) / length(time_sampling) #  Sum it, should be 1/Δt, the average flux over the entire period is 1/Δt for the dirac_train
    @test isapprox(flux_value, 1 / Δt, rtol = 0.1)
end

@testset "Frontend flux test work_dim=$(work_dim)" for work_dim in [(work_dim,1),(work_dim,2)]
    scale_params = (
        β = β,
        γ = γ, 
        J = J,
        Q = Q, 
        fmin = wmin / (2 * pi),
        fmax = wmax / (2 * pi),
        fs = 1,
        wave_dim = wave_dim,
    )

    ((freq_peaks, σ_t), wave_params, WaveC) =
        init_wave_conv_kernel(work_dim; scale_params..., with_sigma = true)

    time_params = (
        kernel_type = :gaussian,
        kernel_dim = kernel_dim,
        kernel_params = [Δt / kernel_dim],
        Δt = Δt,
    )
    ((time_sampling, σ_t), mapped_kernel_params, KernelC) = init_averaging_conv_kernel(
    (work_dim[1], size(wave_params, 2) + 1);
        time_params...,
        with_sigma = true,
    )

    σ = mapped_kernel_params
    δ_train = zeros(work_dim)
    δ_train[time_sampling, :] .= 1 # Flux of one every Δt => should get a constant average flux of 1/Δt

    δ_train_ξ = WaveC(δ_train, wave_params) .^ 2  # Square it
    for j in 1:work_dim[2]
      x = view(δ_train_ξ,:,:,1)
      x = KernelC(x, σ) # Average it
      x = x[time_sampling, 1, :] # Sample it
      flux_value = sum(x) / length(time_sampling) #  Sum it, should be 1/Δt, the average flux over the entire period is 1/Δt for the dirac_train

      @test isapprox(flux_value, 1 / Δt, rtol = 0.1)
    end
end

@testset "Frequency Correction with phase=$(phase) " for phase in [-5,5]
  import GeneralizedMorseWavelets:wdomain
  phase_corr = exp.(im*wdomain(phase,wave_dim,div(wave_dim,2)+1))
  δ_dirac = vcat(1,zeros(work_dim-1))
  δ_dirac_shifted = circshift(δ_dirac,-phase)

  wave_params =
      _init_parameters_wave(; β = β, γ = γ, J = J, Q = Q, wmin = wmin, wmax = wmax)

  crop_range = 1:(work_dim)

  WaveC = WaveletConv(
      (work_dim, 1),
      (wave_dim, size(wave_params, 2)),
      crop_range = crop_range,
      freq_corr = sqrt.(phase_corr)
  )

  out = fft(WaveC(δ_dirac,wave_params),1)
  out .*= out
  out = sum(ifft(out,1),dims=2)

  @test isapprox(out[:]' * δ_dirac_shifted,1,rtol=0.1)
end


@testset "Frequency Correction with inverse filter" begin
  import GeneralizedMorseWavelets:wdomain
  work_dim=1024
  N=wave_dim=128
  w0 = 2pi*(div(N,2)+1)/N
  filter_corr = 1 .+ wdomain(1,N,div(N,2)+1)/w0
  sig_target = irfft(filter_corr,N)
  δ_dirac = vcat(1,zeros(work_dim-1))

  wave_params =
      _init_parameters_wave(; β = β, γ = γ, J = J, Q = Q, wmin = wmin, wmax = wmax)

  crop_range = 1:(work_dim)

  WaveC = WaveletConv(
      (work_dim, 1),
      (wave_dim, size(wave_params, 2)),
      crop_range = crop_range,
      freq_corr = sqrt.(filter_corr)
  )

  out = fft(WaveC(δ_dirac,wave_params),1)
  out .*= out # Convolution with self
  out = sum(ifft(out,1),dims=2)
  @test isapprox(sig_target,vcat(out[1:64],out[end-63:end]),rtol=0.1)
end
