using Test
using FFTW, LinearAlgebra
using TurbulenceFlux
import TurbulenceFlux: CConv, _default_phase_kernel
@testset "FFT-based Convolutional Kernel" begin
    work_dim = 16
    kernel_dim = work_dim
    x = vcat(1, zeros(work_dim - 1))
    kernel = randn(work_dim)
    # Without padding
    # Without phase
    cconv = CConv(Float64, work_dim, kernel_dim; phase = 0)
    y = cconv(x, kernel)
    @test isapprox(y, kernel)
    # With phase compensation
    @testset "Phase compensation: phase=$phase " for phase in [-3, -2, 0, 2, 3]
        cconv = CConv(Float64, work_dim, kernel_dim; phase)
        y = cconv(x, circshift(kernel, phase))
        @test isapprox(y, kernel)
    end
    # With padding it is more tricky to check phase compensation
    # Without phase compensation
    work_dim = 19
    kernel_dim = 5
    x = vcat(1, zeros(work_dim - 2), 0)
    kernel = vcat(ones(kernel_dim), zeros(work_dim - kernel_dim))
    padding = nextpow(2, work_dim) - work_dim
    phase = 0
    cconv = CConv(Float64, work_dim, kernel_dim; padding, phase)
    y = cconv(x, kernel)
    @test isapprox(y, kernel)
    # With phase compensation and padding
    @testset "Phase compensation and padding" for kernel_dim in [4, 5], work_dim in [19, 18]
        x = vcat(1, zeros(work_dim - 2), 0)
        kernel = vcat(ones(kernel_dim), zeros(work_dim - kernel_dim))
        phase = _default_phase_kernel(kernel_dim)
        cconv = CConv(Float64, work_dim, kernel_dim; padding)
        y = cconv(x, kernel)
        @test isapprox(y[1:(kernel_dim-phase)], kernel[1:(kernel_dim-phase)])
        @test all(isapprox.(y[(kernel_dim-phase+1):end], 0, atol = 1e-6))
    end

    @testset "Skipping Kernel Load" begin
        work_dim = 16
        kernel_dim = work_dim
        x = vcat(1, zeros(work_dim - 1))
        kernel = randn(work_dim)
        # Without padding
        # Without phase
        cconv = CConv(Float64, work_dim, kernel_dim; phase = 0)
        y = cconv(x, kernel)
        y_bis = cconv(x, randn(work_dim); load_kernel = false)
        kernel_bis = randn(work_dim)
        y_bis_2 = cconv(x, kernel_bis)
        @test isapprox(y, kernel)
        @test isapprox(y_bis, kernel)
        @test isapprox(y_bis_2, kernel_bis)
    end
end

import TurbulenceFlux: ScaleParams, GMWFrame
@testset "GMWFrame" for wave_dim in [8192, 8191]
    b = 1
    g = 3
    J = 6
    Q = 2
    wmin = 0
    wmax = pi
    sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim)
    gmw = GMWFrame(sp)
    # Test Unit Power
    power = sum(x -> norm(x)^2, gmw.gmw_frame)
    @test isapprox(power, 1)
    # Test Flat spectrum
    spec = sum(map(x -> abs2.(fft(x)), gmw.gmw_frame))
    target = ones(wave_dim)
    @test isapprox(spec, target)

    # Analytic case 
    gmw = GMWFrame(sp, analytic = true)
    # Test Unit Power
    power = sum(x -> norm(x)^2, gmw.gmw_frame)
    @test isapprox(power, 1)
    # Test Flat spectrum
    spec = sum(map(x -> abs2.(fft(x)), gmw.gmw_frame))
    if iseven(wave_dim)
        target = vcat(
            1,
            2 * ones(div(wave_dim, 2) - 1),
            1,
            zeros(wave_dim - div(wave_dim, 2) - 1),
        )
    else
        target = vcat(1, 2 * ones(div(wave_dim, 2)), zeros(wave_dim - div(wave_dim, 2) - 1))
    end
    @test isapprox(target, spec)
end

@testset "CConv with GMWFrame" for wave_dim in [8192, 8191]
    kernel_dim = work_dim = wave_dim
    b = 1
    g = 3
    J = 6
    Q = 2
    wmin = 0
    wmax = pi
    sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim)
    gmw = GMWFrame(sp)
    x = vcat(1, zeros(work_dim - 1))
    cconv = CConv(Float64, work_dim, kernel_dim;)
    x_fft = [cconv(x, g) for g in gmw.gmw_frame]
    # Test Unit Power
    power = sum(x -> norm(x)^2, x_fft)
    @test isapprox(power, 1)
    # Test Flat spectrum
    spec = sum(map(x -> abs2.(fft(x)), x_fft))
    target = ones(wave_dim)
    @test isapprox(target, spec)
end

import TurbulenceFlux: averaging_kernel
@testset "Averaging Kernels" begin
    sum_to_one(x::Vector{<:Real}) = isapprox(sum(x), 1)
    sum_to_one(x::Vector{<:Vector{<:Real}}) = all(sum_to_one.(x))
    function default_params(kernel_type::Symbol, kernel_dim)
        if kernel_type == :gaussian
            [0.2 / kernel_dim]
        elseif kernel_type == :gaussian_exponential
            [0.2 / kernel_dim, 0.5, 4]
        elseif kernel_type == :rect
            [kernel_dim / 2]
        else
            throw(error(""))
        end
    end
    @testset "Sum To One: $kernel_type" for kernel_type in
                                            [:gaussian, :gaussian_exponential, :rect],
        kernel_dim in [32, 33]

        kernel_params = default_params(kernel_type, kernel_dim)
        avg_kernel = averaging_kernel(kernel_type, kernel_params, kernel_dim)
        @test sum_to_one(avg_kernel)
    end
end

import TurbulenceFlux: cross_scalogram, GMWFrame, ScaleParams, _default_phase_kernel
@testset "Cross-Scalogram test" for work_dim in [8192, 8191]
    work_dim = 8192
    kernel_dim = wave_dim = work_dim
    b = 1
    g = 3
    J = 6
    Q = 2
    wmin = 0
    wmax = pi
    sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim)
    x1 = vcat(1, zeros(work_dim - 1))
    x2 = circshift(x1, 1)
    x3 = x2 + x1
    deltat = 1
    gmw = GMWFrame(sp)
    avg_kernel = vcat(ones(1), zeros(kernel_dim - 1))
    avg_kernel = circshift(avg_kernel, _default_phase_kernel(kernel_dim))
    out = cross_scalogram(x1, x1, deltat, gmw, avg_kernel)
    @test isapprox(sum(out), 1) # The flux is one between x1 and x1
    out = cross_scalogram(x1, x2, deltat, gmw, avg_kernel)
    @test isapprox(sum(out), 0, atol = 1e-6) # The flux is zero between x1 and x2
end
