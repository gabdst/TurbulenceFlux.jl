import TurbulenceFlux: CConv
@testset "FFT-based Convolutional Kernel" begin
    work_dim = 16
    kernel_dim = work_dim
    x = vcat(1, zeros(work_dim - 1))
    kernel = randn(work_dim)
    # Without padding
    # Without phase
    cconv = CConv(Float64, work_dim)
    y = cconv(x, kernel; phase = 0)
    @test isapprox(y, kernel)
    # With phase compensation
    @testset "Phase compensation: phase=$phase " for phase in [-3, -2, 0, 2, 3]
        cconv = CConv(Float64, work_dim)
        y = cconv(x, circshift(kernel, phase); phase)
        @test isapprox(y, kernel)
    end
    # With padding it is more tricky to check phase compensation since signals the output is cropped
    # Without phase compensation
    work_dim = 19
    kernel_dim = 5
    x = vcat(1, zeros(work_dim - 2), 0)
    kernel = vcat(ones(kernel_dim), zeros(work_dim - kernel_dim))
    padding = nextpow(2, work_dim) - work_dim
    phase = 0
    cconv = CConv(Float64, work_dim; padding)
    y = cconv(x, kernel; phase)
    @test isapprox(y, kernel)
    # With phase compensation and padding
    @testset "Phase compensation and padding" for kernel_dim in [4, 5], work_dim in [19, 18]
        x = vcat(1, zeros(work_dim - 2), 0)
        kernel = vcat(ones(kernel_dim), zeros(work_dim - kernel_dim))
        phase = default_phase_kernel(kernel_dim)
        cconv = CConv(Float64, work_dim; padding)
        y = cconv(x, kernel; phase)
        @test isapprox(y[1:(kernel_dim - phase)], kernel[1:(kernel_dim - phase)])
        @test all(isapprox.(y[(kernel_dim - phase + 1):end], 0, atol = 1.0e-6))
    end

    @testset "Skipping Kernel Load" begin
        work_dim = 16
        kernel_dim = work_dim
        x = vcat(1, zeros(work_dim - 1))
        kernel = randn(work_dim)
        # Without padding
        # Without phase
        cconv = CConv(Float64, work_dim)
        y = cconv(x, kernel; phase = 0)
        y_bis = cconv(x, randn(work_dim); load_kernel = false, phase = 0) # Dont update kernel
        kernel_bis = randn(work_dim)
        y_bis_2 = cconv(x, kernel_bis, phase = 0) # Update kernel
        @test isapprox(y, kernel)
        @test isapprox(y_bis, kernel)
        @test isapprox(y_bis_2, kernel_bis)
    end
end

@testset "GMWFrame" for wave_dim in [8192, 8191]
    b = 1
    g = 3
    J = 6
    Q = 2
    wmin = 0
    wmax = pi
    sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim)
    gmw = GMWFrame(sp)
    @test sp.frame === gmw
    freeframe!(sp)
    @test sp.frame == nothing
    setframe!(sp)
    @test sp.frame !== gmw
    # Test Unit Power
    power = sum(x -> norm(x)^2, gmw.frame)
    @test isapprox(power, 1)
    # Test Flat spectrum
    spec = sum(map(x -> abs2.(fft(x)), gmw.frame))
    target = ones(wave_dim)
    @test isapprox(spec, target)

    # Analytic case
    sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim; analytic = true)
    gmw = GMWFrame(sp)
    # Test Unit Power
    power = sum(x -> norm(x)^2, gmw.frame)
    @test isapprox(power, 1)
    # Test Flat spectrum
    spec = sum(map(x -> abs2.(fft(x)), gmw.frame))
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
    cconv = CConv(Float64, work_dim)
    x_fft = [cconv(x, g) for g in gmw.frame]
    # Test Unit Power
    power = sum(x -> norm(x)^2, x_fft)
    @test isapprox(power, 1)
    # Test Flat spectrum
    spec = sum(map(x -> abs2.(fft(x)), x_fft))
    target = ones(wave_dim)
    @test isapprox(target, spec)
end

@testset "Averaging Kernels" begin
    kernel_types = [GaussAvg, RectAvg, ScaleAvg]
    sum_to_one(x::Vector{<:Real}) = isapprox(sum(x), 1)
    sum_to_one(x::Vector{<:Vector{<:Real}}) = all(sum_to_one.(x))
    default_params(kernel_type::Type{GaussAvg}, kernel_dim::Integer) = GaussAvg(kernel_dim, kernel_dim / 10)
    default_params(kernel_type::Type{RectAvg}, kernel_dim::Integer) = RectAvg(kernel_dim, kernel_dim / 10)
    default_params(kernel_type::Type{ScaleAvg}, kernel_dim::Integer) = ScaleAvg(kernel_dim, [GaussAvg(kernel_dim, kernel_dim / 10), GaussAvg(kernel_dim, kernel_dim / 20)])
    work_dim = 8192
    kernel_dim = 8192
    kernel_type = kernel_types[1]
    x = randn(work_dim)
    y = randn(work_dim)
    xy = x .* y
    avg_kernel = default_params(kernel_type, kernel_dim)
    tp = TimeParams(avg_kernel)
    xy_avg = average(xy, tp)
    # xy_dtavg = dt_average(xy, tp) TODO
    # xy_dpavg = dp_average(xy, tp)

    @testset "Kernel: $kernel_type" for kernel_type in kernel_types,
            kernel_dim in [32, 33]
        avg_kernel = default_params(kernel_type, kernel_dim)
        tp = TimeParams(avg_kernel)
        avg_kernel = averaging_kernel(tp)
        @test sum_to_one(avg_kernel)
    end
    @testset "Average func $(kernel_type)" for kernel_type in filter(!=(ScaleAvg), kernel_types) , kernel_dim in [16, 15]
        work_dim = 64
        avg_kernel = default_params(kernel_type, kernel_dim)
        # Center dirac at phase of the averaging_kernel
        x = circshift(vcat(1, zeros(work_dim - 1)), default_phase_kernel(kernel_dim))
        # No padding
        tp = TimeParams(avg_kernel)
        x_avg = average(x, tp)
        @test isapprox(x_avg[1:kernel_dim], averaging_kernel(tp))
        # With padding
        tp = TimeParams(
            avg_kernel,
            padding = work_dim - kernel_dim,
        )
        x_avg = average(x, tp)
        @test isapprox(x_avg[1:kernel_dim], averaging_kernel(tp))
        x_avgs = average([x, circshift(x, 1)], tp)
        @test isapprox(x_avgs[1][1:kernel_dim], averaging_kernel(tp))
        @test isapprox(circshift(x_avgs[2], -1)[1:kernel_dim], averaging_kernel(tp))
    end
end

@testset "Cross-Scalogram test" for work_dim in [8192, 8191]
    function get_setup(work_dim, wave_dim, kernel_dim)
        b = 1
        g = 3
        J = 6
        Q = 2
        wmin = 0
        wmax = pi
        x1 = circshift(vcat(1, zeros(work_dim - 1)), div(work_dim, 2)) # Center the signal
        x2 = circshift(x1, 1)
        x3 = x2 + x1
        avg_kernel = GaussAvg(kernel_dim, kernel_dim / 10)
        sp = ScaleParams(b, g, J, Q, wmin, wmax, wave_dim)
        tp = TimeParams(avg_kernel)
        dp = DecompParams(sp, tp)
        gmw = GMWFrame(dp)
        avg_kernel = averaging_kernel(dp)
        L = Dict(:x1 => x1, :x2 => x2, :x3 => x3)
        return L, gmw, avg_kernel, dp
    end
    work_dim = 8192 # padding is equal to 0 when size is pow of 2
    kernel_dim = wave_dim = work_dim
    L, gmw, avg_kernel, dp = get_setup(work_dim, wave_dim, kernel_dim)
    out = cross_scalogram(L[:x1], L[:x1], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 1) # The flux is one between x1 and x1
    out = cross_scalogram(L[:x1], L[:x2], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 0, atol = 1.0e-6) # The flux is zero between x1 and x2
    out = cross_scalogram(L[:x1], L[:x3], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 1) # The flux is one between x1 and x3
    # Check that we obtain the same with or without next2pow padding
    work_dim = 7777
    kernel_dim = wave_dim = work_dim
    L, gmw, avg_kernel, dp = get_setup(work_dim, wave_dim, kernel_dim)
    x1, x2, x3 = L
    out = cross_scalogram(L[:x1], L[:x1], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 1) # The flux is one between x1 and x1
    out = cross_scalogram(L[:x1], L[:x2], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 0, atol = 1.0e-6) # The flux is zero between x1 and x2
    out = cross_scalogram(L[:x1], L[:x3], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 1) # The flux is one between x1 and x3
    # An with reduced kernel sizes
    L, gmw, avg_kernel, dp = get_setup(work_dim, div(wave_dim, 2), div(kernel_dim, 3))
    x1, x2, x3 = L
    out = cross_scalogram(L[:x1], L[:x1], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 1) # The flux is one between x1 and x1
    out = cross_scalogram(L[:x1], L[:x2], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 0, atol = 1.0e-6) # The flux is zero between x1 and x2
    out = cross_scalogram(L[:x1], L[:x3], dp, gmw, avg_kernel)
    @test isapprox(sum(out), 1) # The flux is one between x1 and x3

    # Multiple Signals
    C = Dict((:x1, :x2) => :x12, (:x1, :x3) => :x13, (:x2, :x3) => :x23, (:x2, :x1) => :x21)
    out = cross_scalogram(L, C, dp, gmw, avg_kernel)
    @test isapprox(sum(out[:x12]), 0, atol = 1.0e-6)
    @test isapprox(sum(out[:x13]), 1)
    @test isapprox(sum(out[:x23]), 1)
    @test isapprox(out[:x12], out[:x21])
end
