using Test
using LinearAlgebra
using Statistics
using Rotations
import TurbulenceFlux: optim_timelag, planar_fit, ErrorRotationAmbiguous

@testset "Timelag Optimization" begin
    N = 1024
    fs = 1
    make_lag(N, tlag) = begin
        x = vcat(1, zeros(N - 1))
        y = circshift(x, -tlag)
        return x, y
    end
    @testset "Size Parity and Timelag check" for N in [1024, 1023],
        tlag in [-5, -4, 0, 4, 5]

        x, y = make_lag(N, tlag)
        fc = 0
        tau, out, found_lag = optim_timelag(x, y, fc, fs)
        @test found_lag == tlag
    end
    @testset "With Noise contamination" for N in [1024, 1023], tlag in [-5, -4, 0, 4, 5]
        t = (0:(N-1))
        fc = 0.2
        f0 = 0.1 / fs # noise below cutting frequency
        e = cos.(2pi * f0 * t)
        x, y = make_lag(N, tlag)
        x = x + e
        tau, out, found_lag = optim_timelag(x, y, fc, fs)
        @test found_lag == tlag
    end
end

@testset "Planar Fit" begin
    rand_angles() = (rand(3) .- 0.5) * pi / 2
    function make_rotation(N, pitch, roll, yaw, sigma)
        X0 = randn(N, 3) .* sigma
        X = X0 * RotYXZ(pitch, roll, yaw)'
        return X, X0
    end
    check_norm(X0, X_rot) = @test isapprox(norm(X0, 2), norm(X_rot, 2))
    check_z_coord(X0, X_rot) = @test isapprox(X0[:, 3], X_rot[:, 3]; rtol = 0.1)
    check_coord(X0, X_rot) = @test isapprox(X0, X_rot; rtol = 0.1)
    @testset "Rotation" begin
        N = 1000
        angles = [-pi / 4, 0, pi / 4]
        sigma = [10 1 0.1]
        @testset "pitch only: $pitch $roll $yaw" for (pitch, roll, yaw) in
                                                     zip(angles, zeros(3), zeros(3))
            X, X0 = make_rotation(N, yaw, pitch, roll, sigma)
            X_rot, _ = planar_fit(X)
            check_norm(X0, X_rot)
            check_z_coord(X0, X_rot)
            # check_coord(X0, X_rot)
        end
        @testset "roll only: $pitch $roll $yaw" for (pitch, roll, yaw) in
                                                    zip(zeros(3), angles, zeros(3))
            X, X0 = make_rotation(N, yaw, pitch, roll, sigma)
            X_rot, _ = planar_fit(X)
            check_norm(X0, X_rot)
            check_z_coord(X0, X_rot)
            # check_coord(X0, X_rot)
        end
        @testset "yaw only: $pitch $roll $yaw" for (pitch, roll, yaw) in
                                                   zip(zeros(3), zeros(3), angles)
            X, X0 = make_rotation(N, yaw, pitch, roll, sigma)
            X_rot, _ = planar_fit(X)
            check_norm(X0, X_rot)
            check_z_coord(X0, X_rot)
            # check_coord(X0, X_rot)
        end
        @testset "pitch-roll only: $pitch $roll $yaw" for ((pitch, roll), yaw) in zip(
            Iterators.product(angles, angles),
            zeros(3),
        )
            X, X0 = make_rotation(N, yaw, pitch, roll, sigma)
            X_rot, _ = planar_fit(X)
            check_norm(X0, X_rot)
            check_z_coord(X0, X_rot)
            # check_coord(X0, X_rot)
        end
        # Check Exceptions
        @testset "Rotation ambiguity" for sigma in ([1 1 1], [10 1 1], [1 10 1])
            yaw, pitch, roll = rand_angles()
            X, X0 = make_rotation(N, yaw, pitch, roll, sigma)
            @test_throws ErrorRotationAmbiguous planar_fit(X)
        end
        @test_throws ArgumentError planar_fit(zeros(10, 3), 0.9)
    end
end
