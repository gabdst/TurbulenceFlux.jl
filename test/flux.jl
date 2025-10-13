using TurbulenceFlux

function create_df(work_dim)
    df = Dict()
    df[:TIMESTAMP] = collect(1:work_dim)
    t = 0:(work_dim-1)
    f0 = 1 / 4
    x = cos.(2pi * f0 * t)
    # x = circshift(vcat(100, zeros(work_dim - 1)), default_phase_kernel(work_dim))
    df[:U] = randn(work_dim) * 3
    df[:V] = randn(work_dim) * 2
    df[:W] = randn(work_dim) .+ x
    df[:TA] = ones(work_dim) .- 273.15
    df[:PA] = ones(work_dim) * TurbulenceFlux.R / 1000
    df[:CO2] = randn(work_dim) .+ x
    df[:H2O] = (x .+ randn(work_dim)) / TurbulenceFlux.LAMBDA
    df
end

work_dim = 8192

fs = 20
z_d = 10.0
aux = AuxVars(; fs, z_d)
cp = CorrectionParams()
deleteat!(cp.corrections, findfirst(==(:despiking), cp.corrections))

df = create_df(work_dim)
kernel_dim = wave_dim = work_dim
kernel_type = :gaussian
kernel_params = [1024]
tp = TimeParams(kernel_dim, kernel_type, kernel_params; padding = kernel_dim)
tp_aux = tp
method = ReynoldsEstimation(; tp, tp_aux)
results = estimate_flux(; df, aux, cp, method)

@test isapprox(results.estimate.RHO, ones(work_dim))

@test isapprox(
    results.estimate.FC[.!results.estimate.FC_QC],
    ones(count(.!results.estimate.FC_QC)) / 2,
    rtol = 0.1,
)


@test isapprox(
    results.estimate.LE[.!results.estimate.LE_QC],
    ones(count(.!results.estimate.LE_QC)) / 2,
    rtol = 0.1,
)

@test isapprox(
    results.estimate.H[.!results.estimate.H_QC],
    zeros(count(.!results.estimate.H_QC)),
    atol = 0.05,
)
