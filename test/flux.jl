using TurbulenceFlux

work_dim = 8192
fs = 20
z_d = 10.0
aux = AuxVars(; fs, z_d)
kernel_dim = wave_dim = work_dim

cp = CorrectionParams()
empty!(cp.corrections)

df = Dict()
df[:TIMESTAMP] = collect(1:work_dim)
t = 0:(work_dim - 1)
f0 = 1 / 4
x = cos.(2pi * f0 * t)
df[:U] = 3x
df[:V] = 2x
df[:W] = x
df[:TA] = ones(work_dim) .- 273.15 + x
df[:PA] = ones(work_dim) * TurbulenceFlux.R / 1000
df[:CO2] = x
df[:H2O] = x / TurbulenceFlux.LAMBDA

dt = 1
avg_kernel = GaussAvg(kernel_dim, 1024)
tp = TimeParams(avg_kernel; padding = kernel_dim, dt)
tp_aux = tp

method = ReynoldsEstimation(; tp, tp_aux)
results = estimate_flux(; df, aux, cp, method)

@test isapprox(results.estimate.RHO, ones(work_dim), rtol = 0.05)
@test isapprox(
    results.estimate.FC[.!results.estimate.FC_QC],
    ones(count(.!results.estimate.FC_QC)) / 2,
    rtol = 0.05,
)
@test isapprox(
    results.estimate.LE[.!results.estimate.LE_QC],
    ones(count(.!results.estimate.LE_QC)) / 2,
    rtol = 0.05,
)
@test isapprox(
    results.estimate.H[.!results.estimate.H_QC] ./ TurbulenceFlux.C_p,
    ones(count(.!results.estimate.H_QC)) / 2,
    rtol = 0.05,
)

@test isinteger(method_timestep(method, work_dim))

wave_dim = work_dim
b = 1
g = 3
J = floor(Int, log2(wave_dim))
Q = 2
fmin = 4pi / wave_dim
fmax = pi
sp = ScaleParams(b, g, J, Q, fmin, fmax, wave_dim)

dt = 1
tp = TimeParams(avg_kernel; padding = kernel_dim, dt)
tp_aux = tp

dp = DecompParams(sp, tp)

tr_tau = 0
method = TurbuThreshold(; dp, tp_aux, tr_tau)
results = estimate_flux(; df, aux, cp, method)

@test isapprox(results.estimate.RHO, ones(work_dim), rtol = 0.05)
@test isapprox(
    results.estimate.FC[.!results.estimate.FC_QC],
    ones(count(.!results.estimate.FC_QC)) / 2,
    rtol = 0.05,
)
@test isapprox(
    results.estimate.LE[.!results.estimate.LE_QC],
    ones(count(.!results.estimate.LE_QC)) / 2,
    rtol = 0.05,
)
@test isapprox(
    results.estimate.H[.!results.estimate.H_QC] ./ TurbulenceFlux.C_p,
    ones(count(.!results.estimate.H_QC)) / 2,
    rtol = 0.05,
)

@test isinteger(method_timestep(method, work_dim, fmin))
