using TurbulenceFlux

function create_df(work_dim)
    df = Dict()
    df[:TIMESTAMP] = collect(1:work_dim)
    x = circshift(vcat(100, zeros(work_dim - 1)), default_phase_kernel(work_dim))
    df[:U] = randn(work_dim) * 3
    df[:V] = randn(work_dim) * 2
    df[:W] = randn(work_dim)
    df[:TA] = ones(work_dim) .- 273.15
    df[:PA] = ones(work_dim) * TurbulenceFlux.R / 1000
    df[:CO2] = df[:W] + randn(work_dim)
    df[:H2O] = (df[:W] + randn(work_dim)) / TurbulenceFlux.LAMBDA
    df
end

work_dim = 8192

fs = 20
z_d = 10.0
aux = AuxVars(; fs, z_d)
cp = CorrectionParams()
deleteat!(cp.corrections, findfirst(==(:despiking), cp.corrections))

kernel_dim = wave_dim = work_dim
kernel_type = :gaussian
kernel_params = [fs * 100 / 3]
tp = TimeParams(kernel_dim, kernel_type, kernel_params; padding = kernel_dim)
tp_aux = tp
df = create_df(work_dim)
method = ReynoldsEstimation(; tp, tp_aux)
results = estimate_flux(; df, aux, cp, method)
