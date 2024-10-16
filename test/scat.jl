N=8192
x=vcat(1,zeros(N-1))
β=2
γ=3
J=floor(Int,log2(N))
Q=4
wmin=2pi/N
wmax=pi
gmw_params=TurbulenceFlux.gmw_grid(β,γ,J,Q,wmin,wmax)
gmw_frame = TurbulenceFlux.GMWFrame(N,gmw_params)
σ_waves = gmw_frame.σ_waves
σ_min = minimum(σ_waves)
averaging_kernel = [ TurbulenceFlux._gausskernel(N,10*σ/N,0) for σ in σ_waves ]
averaging_kernel = [ σ_min]

Δt=1

out=TurbulenceFlux.cross_scattering(x,x,Δt,gmw_frame.gmw_frame,averaging_kernel)
