using StatsBase
using PyPlot
using PyCall
pycolors=pyimport("matplotlib.colors")
using Unitful
using UnitfulLatexify
using Latexify
label_time="time (h)"
label_momentum = latexify(u"m^2/s^2")
label_energy_flux = latexify(u"W/m^2")
label_speed = latexify(u"m/s")
label_co2 = latexify(u"μmol/mol")
label_h2o= latexify(u"mmol/mol")
label_T_C = latexify(u"°C")
label_T_rawflux=latexify(u"K * m / s")
label_co2_rawflux = latexify(u"μmol/mol * m / s")
label_h2o_rawflux= latexify(u"mmol/mol * m / s")
label_freq = latexify(u"Hz")
label_η = "Log Normalized Scale"

hist_cmap="viridis"
diverging_0inf ="Reds"
norm_0inf(min=0.)=pycolors.Normalize(vmin=min)
diverging_mpinf ="RdYlGn_r"
norm_mpinf(c=0.)=pycolors.TwoSlopeNorm(c)

function plot_speeds(Signals)
  f=figure()
  plot(time_h[sampling_1min],Signals.U[sampling_1min],label="U")
  plot(time_h[sampling_1min],Signals.V[sampling_1min],label="V")
  plot(time_h[sampling_1min],Signals.W[sampling_1min],label="W")
  xlabel(label_time)
  ylabel(label_speed)
  legend();
  return f
end

function plot_scalars(Signals)
  fig, ax1 = subplots()
  color = "tab:green"
  ax1.set_xlabel(label_time)
  ax1.set_ylabel(label_co2, color=color)
  ax1.plot(time_h[sampling_1min],Signals.CO2[sampling_1min], color=color)
  ax1.tick_params(axis="y", labelcolor=color)

  ax2 = ax1.twinx()

  color = "tab:blue"
  ax2.set_ylabel(label_h2o, color=color)
  ax2.plot(time_h[sampling_1min], Signals.H2O[sampling_1min], color=color)
  ax2.tick_params(axis="y", labelcolor=color)

  ax3 = ax1.twinx()

  color = "tab:red"
  ax3.set_ylabel(label_T_C, color=color)
  ax3.plot(time_h[sampling_1min], Signals.T[sampling_1min], color=color)
  ax3.tick_params(axis="y", labelcolor=color)
  return fig
end

function plot_contour_symlog(x,y,z,linthresh;title_str="",label_x=label_time,label_y=label_freq,label_z=label_momentum,cmap=diverging_mpinf,norm=norm_mpinf())
  fig=plt.figure()
  ax=plt.axes()
  c=ax.contourf(x,y,z,cmap=cmap,norm=norm)
  ax.set_yscale("symlog",linthresh=linthresh)
  ax.set_xlabel(label_x)
  ax.set_ylabel(label_y)
  fig.colorbar(c,label=label_z)
  fig.suptitle(title_str) 
  return fig
end

function plot_contour(ax,x,y,z;title_str="",label_x=label_time,label_y=label_freq,label_z=label_momentum,cmap=diverging_mpinf,norm=norm_mpinf())
  c=ax.contourf(x,y,z,cmap=cmap,norm=norm)
  ax.set_yscale("log")
  ax.set_xlabel(label_x)
  ax.set_ylabel(label_y)
  return (ax,c)
end

function plot_contour(x,y,z;title_str="",label_x=label_time,label_y=label_freq,label_z=label_momentum,cmap=diverging_mpinf,norm=norm_mpinf())
  fig=plt.figure()
  ax=plt.axes()
  c=ax.contourf(x,y,z,cmap=cmap,norm=norm)
  ax.set_yscale("log")
  ax.set_xlabel(label_x)
  ax.set_ylabel(label_y)
  fig.colorbar(c,label=label_z)
  fig.suptitle(title_str) 
  return fig
end

py"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

def get_grid_day_flux():
  fig = plt.figure()
  gs0 = gs.GridSpec(3, 1, figure=fig,height_ratios=[0.1,0.6,0.3],hspace=0)
  ax3 = fig.add_subplot(gs0[2, 0])
  ax2 = fig.add_subplot(gs0[1, 0],sharex=ax3)
  ax1 = fig.add_subplot(gs0[0, 0],sharex=ax3)
  return (fig, gs0, ax1, ax2,ax3)
"""

function mask_colormap(alpha=0.5)
  gcmap=plt.cm.gray.resampled(2)
  newcolors=gcmap.([0,1])
  newcolors[1]=(0,0,0,alpha)
  return pycolors.ListedColormap(newcolors)
end

function plot_histogram(x;title_str="",label_x="",nbins=nothing)
  med_x=median(x)
  mean_x=mean(x)
  if isnothing(nbins)
    h=fit(Histogram,view(x,:))
  else
    h=fit(Histogram,view(x,:),nbins=nbins)
  end
  fig=plt.figure()
  ax=plt.axes()
  ax.stairs(h.weights,edges=h.edges[1],fill=true,alpha=0.65)
  ax.axvline(med_x,linestyle="dashed",color="red",label="median")
  ax.axvline(mean_x,color="red",label="mean")
  ax.set_xlabel(label_x)
  ax.legend()
  fig.suptitle("Histogram" * title_str)
  return fig
end

function plot_2dhistogram(x,y;title_str,label_x,label_y,nbins=100)
  if isnothing(nbins)
    h=fit(Histogram,view(x,:))
  else
    h=fit(Histogram,(x,y),nbins=nbins)
  end
  fig=plt.figure()
  ax=plt.axes()
  ax.pcolormesh(h.edges[2],h.edges[1],h.weights,cmap=hist_cmap)
  ax.set_ylabel(label_x)
  ax.set_xlabel(label_y)
  fig.suptitle("2D Histogram" * title_str)
  return fig
end

function plot_tricontourf(ax,X,Y,Z;label_x,label_y,title_str="",cmap=diverging_mpinf,norm=norm_mpinf(),alphamask=nothing)
  tric=ax.tricontourf(X,Y,Z,cmap=cmap,norm=norm)
  if !isnothing(alphamask)
    ax.tricontourf(X,Y,alphamask,cmap=mask_colormap(0.1))
  end
  ax.set_xlabel(label_x)
  ax.set_ylabel(label_y)
  return (ax,tric)
end

function _plot_flux(ax,X,flux,EP_X,EP_flux;label_flux="")
  ax.plot(X,flux,label=label_flux)
  ax.scatter(EP_X,EP_flux,label="EddyPro UNCORR",color="k")
  return ax
end

function plot_day_flux(X,Y,Z,time_h,flux,SW_IN,time_h_EP,EP_flux,mask=nothing)
  (fig,gs,ax1,ax2,ax3) = py"get_grid_day_flux()"
  tric=_tricontour_dist(ax2,X,Y,Z,mask=mask)
  ax2.set_ylabel("Log Normalized Frequency")# \eta=\frac{\overline{u}(t)z}{\xi}")
  _plot_flux(ax3,time_h,flux,time_h_EP,EP_flux)
  ax3.set_ylabel(L"Flux")
  ax2.xaxis.set_tick_params(labelbottom=false)
  ax1.plot(time_h,SW_IN)
  ax1.set_ylabel("SW_IN")
  ax3.set_xlabel("Time (h)")
  return fig
end

function _save_fig(plot_path,fig,fname_fig)
  cd(plot_path) do
    fig.savefig(fname_fig)
  end
  PyPlot.close_figs()
  nothing
end
