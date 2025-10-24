# TurbulenceFlux.jl  [![DOI](https://zenodo.org/badge/733581341.svg)](https://doi.org/10.5281/zenodo.15310755)

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)


TurbulenceFlux.jl is a Julia package for high‑resolution, fixed‑point turbulence analysis and flux estimation. Although originally developed for flux‑tower measurements above forest ecosystems, the package can be applied to any similar context. Development is ongoing, and new estimation methods will be added, but the core features have been implemented.

## Usage

See the detailed example in `example.jl`, it also includes a link to download the data samples. Below is a brief overview of the package’s structure and functionality. Most functions and types have extensive documentation accessible from the REPL via help mode (e.g., `?estimate_flux`).

### Main entry point

The function `estimate_flux` is the main entry point and is intended for operational flux estimation.
```
function estimate_flux(
    df::Dict,
    aux::AuxVars,
    cp::CorrectionParams,
    method::FluxEstimationMethod,
)
```
It expects:
 - `df`: dictionnary containing the input signals (mandatory variables plus some optional gas variables)
 - `aux`: auxilliary variables  needed for the calculation
 - `cp`: correction parameters
 - `method`: the flux-estimation method to apply

For the naming convention used please check `mandatory_variables` and `gas_variables` constants.

It returns a `FluxEstimate` object:
```
struct FluxEstimate{T<:FluxEstimationMethod}
    estimate::NamedTuple
    qc::QualityControl
    cp::CorrectionParams
    method::T
    units::NamedTuple
end
```
, which holds the estimated fluxes, quality-control information, updated correction parameters, and metadata (method settings and units).

### Estimation Methods

Currently, three different estimation methods (`method::FluxEstimationMethod`) are available:
- `ReynoldsEstimation`: Standard eddy‑covariance method using Reynolds decomposition
- `TurbuThreshold`: Local turbulent transport extraction via wavelet analysis and thresholding of the Reynolds stress tensor
- `TurbuLaplacian`: Similar to TurbuThreshold but with Laplacian analysis of the Reynolds stress tensor

Each method provides a specific set of outputs. See `?TurbuThreshold` for example.

Standard inputs and outputs follow the FLUXNET variable naming convention.

### Parameter structures

 - `TimeParams`: controls temporal averaging and subsampling. Used by all methods; for `ReynoldsEstimation` it defines the two frequency bands (mean vs. variable).
 - `ScaleParams`: configures the wavelet‑based approaches (`TurbuThreshold` and `TurbuLaplacian`), by specifying the number of frequency bands (wavelet basis size) and the wavelet shape.

For more information, see the help `?TimeParams`, `?ScaleParams`, and the method‑specific docs (e.g., `?TurbuThreshold`).

## Advanced Usage

Cross-correlations between signals (i.e., fluxes) are computed using domain-specific methods. `cross_scalogram` is used for wavelet-based approaches while the standard eddy-covariance approach use `cross_correlation_rey`.

## Contributing

Contributions are welcome ! Because the package targets a specialized research field —turbulence analysis and flux estimation— please feel free to open an issue or contact the maintainers if you have ideas for improvements or need specific modifications.
