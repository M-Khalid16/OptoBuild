# Architecture Decision Records

Each major decision is recorded as an ADR (context → decision → alternatives →
consequences). ADRs are immutable once *Accepted*; a changed decision gets a
new ADR that *supersedes* the old one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-signal-representation.md) | Signal representation | Accepted |
| [0002](0002-fft-convention.md) | FFT and spectral-density convention | Accepted |
| [0003](0003-physical-units.md) | Physical unit strategy | Accepted |
| [0004](0004-component-api.md) | Component API | Accepted |
| [0005](0005-graph-execution.md) | Graph execution architecture | Accepted |
| [0006](0006-project-persistence.md) | Project persistence format | Accepted |
| [0007](0007-numerical-backend.md) | Numerical backend strategy | Accepted |
| [0008](0008-reproducible-randomness.md) | Reproducible randomness | Accepted |
| [0009](0009-construction-validation-and-results.md) | Construction-time validation and recorded results (amends 0004) | Accepted |
| [0010](0010-sampling-grid-and-bit-timing.md) | Per-source sampling grids and bit-timing metadata | Partly superseded by 0011 |
| [0011](0011-global-simulation-layout.md) | Global simulation layout (project schema v2) | Accepted |
| [0012](0012-stochastic-components-and-trials.md) | Stochastic components and Monte Carlo trials | Accepted |
| [0013](0013-gui-architecture.md) | GUI architecture (Qt-free models + Qt views) | Accepted |
| [0014](0014-ssfm.md) | Split-step Fourier solver for the NLSE | Accepted |
| [0015](0015-fso-channel.md) | Free-space optical channel model | Accepted |
| [0016](0016-coherent-single-pol.md) | Single-polarization coherent transmission and DSP | Accepted |
| [0017](0017-dual-polarization-coherent.md) | Dual-polarization coherent systems, MIMO equalization, PMD | Accepted |
| [0018](0018-photonic-circuits.md) | Photonic circuits: S-matrix solver and device components | Accepted |
| [0019](0019-laser-models.md) | Laser and amplifier models, fixed-step integration | Accepted |
| [0020](0020-ultrafast-and-cavities.md) | Ultrafast propagation (GNLSE/RK4IP) and cavity iteration | Accepted |

Template: [template.md](template.md).
