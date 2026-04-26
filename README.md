# Improving Distance-Extension Methods for Reliable Wireless Communication in IoT Networks with UAV (Enhanced Version)

This repository contains the implementation and report assets for an academic IoT term paper on UAV-assisted wireless communication range extension. The project evaluates multiple physical-layer techniques and introduces a Genetic Algorithm (GA)-based adaptive power optimization strategy for improving communication reliability and energy efficiency.

## Enhanced Version Features

This enhanced version adds the following over the original:

- **Rician fading channel model** (K=6, LoS-dominant) alongside existing Rayleigh fading
- **Formal energy efficiency metric** (EE = distance / power, m/mW)
- **UAV altitude optimization results table** with altitude-dependent path-loss exponents
- **Expanded 3×3 summary dashboard** with fading, EE, and throughput panels
- **Discussion section** analyzing GA trade-offs, fading implications, and altitude optimization
- **Throughput analysis** under three-tier adaptive modulation

## Project Summary

UAVs can act as aerial base stations to improve coverage for IoT deployments, but long-distance communication quality drops due to path loss and fading. This work studies and compares:

- Baseline fixed-power communication
- Adaptive power control
- Directional antenna gain
- Three-tier adaptive modulation (QPSK, 8-PSK, 16-QAM)
- Forward Error Correction (FEC) coding gain
- Combined optimizations
- Proposed GA-based adaptive power profile optimization

The simulation framework includes multi-altitude analysis, Rayleigh and Rician fading experiments, formal energy efficiency analysis, and throughput analysis.

## Repository Contents

- `uav_iot_simulation.py`: Main simulation and visualization script (enhanced version).
- `term_paper.tex`: LaTeX source code for the team paper report (4-page IEEE format).
- `README.md`: Project documentation.
- `fig1_baseline_analysis.png` to `fig12_summary_dashboard.png`: Generated publication-quality figures.
- `Team_Paper.pdf`: Team paper report (compiled PDF submission).

## Key Technical Highlights

- Log-distance path loss with shadow fading.
- Rayleigh fading (NLoS) and Rician fading (LoS-dominant, K=6).
- BER modeling for QPSK, 8-PSK, and 16-QAM.
- Energy-aware GA fitness formulation balancing coverage and transmit power.
- Formal energy efficiency metric (m/mW) for technique comparison.
- Multi-altitude analysis with altitude-dependent path-loss exponents.
- Expanded 3×3 summary dashboard with 9 panels.

## Environment and Dependencies

Recommended:

- Python 3.9+
- pip

Python packages:

- numpy
- matplotlib
- scipy

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

## How to Run

From the project root:

```bash
python uav_iot_simulation.py
```

The script will:

- execute all simulation scenarios,
- display plots,
- save all 12 figures as PNG files at 250 DPI,
- print performance summary with energy efficiency metrics in the terminal.

## Main Outputs

Running the simulation produces:

- Link budget and BER analyses
- SNR and received power comparison plots
- Adaptive modulation and throughput analysis
- GA convergence and power-profile optimization plots
- Altitude trade-off analysis
- Fading channel comparison (AWGN vs Rayleigh vs Rician)
- Energy efficiency analysis with formal EE metric
- Enhanced 3×3 summary dashboard

## Reproducibility Notes

- The implementation uses deterministic random seeds in key stochastic components.
- Simulation parameters are centralized in the `SystemParameters` class.
- To test alternative scenarios, modify parameters such as altitude, path-loss exponent, power limits, and GA settings.

## Paper Build (LaTeX)

To compile the team paper locally:

```bash
pdflatex term_paper.tex
```

The paper is designed to fit exactly 4 pages in IEEE conference format.

## Authors

- Avasu Palvash Kumar
- Mood Narendar

Department of Computer Science and Engineering  
Indian Institute of Information Technology, Sri City, Chittoor

## License

This project is intended for academic and educational use.