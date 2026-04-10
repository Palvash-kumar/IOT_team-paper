# Improving Distance-Extension Methods for Reliable Wireless Communication in IoT Networks with UAV

This repository contains the implementation and report assets for an academic IoT term paper on UAV-assisted wireless communication range extension. The project evaluates multiple physical-layer techniques and introduces a Genetic Algorithm (GA)-based adaptive power optimization strategy for improving communication reliability and energy efficiency.

## Project Summary

UAVs can act as aerial base stations to improve coverage for IoT deployments, but long-distance communication quality drops due to path loss and fading. This work studies and compares:

- Baseline fixed-power communication
- Adaptive power control
- Directional antenna gain
- Three-tier adaptive modulation (QPSK, 8-PSK, 16-QAM)
- Forward Error Correction (FEC) coding gain
- Combined optimizations
- Proposed GA-based adaptive power profile optimization

The simulation framework also includes multi-altitude analysis and Rayleigh fading experiments.

## Repository Contents

- `uav_iot_simulation.py`: Main simulation and visualization script.
- `term_paper.tex`: LaTeX source code for the team paper report.
- `README.md`: Project documentation.
- `fig1_baseline_analysis.png` to `fig12_summary_dashboard.png`: Generated publication-quality figures.
- `Improving_the_Distance-Extension_Methods_for_Reliable_Wireless_Communication_in_IoT_Networks_with_UAV.pdf`: Full paper PDF.
- `Team_Paper.pdf`: Team paper report (compiled PDF submission).

## Key Technical Highlights

- Log-distance path loss with shadow fading.
- Optional Rayleigh fading channel model.
- BER modeling for QPSK, 8-PSK, and 16-QAM.
- Energy-aware GA fitness formulation balancing coverage and transmit power.
- High-quality figure generation suitable for academic reporting.

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
- save all figures as PNG files at 250 DPI,
- print a concise performance summary in the terminal.

## Main Outputs

Running the simulation produces:

- Link budget and BER analyses
- SNR and received power comparison plots
- Adaptive modulation and throughput analysis
- GA convergence and power-profile optimization plots
- Altitude trade-off and Rayleigh fading impact plots
- Final summary dashboard

## Reproducibility Notes

- The implementation uses deterministic random seeds in key stochastic components for consistent experimental behavior.
- Simulation parameters are centralized in the `SystemParameters` class inside `uav_iot_simulation.py`.
- To test alternative scenarios, modify parameters such as altitude, path-loss exponent, power limits, and GA settings.

## Paper Build (LaTeX)

To compile the team paper locally, use a LaTeX toolchain (for example, TeX Live or MiKTeX) and run:

```bash
pdflatex term_paper.tex
```

If your local setup requires bibliography or multiple passes, run the corresponding LaTeX build sequence.

## Authors

- Avasu Palvash Kumar
- Mood Narendar

Department of Computer Science and Engineering  
Indian Institute of Information Technology, Sri City, Chittoor

## License

This project is intended for academic and educational use.