# GIST (Geomechanical Injection Scenario Toolkit)

Plan and forecast injection scenarios with a comprehensive toolkit for analyzing geomechanical impacts.

## Getting Started

### Prerequisites

- **Python:** 3.9.10
- **Core Dependencies:** `numpy` (1.21.4), `scipy` (1.7.3), `pandas` (1.3.5)
- **Plotting Dependencies:** `seaborn`, `matplotlib`, `geopandas`, `contextily`

### Installation & Setup

It is recommended to use a Python virtual environment to manage dependencies:

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Project Structure

The code is structured as modular steps that can be run independently or as part of a pipeline.

- `src/gistStepCore.py`: Core computational logic.
- `src/gistStep1.py` to `src/gistStep5.py`: Individual workflow steps.
- `src/gistMC.py`: GIST class implementation.
- `src/TexNetWebToolGPWrappers/`: Helper utilities for portal integration.
- `gold/`: Reference CSV files for regression testing.

## Usage

GIST can be run via Jupyter notebooks or via driver scripts.

- **Notebooks:** See `GIST_RunTemplate_*.ipynb` for example workflows.
- **Driver Scripts:** `gistStepsDriver.py` provides a CLI interface for running discrete steps.

## Disclaimer

GIST aims to give the gist of a wide range of potential scenarios and aid collective decision making when responding to seismicity.
The results of GIST are entirely dependent upon the inputs provided, which may be incomplete or inaccurate.
There are other potentially plausible inducement scenarios that are not considered, including fluid migration into the basement, out-of-zone poroelastic stressing, or hydraulic fracturing.
None of the individual models produced by GIST accurately represent what happens in the subsurface and cannot be credibly used to accurately assign liability or responsibility for seismicity.

*"All models are wrong, but some are useful"* — George Box, 1976