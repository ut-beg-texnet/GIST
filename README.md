# GIST (Geomechanical Injection Scenario Toolkit)

Plan and forecast injection scenarios with a comprehensive toolkit for analyzing geomechanical impacts.

## Getting Started

### Prerequisites

- **Python:** 3.14.8
- **Core Dependencies:** `numpy` (2.3.5), `scipy` (1.16.3), `pandas` (2.3.3), `matplotlib` (3.10.9)
- **Notebook-only Dependencies:** `seaborn`, `geopandas`, `contextily` (see `requirements-notebooks.txt`)

### Dependency files

| File | Purpose |
|---|---|
| `requirements.txt` | Direct runtime/test dependencies, exact pins. Edit this when changing a dependency. |
| `requirements-lock.txt` | Full `pip freeze` of the working venv, including transitive dependencies. Install from this file for a reproducible environment. |
| `requirements-notebooks.txt` | Optional extras used only by the Jupyter notebooks. |

### Installation & Setup (development and production)

The venv is named `gist_python_14_venv` and lives at the repo root (it is git-ignored).

```powershell
# 1. Install Python 3.14.8 (python.org Windows x64 installer); confirm with:
py -3.14 --version

# 2. Create the virtual environment
py -3.14 -m venv gist_python_14_venv

# 3. Install the locked dependencies
.\gist_python_14_venv\Scripts\python.exe -m pip install --upgrade pip
.\gist_python_14_venv\Scripts\python.exe -m pip install -r requirements-lock.txt

# 4. (Optional) notebook extras
.\gist_python_14_venv\Scripts\python.exe -m pip install -r requirements-lock.txt -r requirements-notebooks.txt

# 5. Sanity check
.\gist_python_14_venv\Scripts\python.exe -c "import sys, numpy, pandas, scipy, matplotlib; print(sys.version, numpy.__version__, pandas.__version__, scipy.__version__, matplotlib.__version__)"
cd src; ..\gist_python_14_venv\Scripts\python.exe -m pytest tests -q
```

To upgrade a dependency: change its pin in `requirements.txt`, reinstall into a fresh venv,
run the tests, then regenerate the lock with
`.\gist_python_14_venv\Scripts\python.exe -m pip freeze > requirements-lock.txt`.

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