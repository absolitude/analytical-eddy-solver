# Analytical Eddy-Current Solver

**Analytical eddy-current solvers for thin-coated accelerator vacuum chambers.**

This project provides two computational tools for studying eddy currents induced in thin conductive coatings of accelerator vacuum chambers.

## Main Features

- **Green's-function boundary solver:** Supports flexible chamber boundary geometries.
- **Fourier-based analytical solver:** Provides analytical solutions for selected geometries.
- **Time-domain calculations:** Calculates induced surface currents and magnetic fields.

## Available Solvers

### 1. Green Solver

The Green Solver uses a boundary integral formulation to calculate eddy currents on thin conductive chamber walls. It supports flexible boundary geometries, including user-defined geometries.

**Source code:** `Green_v1_2/`

#### Installation

**Option 1: Direct installation**

```powershell
cd Green_v1_2
python -m pip install -r requirements.txt
python eddy_current_v1_2.py
```

**Option 2: Using a virtual environment (recommended)**

```powershell
cd Green_v1_2
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python eddy_current_v1_2.py
```

### 2. Fourier Solver

The Fourier Solver uses a Fourier-based formulation to calculate eddy currents for selected chamber geometries.

**Source code:** `Fourier_v1_0/`

#### Installation

**Option 1: Direct installation**

```powershell
cd Fourier_v1_0
python -m pip install -r requirements.txt
python main.py
```

**Option 2: Using a virtual environment (recommended)**

```powershell
cd Fourier_v1_0
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

## Download

Pre-built Windows executables are available in the [GitHub Releases](https://github.com/absolitude/analytical-eddy-solver/releases) section.

Download the ZIP package for the desired solver, extract the entire folder, and run the executable. Keep the `_internal` folder together with the executable.

## Documentation

See the [User Manual](eddy_current_user_manual_en.pdf) for theoretical background, numerical methods, and usage instructions.