# PINNs for Classical PDEs

## Project structure

```text
pinns-classical-pdes/
├── README.md
├── requirements.txt
├── .gitignore
├── heat/
│   ├── heat_mlp.py
│   └── heat_resnet.py
├── wave/
│   └── wave_mlp.py
├── oscillator/
│   └── damped_oscillator.py
└── results/
```

## Setup

```powershell
cd pinns-classical-pdes
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python heat/heat_mlp.py
python heat/heat_resnet.py
python wave/wave_mlp.py
python oscillator/damped_oscillator.py
```

Plots are saved in `results/`.

## Model notes

- The wave example is configured for a **physics-only PINN**. Set `USE_ANALYTICAL_DATA = True` in `wave/wave_mlp.py` to compare against the analytical solution at all grid points and create a data + physics hybrid.
- The oscillator example uses 500 collocation points, 20,000 Adam epochs, and L-BFGS refinement. The initial-condition loss has a 10× weight.
- The oscillator residual heatmap is labeled as an ODE residual, not a PDE residual.
