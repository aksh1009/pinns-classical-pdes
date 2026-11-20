from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
omega_0 = 1.0
zeta = 0.1
x0 = 1.0
v0 = 0.0


class DampedOscillatorPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, t):
        return self.net(t)


model = DampedOscillatorPINN().to(device)


def derivatives(t):
    t = t.requires_grad_(True)
    x = model(t)
    dx = autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    ddx = autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    return x, dx, ddx


def loss_components(t_collocation):
    x, dx, ddx = derivatives(t_collocation)
    residual = ddx + 2 * zeta * omega_0 * dx + omega_0**2 * x
    loss_pde = torch.mean(residual**2)

    t0 = torch.tensor([[0.0]], device=device, requires_grad=True)
    x0_pred = model(t0)
    dx0_pred = autograd.grad(
        x0_pred, t0, torch.ones_like(x0_pred), create_graph=True
    )[0]
    loss_ic = (x0_pred - x0).pow(2).mean() + (dx0_pred - v0).pow(2).mean()
    return loss_pde, loss_ic


t_collocation = torch.linspace(0, 10, 500, device=device).reshape(-1, 1)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20_000):
    optimizer_adam.zero_grad()
    loss_pde, loss_ic = loss_components(t_collocation)
    loss = loss_pde + 10.0 * loss_ic
    loss.backward()
    optimizer_adam.step()
    if epoch % 500 == 0:
        print(
            f"Epoch {epoch}: Total={loss.item():.6e}, "
            f"PDE={loss_pde.item():.6e}, IC={loss_ic.item():.6e}"
        )

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=500,
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
)


def lbfgs_closure():
    optimizer_lbfgs.zero_grad()
    loss_pde, loss_ic = loss_components(t_collocation)
    loss = loss_pde + 10.0 * loss_ic
    loss.backward()
    return loss


lbfgs_loss = optimizer_lbfgs.step(lbfgs_closure)
print(f"L-BFGS final loss: {lbfgs_loss.item():.6e}")

t_test = torch.linspace(0, 10, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    x_pred = model(t_test).cpu().numpy().flatten()

t_np = t_test.cpu().numpy().flatten()
omega_d = omega_0 * np.sqrt(1 - zeta**2)
x_exact = np.exp(-zeta * omega_0 * t_np) * (
    x0 * np.cos(omega_d * t_np)
    + (v0 + zeta * omega_0 * x0) / omega_d * np.sin(omega_d * t_np)
)

residual_t = t_test.detach().clone().requires_grad_(True)
x, dx, ddx = derivatives(residual_t)
residual = (
    ddx + 2 * zeta * omega_0 * dx + omega_0**2 * x
).detach().cpu().numpy().flatten() ** 2

results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 4))
plt.plot(t_np, x_pred, label="PINN Prediction")
plt.plot(t_np, x_exact, "--", label="Analytical Solution")
plt.xlabel("Time t")
plt.ylabel("Displacement x(t)")
plt.legend()
plt.title("Damped Harmonic Oscillator - PINN vs Analytical")
solution_path = results_dir / "damped_oscillator_solution.png"
plt.savefig(solution_path, dpi=150)
print(f"Saved plot: {solution_path}")
plt.show()

plt.figure(figsize=(8, 3))
plt.imshow(
    residual[np.newaxis, :],
    aspect="auto",
    cmap="hot",
    extent=[0, 10, 0, 1],
    origin="lower",
)
plt.colorbar(label="Residual squared")
plt.yticks([])
plt.xlabel("Time t")
plt.title("ODE Residual (Squared) Over Time")
residual_path = results_dir / "damped_oscillator_residual.png"
plt.savefig(residual_path, dpi=150)
print(f"Saved plot: {residual_path}")
plt.show()
