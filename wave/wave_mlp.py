from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_X = 100
N_T = 100
C = 1.0
USE_ANALYTICAL_DATA = False


class PiNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, inputs):
        for layer in self.layers[:-1]:
            inputs = torch.tanh(layer(inputs))
        return self.layers[-1](inputs)


def compute_pde_residual(u, x, t, c):
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    return u_tt - c**2 * u_xx


def compute_loss(model, x, t, u_true, c):
    u_pred = model(torch.cat([x, t], dim=1))
    mse_pde = torch.mean(compute_pde_residual(u_pred, x, t, c) ** 2)

    x_ic = torch.linspace(0, 1, N_X, device=device).unsqueeze(1)
    t_ic = torch.zeros_like(x_ic, requires_grad=True)
    u_ic = model(torch.cat([x_ic, t_ic], dim=1))
    u_t_ic = torch.autograd.grad(
        u_ic, t_ic, torch.ones_like(u_ic), create_graph=True
    )[0]
    mse_ic = torch.mean((u_ic - torch.sin(np.pi * x_ic)) ** 2)
    mse_ic += torch.mean(u_t_ic**2)

    t_bc = torch.linspace(0, 1, N_T, device=device).unsqueeze(1)
    u_bc_0 = model(torch.cat([torch.zeros_like(t_bc), t_bc], dim=1))
    u_bc_1 = model(torch.cat([torch.ones_like(t_bc), t_bc], dim=1))
    mse_bc = torch.mean(u_bc_0**2) + torch.mean(u_bc_1**2)

    loss = 10.0 * mse_pde + 10.0 * mse_ic + 10.0 * mse_bc
    if USE_ANALYTICAL_DATA:
        loss = loss + torch.mean((u_pred - u_true) ** 2)
    return loss


x_1d = torch.linspace(0, 1, N_X, device=device)
t_1d = torch.linspace(0, 1, N_T, device=device)
x, t = torch.meshgrid(x_1d, t_1d, indexing="ij")
x = x.reshape(-1, 1).detach().requires_grad_(True)
t = t.reshape(-1, 1).detach().requires_grad_(True)
u_true = (torch.sin(np.pi * x) * torch.cos(np.pi * C * t)).detach()

model = PiNN([2, 64, 64, 64, 64, 64, 1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=500, factor=0.5
)

for epoch in range(10_000):
    optimizer.zero_grad()
    loss = compute_loss(model, x, t, u_true, C)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6e}")

x_eval = torch.linspace(0, 1, 256, device=device).unsqueeze(1)
t_eval = torch.linspace(0, 1, 100, device=device).unsqueeze(1)
x_eval, t_eval = torch.meshgrid(
    x_eval.squeeze(), t_eval.squeeze(), indexing="ij"
)
x_eval = x_eval.reshape(-1, 1)
t_eval = t_eval.reshape(-1, 1)

with torch.no_grad():
    u_pred = model(torch.cat([x_eval, t_eval], dim=1)).cpu().numpy()

u_pred = u_pred.reshape(256, 100)
u_true_eval = (
    torch.sin(np.pi * x_eval) * torch.cos(np.pi * C * t_eval)
).cpu().numpy()
u_true_eval = u_true_eval.reshape(256, 100)

results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(
    u_pred.T, aspect="auto", extent=[0, 1, 0, 1],
    origin="lower", cmap="viridis"
)
plt.colorbar(label="u")
plt.title("PINN Prediction")
plt.xlabel("x")
plt.ylabel("t")

plt.subplot(122)
plt.imshow(
    u_true_eval.T, aspect="auto", extent=[0, 1, 0, 1],
    origin="lower", cmap="viridis"
)
plt.colorbar(label="u")
plt.title("Analytical Solution")
plt.xlabel("x")
plt.ylabel("t")

plt.tight_layout()
plot_path = results_dir / "wave_mlp.png"
plt.savefig(plot_path, dpi=150)
print(f"Saved plot: {plot_path}")
plt.show()

mse = np.mean((u_pred - u_true_eval) ** 2)
print(f"Mean Squared Error: {mse:.6e}")
