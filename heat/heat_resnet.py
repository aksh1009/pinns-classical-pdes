from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_X = 100
N_T = 100
X_MIN = -1.0
X_MAX = 1.0
T_MAX = 4.0
ALPHA = 0.01


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        residual = inputs
        out = self.activation(self.linear1(inputs))
        out = self.linear2(out) + residual
        return self.activation(out)


class ResidualNet(nn.Module):
    def __init__(self, input_dim=2, width=24, output_dim=1, n_blocks=6):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, width)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(width) for _ in range(n_blocks)]
        )
        self.output_layer = nn.Linear(width, output_dim)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        out = self.activation(self.input_layer(inputs))
        for block in self.res_blocks:
            out = block(out)
        return self.output_layer(out)


x_1d = torch.linspace(X_MIN, X_MAX, N_X, device=device)
t_1d = torch.linspace(0.0, T_MAX, N_T, device=device)
x, t = torch.meshgrid(x_1d, t_1d, indexing="ij")
x = x.reshape(-1, 1).requires_grad_(True)
t = t.reshape(-1, 1).requires_grad_(True)

model = ResidualNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=500, factor=0.5
)


def compute_loss(model, x, t, alpha):
    u_pred = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    mse_pde = torch.mean((u_t - alpha * u_xx) ** 2)

    t_bc = torch.linspace(0.0, T_MAX, N_T, device=device).unsqueeze(1)
    x_bc_0 = torch.full_like(t_bc, X_MIN)
    x_bc_1 = torch.full_like(t_bc, X_MAX)
    u_bc_0 = model(torch.cat([x_bc_0, t_bc], dim=1))
    u_bc_1 = model(torch.cat([x_bc_1, t_bc], dim=1))
    mse_bc = torch.mean(u_bc_0**2) + torch.mean(u_bc_1**2)

    x_ic = torch.linspace(X_MIN, X_MAX, N_X, device=device).unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)
    u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
    u_ic_true = torch.exp(-20.0 * x_ic**2)
    mse_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
    return mse_pde + mse_bc + mse_ic


for epoch in range(2_000):
    optimizer.zero_grad()
    loss = compute_loss(model, x, t, ALPHA)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6e}")

x_eval = torch.linspace(X_MIN, X_MAX, 256, device=device).unsqueeze(1)
t_eval = torch.linspace(0.0, T_MAX, 100, device=device).unsqueeze(1)
x_eval, t_eval = torch.meshgrid(
    x_eval.squeeze(), t_eval.squeeze(), indexing="ij"
)
xt_eval = torch.cat([
    x_eval.reshape(-1, 1),
    t_eval.reshape(-1, 1),
], dim=1)
u_pred = model(xt_eval).detach().cpu().numpy().reshape(256, 100)

results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(10, 6.5), constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.15])

heatmap_ax = fig.add_subplot(grid[0, :])
image = heatmap_ax.imshow(
    u_pred.T,
    extent=[0.0, T_MAX, X_MIN, X_MAX],
    origin="lower",
    aspect="auto",
    cmap="inferno",
)
heatmap_ax.set_xlabel("t")
heatmap_ax.set_ylabel("x")
heatmap_ax.set_title("ResidualNet PINN Solution to Heat Equation")
colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.025, pad=0.02)
colorbar.set_label("u(t,x)")

x_values = np.linspace(X_MIN, X_MAX, 256)
for column, snapshot_time in enumerate((1.0, 2.0, 3.0)):
    profile_ax = fig.add_subplot(grid[1, column])
    snapshot_index = int(round(snapshot_time / T_MAX * (N_T - 1)))
    profile_ax.plot(x_values, u_pred[:, snapshot_index], linewidth=1.5)
    profile_ax.set_title(f"t={int(snapshot_time)}")
    profile_ax.set_xlabel("x")
    profile_ax.set_ylabel("u(t,x)")
    profile_ax.set_xlim(X_MIN, X_MAX)

plot_path = results_dir / "heat_resnet.png"
plt.savefig(plot_path, dpi=150)
print(f"Saved plot: {plot_path}")
plt.show()
