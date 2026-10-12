import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import time

# 1. Load and preprocess data
mat = scipy.io.loadmat('SNLS80mV.mat')
V1 = mat['V1'].flatten()
V2 = mat['V2'].flatten()

slice_data = slice(42650, 46746)
u_full = V1[slice_data]
y_full = V2[slice_data]

u_mean = np.mean(u_full)
y_mean = np.mean(y_full)

u = u_full - u_mean
y = y_full - y_mean

N_train = 3072
N_test = 1024
dt = 1.0 / 610.35

v = np.zeros_like(y)
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt

y_scale = 0.06
v_scale = 22.0
u_scale = 0.02
acc_scale = 10000.0

device = torch.device('cpu')

u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
v_tensor = torch.tensor(v, dtype=torch.float32, device=device)

class DeepAccNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 48),
            nn.GELU(),
            nn.Linear(48, 48),
            nn.GELU(),
            nn.Linear(48, 1)
        )
    def forward(self, y_s, v_s, u_s):
        x = torch.stack([y_s, v_s, u_s], dim=-1)
        return self.net(x).squeeze(-1) * acc_scale

model = DeepAccNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

L_rollout = 64
batch_size = 32
epochs = 400

def interpolate_u(u_seq, t_indices):
    idx_floor = torch.floor(t_indices).long()
    idx_ceil = torch.clamp(idx_floor + 1, 0, len(u_seq) - 1)
    idx_floor = torch.clamp(idx_floor, 0, len(u_seq) - 1)
    frac = t_indices - idx_floor.float()
    return u_seq[idx_floor] * (1.0 - frac) + u_seq[idx_ceil] * frac

print("Starting training of deep model...")
for epoch in range(epochs):
    start_indices = torch.randint(0, N_train - L_rollout, (batch_size,))
    
    y_batch = torch.stack([y_tensor[si:si+L_rollout] for si in start_indices])
    v_batch = torch.stack([v_tensor[si:si+L_rollout] for si in start_indices])
    
    y_curr = y_batch[:, 0]
    v_curr = v_batch[:, 0]
    
    y_preds = [y_curr]
    
    for step in range(L_rollout - 1):
        t_base = start_indices.float() + step
        
        u_k1 = interpolate_u(u_tensor, t_base)
        a_k1 = model(y_curr / y_scale, v_curr / v_scale, u_k1)
        dy1, dv1 = v_curr, a_k1
        
        u_k2 = interpolate_u(u_tensor, t_base + 0.5)
        y_k2 = y_curr + 0.5 * dt * dy1
        v_k2 = v_curr + 0.5 * dt * dv1
        a_k2 = model(y_k2 / y_scale, v_k2 / v_scale, u_k2)
        dy2, dv2 = v_k2, a_k2
        
        u_k3 = interpolate_u(u_tensor, t_base + 0.5)
        y_k3 = y_curr + 0.5 * dt * dy2
        v_k3 = v_curr + 0.5 * dt * dv2
        a_k3 = model(y_k3 / y_scale, v_k3 / v_scale, u_k3)
        dy3, dv3 = v_k3, a_k3
        
        u_k4 = interpolate_u(u_tensor, t_base + 1.0)
        y_k4 = y_curr + dt * dy3
        v_k4 = v_curr + dt * dv3
        a_k4 = model(y_k4 / y_scale, v_k4 / v_scale, u_k4)
        dy4, dv4 = v_k4, a_k4
        
        y_next = y_curr + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        v_next = v_curr + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        y_curr = y_next
        v_curr = v_next
        y_preds.append(y_curr)
        
    y_preds = torch.stack(y_preds, dim=1)
    loss = torch.mean((y_preds - y_batch) ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_free = np.zeros_like(y)
    v_free = np.zeros_like(v)
    y_free[0] = y[0]
    v_free[0] = v[0]
    
    y_curr = torch.tensor(y[0], dtype=torch.float32, device=device)
    v_curr = torch.tensor(v[0], dtype=torch.float32, device=device)
    
    for i in range(len(y) - 1):
        t_val = torch.tensor(float(i), dtype=torch.float32, device=device)
        u_k1 = interpolate_u(u_tensor, t_val)
        a_k1 = model(y_curr / y_scale, v_curr / v_scale, u_k1)
        dy1, dv1 = v_curr, a_k1
        
        u_k2 = interpolate_u(u_tensor, t_val + 0.5)
        y_k2 = y_curr + 0.5 * dt * dy1
        v_k2 = v_curr + 0.5 * dt * dv1
        a_k2 = model(y_k2 / y_scale, v_k2 / v_scale, u_k2)
        dy2, dv2 = v_k2, a_k2
        
        u_k3 = interpolate_u(u_tensor, t_val + 0.5)
        y_k3 = y_curr + 0.5 * dt * dy2
        v_k3 = v_curr + 0.5 * dt * dv2
        a_k3 = model(y_k3 / y_scale, v_k3 / v_scale, u_k3)
        dy3, dv3 = v_k3, a_k3
        
        u_k4 = interpolate_u(u_tensor, t_val + 1.0)
        y_k4 = y_curr + dt * dy3
        v_k4 = v_curr + dt * dv3
        a_k4 = model(y_k4 / y_scale, v_k4 / v_scale, u_k4)
        dy4, dv4 = v_k4, a_k4
        
        y_curr = y_curr + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        v_curr = v_curr + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        y_free[i+1] = y_curr.item()
        v_free[i+1] = v_curr.item()

train_rmse = np.sqrt(np.mean((y_free[:N_train] - y[:N_train]) ** 2))
test_rmse = np.sqrt(np.mean((y_free[N_train:] - y[N_train:]) ** 2))

print(f"RESULT mlp_deep TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

plt.figure(figsize=(10, 4))
time_axis = np.arange(N_test) * dt
plt.plot(time_axis, y[N_train:], label='Measured y', color='black', alpha=0.7)
plt.plot(time_axis, y_free[N_train:], label='Free-run Predicted y', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Silverbox Test Segment Free Run (Test RMSE: {test_rmse:.5f} V)')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()

results = {
    "model": "mlp_deep",
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse)
}
with open('results_mlp_deep.json', 'w') as f:
    json.dump(results, f, indent=2)
