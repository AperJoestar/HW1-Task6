import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json

torch.manual_seed(123)
np.random.seed(123)

mat = scipy.io.loadmat('SNLS80mV.mat')
V1 = mat['V1'].flatten()
V2 = mat['V2'].flatten()

slice_idx = slice(42650, 46746)
u_full = V1[slice_idx]
y_full = V2[slice_idx]

u = u_full - np.mean(u_full)
y = y_full - np.mean(y_full)

fs = 610.35
dt = 1.0 / fs
n_train = 3072
n_test = 1024

v = np.zeros_like(y)
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt

u_t = torch.tensor(u, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
v_t = torch.tensor(v, dtype=torch.float32)

y_scale = 0.06
v_scale = 22.0
u_scale = 0.022
acc_scale = 10000.0

class SilverboxPINN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, y_val, v_val, u_val):
        ys = y_val / y_scale
        vs = v_val / v_scale
        us = u_val / u_scale
        inp = torch.stack([ys, vs, us], dim=-1)
        acc_scaled = self.net(inp).squeeze(-1)
        return acc_scaled * acc_scale

model = SilverboxPINN(hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=2e-3)

def get_u(t_idx_float):
    idx_lo = torch.floor(t_idx_float).long().clamp(0, len(u_t)-2)
    idx_hi = idx_lo + 1
    frac = t_idx_float - idx_lo.float()
    u_lo = u_t[idx_lo]
    u_hi = u_t[idx_hi]
    return u_lo + frac * (u_hi - u_lo)

def rk4_step(y_curr, v_curr, t_idx):
    u0 = u_t[t_idx]
    a0 = model(y_curr, v_curr, u0)
    
    y1 = y_curr + 0.5 * dt * v_curr
    v1 = v_curr + 0.5 * dt * a0
    u1 = get_u(t_idx.float() + 0.5)
    a1 = model(y1, v1, u1)
    
    y2 = y_curr + 0.5 * dt * v1
    v2 = v_curr + 0.5 * dt * a1
    u2 = u1
    a2 = model(y2, v2, u2)
    
    y3 = y_curr + dt * v2
    v3 = v_curr + dt * a2
    u3 = u_t[(t_idx + 1).clamp(0, len(u_t)-1)]
    a3 = model(y3, v3, u3)
    
    y_next = y_curr + (dt / 6.0) * (v_curr + 2*v1 + 2*v2 + v3)
    v_next = v_curr + (dt / 6.0) * (a0 + 2*a1 + 2*a2 + a3)
    return y_next, v_next

# Fast training parameters
epochs = 400
rollout_len = 64
batch_size = 64

print("Starting training of tuned model...")
for epoch in range(epochs):
    max_start = n_train - rollout_len - 1
    start_indices = torch.randint(0, max_start, (batch_size,))
    
    y_batch = y_t[start_indices]
    v_batch = v_t[start_indices]
    
    loss = 0.0
    for step_idx in range(rollout_len):
        t_indices = start_indices + step_idx
        y_target = y_t[t_indices + 1]
        v_target = v_t[t_indices + 1]
        
        y_batch, v_batch = rk4_step(y_batch, v_batch, t_indices)
        loss = loss + torch.mean((y_batch - y_target)**2) + 0.05 * torch.mean((v_batch - v_target)**2)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

def free_run(start_idx, end_idx):
    y_pred = np.zeros(end_idx - start_idx)
    v_pred = np.zeros(end_idx - start_idx)
    
    curr_y = y_t[start_idx]
    curr_v = v_t[start_idx]
    
    y_pred[0] = curr_y.item()
    v_pred[0] = curr_v.item()
    
    with torch.no_grad():
        for i in range(end_idx - start_idx - 1):
            t_idx = start_idx + i
            curr_y, curr_v = rk4_step(curr_y, curr_v, torch.tensor(t_idx))
            y_pred[i+1] = curr_y.item()
            v_pred[i+1] = curr_v.item()
            
    return y_pred, v_pred

y_full_pred, _ = free_run(0, len(y))
y_train_pred = y_full_pred[:n_train]
y_test_pred = y_full_pred[n_train:]

train_rmse = np.sqrt(np.mean((y_train_pred - y[:n_train])**2))
test_rmse = np.sqrt(np.mean((y_test_pred - y[n_train:])**2))

print(f"RESULT mlp_tuned TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

plt.figure(figsize=(10, 4))
time_axis = np.arange(n_test) * dt
plt.plot(time_axis, y[n_train:], label='Measured y', alpha=0.8)
plt.plot(time_axis, y_test_pred, label='Predicted y (Free Run)', alpha=0.8, linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Silverbox Test Segment Free Run (RMSE: {test_rmse:.5f} V)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()

results = {
    "name": "mlp_tuned",
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse)
}
with open('results_mlp_tuned.json', 'w') as f:
    json.dump(results, f)
