import json
import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

mat_path = 'SNLS80mV.mat'
if not os.path.exists(mat_path):
    mat_path = '../SNLS80mV.mat'
if not os.path.exists(mat_path):
    mat_path = r'C:\Users\А\Downloads\task 6\SNLS80mV.mat'

mat = scipy.io.loadmat(mat_path)
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

start_idx = 42650
excerpt_len = 4096
u_exc = u_full[start_idx : start_idx + excerpt_len]
y_exc = y_full[start_idx : start_idx + excerpt_len]

u_mean = np.mean(u_exc)
y_mean = np.mean(y_exc)

u = u_exc - u_mean
y = y_exc - y_mean

fs = 610.35
dt = 1.0 / fs

train_len = 3072
test_len = 1024

y_train = y[:train_len]
u_train = u[:train_len]
v_train = np.gradient(y_train, dt)

y_scale = 0.055
v_scale = 21.5
u_scale = 0.022
acc_scale = 8.7e3

class AccelerationMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )
    
    def forward(self, y, v, u):
        inp = torch.stack([y / y_scale, v / v_scale, u / u_scale], dim=-1)
        acc = self.net(inp).squeeze(-1) * (acc_scale / 1000.0)
        return acc

device = torch.device('cpu')
model = AccelerationMLP2().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def interpolate_u(u_arr, t_steps):
    idx_low = torch.floor(t_steps).long().clamp(0, len(u_arr) - 1)
    idx_high = (idx_low + 1).clamp(0, len(u_arr) - 1)
    frac = t_steps - idx_low.float()
    u_t = torch.tensor(u_arr, dtype=torch.float32)
    return u_t[idx_low] * (1.0 - frac) + u_t[idx_high] * frac

def rk4_step(y_t, v_t, t_idx, dt, u_arr):
    u0 = interpolate_u(u_arr, t_idx)
    u1 = interpolate_u(u_arr, t_idx + 0.5)
    u2 = interpolate_u(u_arr, t_idx + 1.0)
    
    k1_y = v_t
    k1_v = model(y_t, v_t, u0)
    
    y2 = y_t + 0.5 * dt * k1_y
    v2 = v_t + 0.5 * dt * k1_v
    k2_y = v2
    k2_v = model(y2, v2, u1)
    
    y3 = y_t + 0.5 * dt * k2_y
    v3 = v_t + 0.5 * dt * k2_v
    k3_y = v3
    k3_v = model(y3, v3, u1)
    
    y4 = y_t + dt * k3_y
    v4 = v_t + dt * k3_v
    k4_y = v4
    k4_v = model(y4, v4, u2)
    
    y_next = y_t + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_next = v_t + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return y_next, v_next

rollout_len = 16
n_steps = 1000
batch_size = 64

y_tensor = torch.tensor(y_train, dtype=torch.float32)
v_tensor = torch.tensor(v_train, dtype=torch.float32)

model.train()
for step in range(n_steps):
    max_idx = len(y_train) - rollout_len - 1
    start_indices = torch.randint(0, max_idx, (batch_size,))
    
    y_b = y_tensor[start_indices]
    v_b = v_tensor[start_indices]
    
    loss = 0.0
    y_curr = y_b
    v_curr = v_b
    
    for r in range(rollout_len):
        t_indices = start_indices.float() + r
        y_next, v_next = rk4_step(y_curr, v_curr, t_indices, dt, u_train)
        target_y = y_tensor[start_indices + r + 1]
        loss = loss + torch.mean((y_next - target_y) ** 2)
        y_curr, v_curr = y_next, v_next
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 250 == 0:
        print(f"Step {step+1}/{n_steps}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred_full = np.zeros(excerpt_len)
    y_curr = torch.tensor([y[0]], dtype=torch.float32)
    v_curr = torch.tensor([v_train[0]], dtype=torch.float32)
    y_pred_full[0] = y_curr.item()
    
    for i in range(excerpt_len - 1):
        y_next, v_next = rk4_step(y_curr, v_curr, torch.tensor([float(i)]), dt, u)
        y_pred_full[i+1] = y_next.item()
        y_curr, v_curr = y_next, v_next

train_rmse = float(np.sqrt(np.mean((y_pred_full[:train_len] - y[:train_len])**2)))
test_rmse = float(np.sqrt(np.mean((y_pred_full[train_len:] - y[train_len:])**2)))

print(f"RESULT arch2 TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")

with open("results_arch2.json", "w") as f:
    json.dump({"train_rmse": train_rmse, "test_rmse": test_rmse}, f)

# Also generate test_plot.png as required for the kept version or final submission
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
t_axis = np.arange(excerpt_len) * dt
plt.plot(t_axis[train_len:], y[train_len:], label='Measured y', color='black', alpha=0.7)
plt.plot(t_axis[train_len:], y_pred_full[train_len:], label='Predicted y (Free Run)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Silverbox Test Segment Free Run Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png', dpi=300)
print("TEST_RMSE:", test_rmse)
