import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

mat_data = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat_data['V1'].flatten()
y_full = mat_data['V2'].flatten()

start_idx = 42650
end_idx = 46746
u_excerpt = u_full[start_idx:end_idx]
y_excerpt = y_full[start_idx:end_idx]

u_mean = np.mean(u_excerpt)
y_mean = np.mean(y_excerpt)
u_centered = u_excerpt - u_mean
y_centered = y_excerpt - y_mean

fs = 610.35
dt = 1.0 / fs

n_train = 3072
n_test = 1024

u_train = u_centered[:n_train]
y_train = y_centered[:n_train]
u_test = u_centered[n_train:]
y_test = y_centered[n_train:]

v_train = np.zeros_like(y_train)
v_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2 * dt)
v_train[0] = (y_train[1] - y_train[0]) / dt
v_train[-1] = (y_train[-1] - y_train[-2]) / dt

y_t = torch.tensor(y_train, dtype=torch.float32)
v_t = torch.tensor(v_train, dtype=torch.float32)
u_t = torch.tensor(u_train, dtype=torch.float32)

# Architecture 2: 2 hidden layers of size 32 with SiLU, fully vectorized batch training over sliding windows
class AccelerationMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
    def forward(self, y, v, u):
        x = torch.cat([y, v, u], dim=-1)
        return self.net(x).squeeze(-1)

model = AccelerationMLP2()

def rk4_step(y, v, u_curr, u_next, dt, model):
    acc1 = model(y.unsqueeze(-1), v.unsqueeze(-1), u_curr.unsqueeze(-1))
    dy1 = v
    dv1 = acc1
    
    y2 = y + 0.5 * dt * dy1
    v2 = v + 0.5 * dt * dv1
    u2 = 0.5 * (u_curr + u_next)
    acc2 = model(y2.unsqueeze(-1), v2.unsqueeze(-1), u2.unsqueeze(-1))
    dy2 = v2
    dv2 = acc2
    
    y3 = y + 0.5 * dt * dy2
    v3 = v + 0.5 * dt * dv2
    u3 = 0.5 * (u_curr + u_next)
    acc3 = model(y3.unsqueeze(-1), v3.unsqueeze(-1), u3.unsqueeze(-1))
    dy3 = v3
    dv3 = acc3
    
    y4 = y + dt * dy3
    v4 = v + dt * dv3
    u4 = u_next
    acc4 = model(y4.unsqueeze(-1), v4.unsqueeze(-1), u4.unsqueeze(-1))
    dy4 = v4
    dv4 = acc4
    
    y_next = y + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
    v_next = v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
    
    return y_next, v_next

optimizer = optim.Adam(model.parameters(), lr=1e-3)
rollout_len = 20
n_steps = 1000
batch_size = 32

max_start_idx = n_train - rollout_len - 1

for step in range(n_steps):
    optimizer.zero_grad()
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    loss = 0.0
    for idx in start_indices:
        y_curr = y_t[idx]
        v_curr = v_t[idx]
        
        pred_y_list = [y_curr]
        for k in range(rollout_len):
            t_idx = idx + k
            u_curr = u_t[t_idx]
            u_next = u_t[t_idx + 1]
            y_curr, v_curr = rk4_step(y_curr, v_curr, u_curr, u_next, dt, model)
            pred_y_list.append(y_curr)
            
        pred_y_tensor = torch.stack(pred_y_list)
        true_y_tensor = y_t[idx : idx + rollout_len + 1]
        
        loss = loss + torch.mean((pred_y_tensor - true_y_tensor) ** 2)
        
    loss = loss / batch_size
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 250 == 0:
        print(f"Step {step+1}/{n_steps}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    full_u = torch.tensor(u_centered, dtype=torch.float32)
    full_y_pred = np.zeros(len(u_centered))
    
    y_curr = torch.tensor(y_centered[0], dtype=torch.float32)
    v_curr = torch.tensor(v_train[0], dtype=torch.float32)
    full_y_pred[0] = y_curr.item()
    
    for i in range(len(u_centered) - 1):
        u_curr = full_u[i]
        u_next = full_u[i+1]
        y_curr, v_curr = rk4_step(y_curr, v_curr, u_curr, u_next, dt, model)
        full_y_pred[i+1] = y_curr.item()

train_pred = full_y_pred[:n_train]
test_pred = full_y_pred[n_train:]

train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))

print(f"RESULT mlp_32_silu_v2 TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")

plt.figure(figsize=(10, 4))
time_axis = np.arange(n_test) * dt
plt.plot(time_axis, y_test, label='Measured $y$', alpha=0.8)
plt.plot(time_axis, test_pred, label='Predicted $y$ (Free Run)', alpha=0.8, linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Silverbox PINN Free Run on Test Segment (RMSE: {test_rmse:.4f} V)')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()

print(f"TEST_RMSE: {test_rmse:.6f}")
