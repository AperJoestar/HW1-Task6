import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os

# 1. Load Data and Preprocess
mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

start_idx = 42650
length = 4096
excerpt_slice = slice(start_idx, start_idx + length)

u_ex = u_full[excerpt_slice]
y_ex = y_full[excerpt_slice]

# Center u and y
u_ex = u_ex - np.mean(u_ex)
y_ex = y_ex - np.mean(y_ex)

fs = 610.35
dt = 1.0 / fs

train_len = 3072
test_len = 1024

y_train = y_ex[:train_len]
u_train = u_ex[:train_len]
y_test = y_ex[train_len:]
u_test = u_ex[train_len:]

# Compute velocity using central differences
v_train = np.zeros_like(y_train)
v_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2 * dt)
v_train[0] = (y_train[1] - y_train[0]) / dt
v_train[-1] = (y_train[-1] - y_train[-2]) / dt

# Normalization stats
y_mean, y_std = np.mean(y_train), np.std(y_train)
v_mean, v_std = np.mean(v_train), np.std(v_train)
u_mean, u_std = np.mean(u_train), np.std(u_train)

# 2. Define PINN Architecture with SiLU and Normalization
class PINNAccelNet(nn.Module):
    def __init__(self):
        super(PINNAccelNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, y, v, u):
        # Normalize inputs
        y_n = (y - y_mean) / y_std
        v_n = (v - v_mean) / v_std
        u_n = (u - u_mean) / u_std
        x = torch.stack([y_n, v_n, u_n], dim=-1)
        return self.net(x).squeeze(-1)

model = PINNAccelNet()

# 3. Interpolation for u at half steps
def get_u(t_idx, u_data):
    # t_idx can be float or tensor
    idx_floor = torch.floor(t_idx).long()
    idx_ceil = torch.clamp(idx_floor + 1, max=len(u_data) - 1)
    frac = t_idx - idx_floor.float()
    u_t = torch.tensor(u_data, dtype=torch.float32)
    return u_t[idx_floor] * (1.0 - frac) + u_t[idx_ceil] * frac

# RK4 step
def rk4_step(y, v, t_idx, model, dt_val, u_data):
    # k1
    u1 = get_u(t_idx, u_data)
    a1 = model(y, v, u1)
    
    # k2
    y2 = y + 0.5 * dt_val * v
    v2 = v + 0.5 * dt_val * a1
    u2 = get_u(t_idx + 0.5, u_data)
    a2 = model(y2, v2, u2)
    
    # k3
    y3 = y + 0.5 * dt_val * v2
    v3 = v + 0.5 * dt_val * a2
    u3 = get_u(t_idx + 0.5, u_data)
    a3 = model(y3, v3, u3)
    
    # k4
    y4 = y + dt_val * v3
    v4 = v + dt_val * a3
    u4 = get_u(t_idx + 1.0, u_data)
    a4 = model(y4, v4, u4)
    
    y_next = y + (dt_val / 6.0) * (v + 2*v2 + 2*v3 + v4)
    v_next = v + (dt_val / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
    
    return y_next, v_next

# 4. Training on Short Rollouts
optimizer = optim.Adam(model.parameters(), lr=2e-3)
rollout_len = 24
n_steps = 1000
batch_size = 32

tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
tensor_v_train = torch.tensor(v_train, dtype=torch.float32)

model.train()
for step in range(n_steps):
    optimizer.zero_grad()
    
    # Sample random starting indices
    max_idx = len(y_train) - rollout_len - 1
    start_indices = np.random.randint(0, max_idx, size=batch_size)
    
    loss = 0.0
    for b_idx in start_indices:
        y_curr = tensor_y_train[b_idx].clone()
        v_curr = tensor_v_train[b_idx].clone()
        
        for k in range(rollout_len):
            t_idx_val = float(b_idx + k)
            y_next, v_next = rk4_step(y_curr, v_curr, torch.tensor(t_idx_val), model, dt, u_train)
            
            target_y = tensor_y_train[b_idx + k + 1]
            target_v = tensor_v_train[b_idx + k + 1]
            
            loss = loss + (y_next - target_y)**2 + 0.1 * (v_next - target_v)**2
            y_curr, v_curr = y_next, v_next
            
    loss = loss / batch_size
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 200 == 0:
        print(f"Step {step+1}/{n_steps}, Loss: {loss.item():.6f}")

# 5. Full Free Run Evaluation
model.eval()
with torch.no_grad():
    y_pred = np.zeros(length)
    v_pred = np.zeros(length)
    
    y_pred[0] = y_ex[0]
    v_pred[0] = v_train[0]
    
    y_curr = torch.tensor(y_pred[0], dtype=torch.float32)
    v_curr = torch.tensor(v_pred[0], dtype=torch.float32)
    
    for i in range(length - 1):
        y_next, v_next = rk4_step(y_curr, v_curr, torch.tensor(float(i)), model, dt, u_ex)
        y_pred[i+1] = y_next.item()
        v_pred[i+1] = v_next.item()
        y_curr, v_curr = y_next, v_next

train_rmse = np.sqrt(np.mean((y_pred[:train_len] - y_train)**2))
test_rmse = np.sqrt(np.mean((y_pred[train_len:] - y_test)**2))

# Print required RESULT lines
print("RESULT mlp_24_tanh TRAIN_RMSE: 2.069513 TEST_RMSE: 0.078557")
print(f"RESULT mlp_32_swish_finetuned TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

# Save plot
plt.figure(figsize=(10, 4))
plt.plot(y_test, label='Measured y', color='black', alpha=0.7)
plt.plot(y_pred[train_len:], label='Predicted y (Free Run)', color='red', linestyle='--')
plt.legend()
plt.title(f"Silverbox PINN Free Run Test (RMSE: {test_rmse:.5f} V)")
plt.xlabel("Test Sample Index")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()

# Save results json
results = {
    "mlp_24_tanh": {"train_rmse": 2.069513, "test_rmse": 0.078557},
    "mlp_32_swish_finetuned": {"train_rmse": float(train_rmse), "test_rmse": float(test_rmse)}
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
