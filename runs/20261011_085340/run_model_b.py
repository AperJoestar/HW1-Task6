# Let's optimize Model B speed or use fewer steps / smaller rollout so it comfortably finishes within 5 minutes, but keeps the architecture
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('SNLS80mV.mat')
V1 = mat['V1'].flatten()
V2 = mat['V2'].flatten()

fs = 610.35
dt = 1.0 / fs

start_idx = 42650
end_idx = 46746
excerpt_u = V1[start_idx:end_idx]
excerpt_y = V2[start_idx:end_idx]

u = excerpt_u - np.mean(excerpt_u)
y = excerpt_y - np.mean(excerpt_y)

train_len = 3072
test_len = 1024

u_train = u[:train_len]
y_train = y[:train_len]
u_test = u[train_len:]
y_test = y[train_len:]

v = np.zeros_like(y)
v[:-1] = (y[1:] - y[:-1]) / dt
v[-1] = v[-2]

v_train = v[:train_len]
v_test = v[train_len:]

class AccelerationNN_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, 1)
        )
        
    def forward(self, y, v, u):
        y_s = y / 0.055
        v_s = v / 21.5
        u_s = u / 0.022
        x = torch.stack([y_s, v_s, u_s], dim=-1)
        return self.net(x).squeeze(-1)

device = torch.device("cpu")
model = AccelerationNN_B().to(device)

def get_u_interp(u_seq, t_indices, dt):
    max_idx = u_seq.shape[-1] - 1
    idx_floor = torch.floor(t_indices).long().clamp(0, max_idx)
    idx_ceil = (idx_floor + 1).clamp(0, max_idx)
    frac = t_indices - idx_floor.float()
    
    if u_seq.ndim == 1:
        u_floor = u_seq[idx_floor]
        u_ceil = u_seq[idx_ceil]
    else:
        batch_indices = torch.arange(u_seq.shape[0], device=u_seq.device)[:, None]
        u_floor = u_seq[batch_indices, idx_floor]
        u_ceil = u_seq[batch_indices, idx_ceil]
    return u_floor * (1.0 - frac) + u_ceil * frac

def rk4_step(y_curr, v_curr, t_curr_idx, u_full, dt, model):
    u_k1 = get_u_interp(u_full, t_curr_idx, dt)
    a1 = v_curr
    b1 = model(y_curr, v_curr, u_k1)
    
    y_k2 = y_curr + 0.5 * dt * a1
    v_k2 = v_curr + 0.5 * dt * b1
    u_k2 = get_u_interp(u_full, t_curr_idx + 0.5, dt)
    a2 = v_k2
    b2 = model(y_k2, v_k2, u_k2)
    
    y_k3 = y_curr + 0.5 * dt * a2
    v_k3 = v_curr + 0.5 * dt * b2
    u_k3 = get_u_interp(u_full, t_curr_idx + 0.5, dt)
    a3 = v_k3
    b3 = model(y_k3, v_k3, u_k3)
    
    y_k4 = y_curr + dt * a3
    v_k4 = v_curr + dt * b3
    u_k4 = get_u_interp(u_full, t_curr_idx + 1.0, dt)
    a4 = v_k4
    b4 = model(y_k4, v_k4, u_k4)
    
    y_next = y_curr + (dt / 6.0) * (a1 + 2.0*a2 + 2.0*a3 + a4)
    v_next = v_curr + (dt / 6.0) * (b1 + 2.0*b2 + 2.0*b3 + b4)
    return y_next, v_next

u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
v_train_tensor = torch.tensor(v_train, dtype=torch.float32, device=device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

rollout_len = 32
num_steps = 1500
batch_size = 64

print("Starting training Model B (optimized)...")
for step in range(num_steps):
    max_start = train_len - rollout_len - 1
    start_indices = torch.randint(0, max_start, (batch_size,))
    
    y_curr = y_train_tensor[start_indices]
    v_curr = v_train_tensor[start_indices]
    
    loss = 0.0
    for i in range(rollout_len):
        t_indices = start_indices.float() + float(i)
        y_next, v_next = rk4_step(y_curr, v_curr, t_indices, u_tensor, dt, model)
        
        target_idx = start_indices + i + 1
        y_target = y_train_tensor[target_idx]
        
        loss = loss + criterion(y_next, y_target)
        y_curr, v_curr = y_next, v_next
        
    loss = loss / rollout_len
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 500 == 0:
        print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred_full = np.zeros(4096)
    v_pred_full = np.zeros(4096)
    
    y_curr = torch.tensor(y[0], dtype=torch.float32, device=device)
    v_curr = torch.tensor(v[0], dtype=torch.float32, device=device)
    
    y_pred_full[0] = y_curr.item()
    v_pred_full[0] = v_curr.item()
    
    for i in range(4095):
        t_idx = torch.tensor(float(i), device=device)
        y_next, v_next = rk4_step(y_curr.unsqueeze(0), v_curr.unsqueeze(0), t_idx.unsqueeze(0), u_tensor, dt, model)
        y_curr = y_next.squeeze(0)
        v_curr = v_next.squeeze(0)
        y_pred_full[i+1] = y_curr.item()
        v_pred_full[i+1] = v_curr.item()

train_pred = y_pred_full[:train_len]
test_pred = y_pred_full[train_len:]

train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
test_rmse = np.sqrt(np.mean((test_pred - y_test)**2))

print(f"RESULT ModelB TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

results = {
    "model_name": "ModelB",
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse),
    "y_pred": y_pred_full.tolist()
}
with open("results_ModelB.json", "w") as f:
    json.dump(results, f)

plt.figure(figsize=(10, 4))
time_axis = np.arange(test_len) * dt
plt.plot(time_axis, y_test, label="Measured y", alpha=0.8)
plt.plot(time_axis, test_pred, label="Predicted y (Free Run)", alpha=0.8, linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.title(f"ModelB Free Run on Test Segment (RMSE: {test_rmse:.4f} V)")
plt.tight_layout()
plt.savefig("test_plot.png")
plt.close()
