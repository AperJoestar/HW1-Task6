import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].squeeze()
y_full = mat['V2'].squeeze()

excerpt_slice = slice(42650, 46746)
u_ex = u_full[excerpt_slice]
y_ex = y_full[excerpt_slice]

u_mean = np.mean(u_ex)
y_mean = np.mean(y_ex)
u_centered = u_ex - u_mean
y_centered = y_ex - y_mean

train_len = 3072
test_len = 1024
dt = 1.0 / 610.35

u_train = u_centered[:train_len]
y_train = y_centered[:train_len]
v_train = np.gradient(y_train, dt)
a_train = np.gradient(v_train, dt)

sy = float(np.std(y_train))
sv = float(np.std(v_train))
su = float(np.std(u_train))
sa = float(np.std(a_train))

u_tensor = torch.tensor(u_centered, dtype=torch.float32)
y_tensor = torch.tensor(y_centered, dtype=torch.float32)
v_centered = np.gradient(y_centered, dt)
v_tensor = torch.tensor(v_centered, dtype=torch.float32)

class PINN_Acceleration_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, y, v, u):
        y_in = y / sy
        v_in = v / sv
        u_in = u / su
        inp = torch.stack([y_in, v_in, u_in], dim=-1)
        acc_scaled = self.net(inp).squeeze(-1)
        return acc_scaled * sa

device = torch.device('cpu')
model = PINN_Acceleration_v3().to(device)

def rk4_step(y, v, u_curr, u_next, dt):
    a1 = model(y, v, u_curr)
    dy1 = v
    dv1 = a1
    
    y2 = y + 0.5 * dt * dy1
    v2 = v + 0.5 * dt * dv1
    u2 = 0.5 * (u_curr + u_next)
    a2 = model(y2, v2, u2)
    dy2 = v2
    dv2 = a2
    
    y3 = y + 0.5 * dt * dy2
    v3 = v + 0.5 * dt * dv2
    u3 = u2
    a3 = model(y3, v3, u3)
    dy3 = v3
    dv3 = a3
    
    y4 = y + dt * dy3
    v4 = v + dt * dv3
    u4 = u_next
    a4 = model(y4, v4, u4)
    dy4 = v4
    dv4 = a4
    
    y_next = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
    v_next = v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
    
    return y_next, v_next

def rollout(y0, v0, u_seq, dt):
    if u_seq.ndim == 1:
        u_seq = u_seq.unsqueeze(0)
    batch_size, length = u_seq.shape
    
    y_list = [y0]
    v_list = [v0]
    
    curr_y = y0
    curr_v = v0
    
    for t in range(length - 1):
        u_curr = u_seq[:, t]
        u_next = u_seq[:, t+1]
        curr_y, curr_v = rk4_step(curr_y, curr_v, u_curr, u_next, dt)
        y_list.append(curr_y)
        v_list.append(curr_v)
        
    return torch.stack(y_list, dim=1), torch.stack(v_list, dim=1)

optimizer = optim.Adam(model.parameters(), lr=2e-3)
stages = [
    (32, 400),
    (64, 400),
    (128, 400)
]

total_steps = 0
for L, steps in stages:
    for step in range(steps):
        total_steps += 1
        max_idx = train_len - L
        if max_idx <= 0:
            max_idx = 1
        start_indices = torch.randint(0, max_idx, (32,))
        
        u_batch = torch.stack([u_tensor[i:i+L] for i in start_indices])
        y_target = torch.stack([y_tensor[i:i+L] for i in start_indices])
        
        y0_batch = y_tensor[start_indices]
        v0_batch = v_tensor[start_indices]
        
        y_pred, _ = rollout(y0_batch, v0_batch, u_batch, dt)
        loss = torch.mean((y_pred - y_target) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

with torch.no_grad():
    y0_full = y_tensor[0:1]
    v0_full = v_tensor[0:1]
    y_pred_full, _ = rollout(y0_full, v0_full, u_tensor.unsqueeze(0), dt)
    y_pred_full = y_pred_full.squeeze(0)

y_pred_np = y_pred_full.numpy()
y_true_np = y_centered

train_pred = y_pred_np[:train_len]
train_true = y_true_np[:train_len]
test_pred = y_pred_np[train_len:]
test_true = y_true_np[train_len:]

train_rmse = float(np.sqrt(np.mean((train_pred - train_true) ** 2)))
test_rmse = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
baseline_test_rmse = float(np.sqrt(np.mean(test_true ** 2)))

print(f"RESULT arch2 TRAIN_RMSE: {train_rmse:.5f} TEST_RMSE: {test_rmse:.5f}")
print(f"TEST_RMSE: {test_rmse:.5f}")

results = {
    "arch": "arch2",
    "train_rmse": train_rmse,
    "test_rmse": test_rmse,
    "baseline_test_rmse": baseline_test_rmse
}
with open("results_arch2.json", "w") as f:
    json.dump(results, f)

plt.figure(figsize=(10, 4))
time_axis = np.arange(test_len) * dt
plt.plot(time_axis, test_true, label='Measured $y(t)$', color='black', alpha=0.7)
plt.plot(time_axis, test_pred, label='Free-run Predicted $y(t)$ (arch2)', color='red', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title(f'Silverbox PINN Free-Run Test Segment (RMSE = {test_rmse:.5f} V)')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png', dpi=300)
plt.close()
