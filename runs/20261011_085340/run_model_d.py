import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

start_idx = 42650
end_idx = 46746
excerpt_slice = slice(start_idx, end_idx)

u_raw = u_full[excerpt_slice]
y_raw = y_full[excerpt_slice]

u_mean = np.mean(u_raw)
y_mean = np.mean(y_raw)
u = u_raw - u_mean
y = y_raw - y_mean

fs = 610.35
dt = 1.0 / fs

train_len = 3072
test_len = 1024

y_train = y[:train_len]
u_train = u[:train_len]

v_train = np.zeros_like(y_train)
v_train[0] = (y_train[1] - y_train[0]) / dt
v_train[1:] = (y_train[1:] - y_train[:-1]) / dt

# Model D: Wider network (64 units), 2 hidden layers, Tanh activation
class AccelerationNetD(nn.Module):
    def __init__(self):
        super(AccelerationNetD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, y, v, u_val):
        y_n = y / 0.055
        v_n = v / 21.5
        u_n = u_val / 0.022
        inputs = torch.stack([y_n, v_n, u_n], dim=-1)
        return self.net(inputs).squeeze(-1)

model = AccelerationNetD()

def rk4_step(y_t, v_t, t_idx, u_tensor, model, dt):
    def get_u(t):
        idx0 = torch.floor(t).long()
        idx1 = torch.clamp(idx0 + 1, max=len(u_tensor) - 1)
        frac = t - idx0.float()
        u0 = u_tensor[idx0]
        u1 = u_tensor[idx1]
        return u0 + frac * (u1 - u0)

    u_k1 = get_u(t_idx)
    a1 = model(y_t, v_t, u_k1)
    dy1 = v_t
    dv1 = a1

    t_k2 = t_idx + 0.5
    u_k2 = get_u(t_k2)
    y_k2 = y_t + 0.5 * dt * dy1
    v_k2 = v_t + 0.5 * dt * dv1
    a2 = model(y_k2, v_k2, u_k2)
    dy2 = v_k2
    dv2 = a2

    t_k3 = t_idx + 0.5
    u_k3 = get_u(t_k3)
    y_k3 = y_t + 0.5 * dt * dy2
    v_k3 = v_t + 0.5 * dt * dv2
    a3 = model(y_k3, v_k3, u_k3)
    dy3 = v_k3
    dv3 = a3

    t_k4 = t_idx + 1.0
    u_k4 = get_u(t_k4)
    y_k4 = y_t + dt * dy3
    v_k4 = v_t + dt * dv3
    a4 = model(y_k4, v_k4, u_k4)
    dy4 = v_k4
    dv4 = a4

    y_next = y_t + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
    v_next = v_t + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)

    return y_next, v_next

optimizer = optim.Adam(model.parameters(), lr=1e-3)
u_tensor = torch.tensor(u, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

rollout_len = 32
n_steps = 1200
batch_size = 32

print("Starting training Model D...")
model.train()
for step in range(1, n_steps + 1):
    max_start = train_len - rollout_len - 1
    start_indices = np.random.randint(0, max_start, size=batch_size)
    
    y_batch = torch.tensor([y[i] for i in start_indices], dtype=torch.float32)
    v_batch = torch.tensor([v_train[i] for i in start_indices], dtype=torch.float32)
    
    loss = 0.0
    curr_y = y_batch
    curr_v = v_batch
    
    for s in range(rollout_len):
        t_indices = torch.tensor([si + s for si in start_indices], dtype=torch.float32)
        next_y, next_v = rk4_step(curr_y, curr_v, t_indices, u_tensor, model, dt)
        
        target_y = torch.tensor([y[si + s + 1] for si in start_indices], dtype=torch.float32)
        loss = loss + torch.mean((next_y - target_y) ** 2)
        
        curr_y = next_y
        curr_v = next_v
        
    loss = loss / rollout_len
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if step % 300 == 0 or step == 1:
        print(f"Step {step}/{n_steps}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred = np.zeros_like(y)
    v_pred = np.zeros_like(y)
    
    curr_y = torch.tensor([y[0]], dtype=torch.float32)
    curr_v = torch.tensor([v_train[0]], dtype=torch.float32)
    
    y_pred[0] = curr_y.item()
    v_pred[0] = curr_v.item()
    
    for i in range(len(y) - 1):
        t_idx = torch.tensor([float(i)], dtype=torch.float32)
        curr_y, curr_v = rk4_step(curr_y, curr_v, t_idx, u_tensor, model, dt)
        y_pred[i+1] = curr_y.item()
        v_pred[i+1] = curr_v.item()

train_pred = y_pred[:train_len]
test_pred = y_pred[train_len:]
train_true = y[:train_len]
test_true = y[train_len:]

train_rmse = float(np.sqrt(np.mean((train_pred - train_true) ** 2)))
test_rmse = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))

print(f"RESULT ModelD TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

results = {
    "model": "ModelD",
    "train_rmse": train_rmse,
    "test_rmse": test_rmse
}
with open("results_ModelD.json", "w") as f:
    json.dump(results, f)

plt.figure(figsize=(10, 4))
time_axis = np.arange(test_len) * dt
plt.plot(time_axis, test_true, label='Measured y(t)', color='black', alpha=0.7)
plt.plot(time_axis, test_pred, label='Predicted Free Run y_hat(t)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Silverbox PINN Free Run on Test Segment (RMSE = {test_rmse:.4f} V)')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()
