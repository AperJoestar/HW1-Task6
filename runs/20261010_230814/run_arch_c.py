import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data
mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

slice_idx = slice(42650, 46746)
u_slice = u_full[slice_idx]
y_slice = y_full[slice_idx]

u_mean = np.mean(u_slice)
y_mean = np.mean(y_slice)

u = u_slice - u_mean
y = y_slice - y_mean

train_len = 3072
test_len = 1024
dt = 1.0 / 610.35

y_std = np.float32(np.std(y[:train_len]))
v_full = np.gradient(y, dt)
v_std = np.float32(np.std(v_full[:train_len]))
u_std = np.float32(np.std(u[:train_len]))

# Define Kept Architecture C (Deeper/Wider or optimized hyperparameters for ArchB variant)
# Let's train an optimized architecture (ArchC: Linear(3, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 1) with L=64 and lr=1e-3, 2000 steps)
class AccelNetC(nn.Module):
    def __init__(self):
        super(AccelNetC, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, y_s, v_s, u_s):
        if y_s.ndim == 1:
            y_s = y_s.unsqueeze(-1)
        if v_s.ndim == 1:
            v_s = v_s.unsqueeze(-1)
        if u_s.ndim == 1:
            u_s = u_s.unsqueeze(-1)
        x = torch.cat([y_s, v_s, u_s], dim=-1)
        return self.net(x)

device = torch.device("cpu")
model = AccelNetC().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

u_t = torch.tensor(u, dtype=torch.float32, device=device)
y_t = torch.tensor(y, dtype=torch.float32, device=device)

u_np = u
u_half_np = 0.5 * (u_np[:-1] + u_np[1:])
u_half_t = torch.tensor(u_half_np, dtype=torch.float32, device=device)

def rk4_step(y_curr, v_curr, u_curr, u_next, u_half):
    def accel(yy, vv, uu):
        return model(yy / y_std, vv / v_std, uu / u_std).squeeze(-1)

    a1 = accel(y_curr, v_curr, u_curr)
    k1_y = v_curr
    k1_v = a1

    y2 = y_curr + 0.5 * dt * k1_y
    v2 = v_curr + 0.5 * dt * k1_v
    a2 = accel(y2, v2, u_half)
    k2_y = v2
    k2_v = a2

    y3 = y_curr + 0.5 * dt * k2_y
    v3 = v_curr + 0.5 * dt * k2_v
    a3 = accel(y3, v3, u_half)
    k3_y = v3
    k3_v = a3

    y4 = y_curr + dt * k3_y
    v4 = v_curr + dt * k3_v
    a4 = accel(y4, v4, u_next)
    k4_y = v4
    k4_v = a4

    y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_next = v_curr + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return y_next, v_next

L = 64
batch_size = 64
num_steps = 2000
max_start = train_len - L - 1

model.train()
for step in range(num_steps):
    starts = np.random.randint(0, max_start, size=batch_size)
    
    y_batch = y_t[starts]
    v_batch = torch.tensor(v_full[starts], dtype=torch.float32, device=device)
    
    loss = 0.0
    curr_y = y_batch
    curr_v = v_batch
    
    for i in range(L):
        idx = starts + i
        u_curr = u_t[idx]
        u_next = u_t[idx+1]
        u_half = u_half_t[idx]
        
        next_y, next_v = rk4_step(curr_y, curr_v, u_curr, u_next, u_half)
        target_y = y_t[idx+1]
        
        loss = loss + torch.mean((next_y - target_y)**2)
        curr_y = next_y
        curr_v = next_v
        
    loss = loss / L
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 400 == 0:
        print(f"ArchC Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    pred_y = np.zeros_like(y)
    curr_y = torch.tensor(y[0], dtype=torch.float32, device=device)
    curr_v = torch.tensor(v_full[0], dtype=torch.float32, device=device)
    pred_y[0] = curr_y.item()
    
    for i in range(len(y) - 1):
        u_curr = u_t[i]
        u_next = u_t[i+1]
        u_half = u_half_t[i]
        
        next_y, next_v = rk4_step(curr_y.unsqueeze(0), curr_v.unsqueeze(0), u_curr.unsqueeze(0), u_next.unsqueeze(0), u_half.unsqueeze(0))
        curr_y = next_y.squeeze(0)
        curr_v = next_v.squeeze(0)
        pred_y[i+1] = curr_y.item()

train_rmse = np.sqrt(np.mean((pred_y[:train_len] - y[:train_len])**2))
test_rmse = np.sqrt(np.mean((pred_y[train_len:] - y[train_len:])**2))

print(f"RESULT ArchC TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

# Save test plot
plt.figure(figsize=(10, 4))
time_axis = np.arange(test_len) * dt
plt.plot(time_axis, y[train_len:], label='Measured y (Test)', color='black', alpha=0.7)
plt.plot(time_axis, pred_y[train_len:], label='Predicted y (Free Run)', color='red', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Silverbox PINN Free-Run Test Segment Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('test_plot.png', dpi=300)
plt.close()
