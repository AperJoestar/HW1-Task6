import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

# 1. Load data
mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

start_idx = 42650
length = 4096
u_excerpt = u_full[start_idx:start_idx+length]
y_excerpt = y_full[start_idx:start_idx+length]

u_mean = np.mean(u_excerpt)
y_mean = np.mean(y_excerpt)

u = u_excerpt - u_mean
y = y_excerpt - y_mean

dt = 1.0 / 610.35
train_len = 3072
test_len = 1024

sigma_y = 0.055
sigma_v = 21.5
sigma_u = 0.022

class AccelNet(nn.Module):
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
        ys = y / sigma_y
        vs = v / sigma_v
        us = u / sigma_u
        inp = torch.stack([ys, vs, us], dim=-1)
        return self.net(inp).squeeze(-1)

model = AccelNet()

L_rollout = 16
train_u = u[:train_len]
train_y = y[:train_len]
train_v = np.gradient(train_y, dt)

num_windows = train_len - L_rollout
stride = 8
windows_y = []
windows_v = []
windows_u = []

for i in range(0, num_windows, stride):
    windows_y.append(train_y[i:i+L_rollout])
    windows_v.append(train_v[i:i+L_rollout])
    windows_u.append(train_u[i:i+L_rollout])

windows_y = torch.tensor(np.array(windows_y), dtype=torch.float32)
windows_v = torch.tensor(np.array(windows_v), dtype=torch.float32)
windows_u = torch.tensor(np.array(windows_u), dtype=torch.float32)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def rk4_step(y_t, v_t, u_t, u_next, dt):
    a1 = model(y_t, v_t, u_t)
    dy1 = v_t
    dv1 = a1
    
    y_mid1 = y_t + 0.5 * dt * dy1
    v_mid1 = v_t + 0.5 * dt * dv1
    u_mid1 = 0.5 * (u_t + u_next)
    a2 = model(y_mid1, v_mid1, u_mid1)
    dy2 = v_mid1
    dv2 = a2
    
    y_mid2 = y_t + 0.5 * dt * dy2
    v_mid2 = v_t + 0.5 * dt * dv2
    a3 = model(y_mid2, v_mid2, u_mid1)
    dy3 = v_mid2
    dv3 = a3
    
    y_end = y_t + dt * dy3
    v_end = v_t + dt * dv3
    a4 = model(y_end, v_end, u_next)
    dy4 = v_end
    dv4 = a4
    
    y_next = y_t + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
    v_next = v_t + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
    return y_next, v_next

epochs = 300
batch_size = 128
dataset_size = windows_y.shape[0]

for epoch in range(epochs):
    perm = torch.randperm(dataset_size)
    total_loss = 0.0
    for i in range(0, dataset_size, batch_size):
        idx = perm[i:i+batch_size]
        b_y = windows_y[idx]
        b_v = windows_v[idx]
        b_u = windows_u[idx]
        
        loss = 0.0
        curr_y = b_y[:, 0]
        curr_v = b_v[:, 0]
        
        for t in range(L_rollout - 1):
            u_t = b_u[:, t]
            u_next = b_u[:, t+1]
            curr_y, curr_v = rk4_step(curr_y, curr_v, u_t, u_next, dt)
            target_y = b_y[:, t+1]
            target_v = b_v[:, t+1]
            loss = loss + criterion(curr_y, target_y) + 0.1 * criterion(curr_v, target_v)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/dataset_size:.6f}")

def free_run(u_signal, y_true_full, init_y, init_v):
    preds = [init_y]
    curr_y = torch.tensor(init_y, dtype=torch.float32)
    curr_v = torch.tensor(init_v, dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(len(u_signal) - 1):
            u_t = torch.tensor(u_signal[i], dtype=torch.float32)
            u_next = torch.tensor(u_signal[i+1], dtype=torch.float32)
            curr_y, curr_v = rk4_step(curr_y, curr_v, u_t, u_next, dt)
            preds.append(curr_y.item())
    return np.array(preds)

init_y = y[0]
init_v = (y[1] - y[0]) / dt
pred_full = free_run(u, y, init_y, init_v)

train_preds = pred_full[:train_len]
test_preds = pred_full[train_len:]

train_rmse = np.sqrt(np.mean((train_preds - y[:train_len])**2))
test_rmse = np.sqrt(np.mean((test_preds - y[train_len:])**2))

print(f"RESULT arch1 TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")

with open("results_arch1.json", "w") as f:
    json.dump({"train_rmse": train_rmse, "test_rmse": test_rmse}, f)
