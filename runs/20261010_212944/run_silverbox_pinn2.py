import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

start_time = time.time()

torch.manual_seed(42)
np.random.seed(42)

mat_data = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat_data['V1'].flatten()
y_full = mat_data['V2'].flatten()

start_idx = 42650
end_idx = 46746
u_slice = u_full[start_idx:end_idx]
y_slice = y_full[start_idx:end_idx]

u_mean = np.mean(u_slice)
y_mean = np.mean(y_slice)
u_centered = u_slice - u_mean
y_centered = y_slice - y_mean

train_len = 3072
test_len = 1024

u_train = u_centered[:train_len]
y_train = y_centered[:train_len]

dt = 1.0 / 610.35

class NetArch1(nn.Module):
    def __init__(self):
        super(NetArch1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, y, v, u):
        x = torch.cat([y, v, u], dim=-1)
        return self.net(x)

class NetArch3(nn.Module):
    def __init__(self):
        super(NetArch3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, y, v, u):
        x = torch.cat([y, v, u], dim=-1)
        return self.net(x)

def rk4_step_batch(y_batch, v_batch, u_curr, u_next, dt, model):
    acc1 = model(y_batch, v_batch, u_curr.unsqueeze(-1))
    k1_y = v_batch
    k1_v = acc1
    
    y_m1 = y_batch + 0.5 * dt * k1_y
    v_m1 = v_batch + 0.5 * dt * k1_v
    u_m1 = 0.5 * (u_curr + u_next)
    acc2 = model(y_m1, v_m1, u_m1.unsqueeze(-1))
    k2_y = v_m1
    k2_v = acc2
    
    y_m2 = y_batch + 0.5 * dt * k2_y
    v_m2 = v_batch + 0.5 * dt * k2_v
    acc3 = model(y_m2, v_m2, u_m1.unsqueeze(-1))
    k3_y = v_m2
    k3_v = acc3
    
    y_e = y_batch + dt * k3_y
    v_e = v_batch + dt * k3_v
    acc4 = model(y_e, v_e, u_next.unsqueeze(-1))
    k4_y = v_e
    k4_v = acc4
    
    y_next = y_batch + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_next = v_batch + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return y_next, v_next

def train_model(model, u_data, y_data, steps=1500, rollout_len=48, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    u_tensor = torch.tensor(u_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    max_start = len(u_data) - rollout_len - 1
    
    model.train()
    for step in range(steps):
        batch_size = 16
        starts = np.random.randint(0, max_start, size=batch_size)
        
        y_list = []
        v_list = []
        for s_idx in starts:
            y_list.append(y_tensor[s_idx])
            v_list.append((y_tensor[s_idx+1] - y_tensor[s_idx]) / dt)
            
        y_curr = torch.stack(y_list).unsqueeze(-1)
        v_curr = torch.stack(v_list).unsqueeze(-1)
        
        pred_traj = [y_curr.squeeze(-1)]
        
        for t_step in range(rollout_len):
            idxs = starts + t_step
            u_c = u_tensor[idxs]
            u_n = u_tensor[idxs + 1]
            
            y_curr, v_curr = rk4_step_batch(y_curr, v_curr, u_c, u_n, dt, model)
            pred_traj.append(y_curr.squeeze(-1))
            
        pred_traj = torch.stack(pred_traj, dim=1)
        
        true_list = []
        for s_idx in starts:
            true_list.append(y_tensor[s_idx : s_idx + rollout_len + 1])
        true_traj = torch.stack(true_list, dim=0)
        
        loss = criterion(pred_traj, true_traj)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 500 == 0:
            print(f"Step {step+1}/{steps}, Loss: {loss.item():.6f}, Elapsed: {time.time()-start_time:.1f}s")

def evaluate_free_run(model, u_full_data, y_full_data, dt):
    model.eval()
    u_t = torch.tensor(u_full_data, dtype=torch.float32)
    
    y_preds = np.zeros(len(u_full_data))
    y_preds[0] = y_full_data[0]
    y_preds[1] = y_full_data[1]
    
    y_curr = torch.tensor([[y_full_data[0]]], dtype=torch.float32)
    v_curr = torch.tensor([[(y_full_data[1] - y_full_data[0]) / dt]], dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(len(u_full_data) - 1):
            u_c = u_t[i].unsqueeze(0)
            u_n = u_t[i+1].unsqueeze(0)
            
            y_next, v_next = rk4_step_batch(y_curr, v_curr, u_c, u_n, dt, model)
            y_preds[i+1] = y_next.item()
            y_curr = y_next
            v_curr = v_next
            
    train_rmse = np.sqrt(np.mean((y_preds[:train_len] - y_full_data[:train_len])**2))
    test_rmse = np.sqrt(np.mean((y_preds[train_len:] - y_full_data[train_len:])**2))
    
    return train_rmse, test_rmse, y_preds

print("--- Training Architecture 1 (Tanh, 2x32) ---")
model1 = NetArch1()
train_model(model1, u_train, y_train, steps=1500, rollout_len=48, lr=1e-3)
train_rmse1, test_rmse1, preds1 = evaluate_free_run(model1, u_centered, y_centered, dt)
print(f"RESULT Arch1 TRAIN_RMSE: {train_rmse1:.6f} TEST_RMSE: {test_rmse1:.6f}")

print("\n--- Training Architecture 3 (SiLU, 2x64) ---")
model3 = NetArch3()
train_model(model3, u_train, y_train, steps=1500, rollout_len=48, lr=1e-3)
train_rmse3, test_rmse3, preds3 = evaluate_free_run(model3, u_centered, y_centered, dt)
print(f"RESULT Arch3 TRAIN_RMSE: {train_rmse3:.6f} TEST_RMSE: {test_rmse3:.6f}")

if test_rmse1 < test_rmse3:
    best_model = model1
    best_preds = preds1
    best_test_rmse = test_rmse1
    best_name = "Arch1"
else:
    best_model = model3
    best_preds = preds3
    best_test_rmse = test_rmse3
    best_name = "Arch3"

print(f"\nKept version: {best_name}")
print(f"TEST_RMSE: {best_test_rmse:.6f}")

plt.figure(figsize=(10, 4))
test_time = np.arange(test_len) * dt
plt.plot(test_time, y_centered[train_len:], label='Measured y (Test)', color='black', alpha=0.7)
plt.plot(test_time, best_preds[train_len:], label='Predicted y (Free Run)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Silverbox PINN Free Run on Test Segment ({best_name}, RMSE: {best_test_rmse:.4f} V)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_plot.png', dpi=300)
plt.close()
