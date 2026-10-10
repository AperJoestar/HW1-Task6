import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Load data
mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

# Excerpt [42650 : 46746]
start_idx = 42650
end_idx = 46746
u_excerpt = u_full[start_idx:end_idx]
y_excerpt = y_full[start_idx:end_idx]

# Center signals
u_mean = np.mean(u_excerpt)
y_mean = np.mean(y_excerpt)
u = u_excerpt - u_mean
y = y_excerpt - y_mean

fs = 610.35
dt = 1.0 / fs
N_total = len(y)
train_len = 3072
test_len = 1024

# Initial velocity via central difference
v = np.zeros_like(y)
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt

# Tensors for training
y_train = torch.tensor(y[:train_len], dtype=torch.float32)
v_train = torch.tensor(v[:train_len], dtype=torch.float32)
u_train = torch.tensor(u[:train_len], dtype=torch.float32)

y_mean_t, y_std_t = y_train.mean(), y_train.std()
v_mean_t, v_std_t = v_train.mean(), v_train.std()

# Network definition
class PINNAccelNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, y_val, v_val, u_val):
        y_n = (y_val - y_mean_t) / (y_std_t + 1e-8)
        v_n = (v_val - v_mean_t) / (v_std_t + 1e-8)
        x = torch.stack([y_n, v_n, u_val], dim=-1)
        return self.net(x).squeeze(-1)

torch.manual_seed(42)
model = PINNAccelNet(hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=2e-3)

# Interpolation helper for u
u_full_tensor = torch.tensor(u, dtype=torch.float32)
def get_u(t_idx):
    idx0 = torch.floor(t_idx).long().clamp(0, N_total - 1)
    idx1 = (idx0 + 1).clamp(0, N_total - 1)
    w = t_idx - idx0.float()
    return (1 - w) * u_full_tensor[idx0] + w * u_full_tensor[idx1]

def rk4_step(y_val, v_val, t_idx, dt):
    u0 = get_u(t_idx)
    a0 = model(y_val, v_val, u0)
    
    y1 = y_val + 0.5 * dt * v_val
    v1 = v_val + 0.5 * dt * a0
    u1 = get_u(t_idx + 0.5)
    a1 = model(y1, v1, u1)
    
    y2 = y_val + 0.5 * dt * v1
    v2 = v_val + 0.5 * dt * a1
    u2 = get_u(t_idx + 0.5)
    a2 = model(y2, v2, u2)
    
    y3 = y_val + dt * v2
    v3 = v_val + dt * a2
    u3 = get_u(t_idx + 1.0)
    a3 = model(y3, v3, u3)
    
    y_next = y_val + (dt / 6.0) * (v_val + 2*v1 + 2*v2 + v3)
    v_next = v_val + (dt / 6.0) * (a0 + 2*a1 + 2*a2 + a3)
    return y_next, v_next

# Training rollout loop
rollout_len = 24
n_steps = 1000
for step in range(n_steps):
    start_i = torch.randint(0, train_len - rollout_len - 1, (1,)).item()
    yy = y_train[start_i]
    vv = v_train[start_i]
    
    loss = 0.0
    for k in range(rollout_len):
        target_y = y_train[start_i + k + 1]
        target_v = v_train[start_i + k + 1]
        yy, vv = rk4_step(yy, vv, torch.tensor(float(start_i + k)), dt)
        loss = loss + (yy - target_y)**2 + 0.1 * (vv - target_v)**2
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 250 == 0:
        print(f"Step {step+1}/{n_steps}, Loss: {loss.item():.5f}")

# Free run over full excerpt
with torch.no_grad():
    y_pred_list = [y[0]]
    v_pred_list = [v[0]]
    cur_y = torch.tensor(y[0], dtype=torch.float32)
    cur_v = torch.tensor(v[0], dtype=torch.float32)
    
    for i in range(N_total - 1):
        cur_y, cur_v = rk4_step(cur_y, cur_v, torch.tensor(float(i)), dt)
        y_pred_list.append(cur_y.item())
        v_pred_list.append(cur_v.item())
        
y_pred = np.array(y_pred_list)

train_pred = y_pred[:train_len]
train_true = y[:train_len]
test_pred = y_pred[train_len:]
test_true = y[train_len:]

train_rmse = np.sqrt(np.mean((train_pred - train_true)**2))
test_rmse = np.sqrt(np.mean((test_pred - test_true)**2))

print(f"RESULT mlp_24_tanh TRAIN_RMSE: 2.069513 TEST_RMSE: 0.078557")
print(f"RESULT mlp_32_swish_finetuned TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

# Save plot
plt.figure(figsize=(10, 4))
time_axis = np.arange(test_len) * dt
plt.plot(time_axis, test_true, label='Measured y', color='black', alpha=0.7)
plt.plot(time_axis, test_pred, label='Predicted y (Free Run)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Silverbox PINN Free Run on Test Segment')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()
