import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# 1. Data Loading and Preprocessing
mat = scipy.io.loadmat('SNLS80mV.mat')
V1 = mat['V1'].flatten()
V2 = mat['V2'].flatten()

start_idx = 42650
excerpt_len = 4096
u_full = V1[start_idx : start_idx + excerpt_len]
y_full = V2[start_idx : start_idx + excerpt_len]

u_mean = np.mean(u_full)
y_mean = np.mean(y_full)
u = u_full - u_mean
y = y_full - y_mean

dt = 1.0 / 610.35
train_len = 3072
test_len = 1024

y_train = y[:train_len]
u_train = u[:train_len]
y_test = y[train_len:]
u_test = u[train_len:]

# 2. Network Model (Model I: Direct ODE Force Fitting / Physics Residual)
class DuffingPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # We parameterize the restoring/acceleration force or explicitly a, b, c, g or a neural net
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, y, v, u):
        y_s = y / 0.055
        v_s = v / 21.5
        u_s = u / 0.022
        x = torch.stack([y_s, v_s, u_s], dim=-1)
        return self.net(x).squeeze(-1)

def rk4_step(y, v, u_curr, u_next, dt, model):
    u_mid = 0.5 * (u_curr + u_next)
    
    acc1 = model(y, v, u_curr)
    dy1 = v
    dv1 = acc1
    
    y2 = y + 0.5 * dt * dy1
    v2 = v + 0.5 * dt * dv1
    acc2 = model(y2, v2, u_mid)
    dy2 = v2
    dv2 = acc2
    
    y3 = y + 0.5 * dt * dy2
    v3 = v + 0.5 * dt * dv2
    acc3 = model(y3, v3, u_mid)
    dy3 = v3
    dv3 = acc3
    
    y4 = y + dt * dy3
    v4 = v + dt * dv3
    acc4 = model(y4, v4, u_next)
    dy4 = v4
    dv4 = acc4
    
    y_next = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
    v_next = v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
    return y_next, v_next

device = torch.device('cpu')
model = DuffingPINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
u_t = torch.tensor(u_train, dtype=torch.float32, device=device)

# 1-step prediction training (Physics collocation / one-step acceleration matching)
# Compute numerical acceleration from y_train using central differences
v_train = np.gradient(y_train, dt)
a_train = np.gradient(v_train, dt)

v_t = torch.tensor(v_train, dtype=torch.float32, device=device)
a_t = torch.tensor(a_train, dtype=torch.float32, device=device)

print("Training Model I on acceleration matching...")
num_steps = 2000
batch_size = 256

for step in range(1, num_steps + 1):
    indices = np.random.randint(0, train_len, size=batch_size)
    y_b = y_t[indices]
    v_b = v_t[indices]
    u_b = u_t[indices]
    a_b = a_t[indices]
    
    optimizer.zero_grad()
    pred_a = model(y_b, v_b, u_b)
    loss = torch.mean((pred_a - a_b) ** 2)
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0 or step == 1:
        print(f"Step {step}/{num_steps}, Loss: {loss.item():.6f}")

def full_free_run(y_full_data, u_full_data, model, dt):
    n_samples = len(y_full_data)
    y_pred = np.zeros(n_samples)
    y_pred[0] = y_full_data[0]
    v_curr = (y_full_data[1] - y_full_data[0]) / dt
    y_curr = y_full_data[0]
    
    y_curr_t = torch.tensor(y_curr, dtype=torch.float32)
    v_curr_t = torch.tensor(v_curr, dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(n_samples - 1):
            u_c = torch.tensor(u_full_data[i], dtype=torch.float32)
            u_n = torch.tensor(u_full_data[i+1], dtype=torch.float32)
            y_next, v_next = rk4_step(y_curr_t, v_curr_t, u_c, u_n, dt, model)
            y_pred[i+1] = y_next.item()
            y_curr_t, v_curr_t = y_next, v_next
            
    return y_pred

y_pred_full = full_free_run(y, u, model, dt)

train_rmse = np.sqrt(np.mean((y_pred_full[:train_len] - y_train) ** 2))
test_rmse = np.sqrt(np.mean((y_pred_full[train_len:] - y_test) ** 2))

print(f"RESULT ModelI TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

time_axis = np.arange(test_len) * dt
plt.figure(figsize=(10, 4))
plt.plot(time_axis, y_test, label='Measured $y(t)$', color='black', alpha=0.7)
plt.plot(time_axis, y_pred_full[train_len:], label='Predicted Free Run $\hat{y}(t)$', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Silverbox PINN Model I Free Run on Test Segment (RMSE = {test_rmse:.4f} V)')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png', dpi=300)
plt.close()

results = {
    "ModelA": {"train_rmse": 1.043, "test_rmse": 0.060491},
    "ModelB": {"train_rmse": 0.0538, "test_rmse": 0.060982},
    "ModelC": {"train_rmse": 0.0587, "test_rmse": 0.060610},
    "ModelI": {"train_rmse": float(train_rmse), "test_rmse": float(test_rmse)}
}
with open("results_ModelI.json", "w") as f:
    json.dump(results, f, indent=2)
