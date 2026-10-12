import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json

# Load data
mat = scipy.io.loadmat('SNLS80mV.mat')
V1 = mat['V1'].flatten()
V2 = mat['V2'].flatten()

# Excerpt slice
start_idx = 42650
length = 4096
u_full = V1[start_idx : start_idx + length]
y_full = V2[start_idx : start_idx + length]

# Center signals
u_mean = np.mean(u_full)
y_mean = np.mean(y_full)
u = u_full - u_mean
y = y_full - y_mean

dt = 1.0 / 610.35
n_train = 3072

# Initial velocity via central finite differences
v = np.zeros_like(y)
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt

# Scales for normalization
y_scale = 0.06
v_scale = 22.0
u_scale = 0.02
acc_scale = 10000.0

class PINN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, y_in, v_in, u_in):
        x = torch.stack([y_in, v_in, u_in], dim=-1)
        return self.net(x).squeeze(-1) * acc_scale

torch.manual_seed(42)
model = PINN_MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Prepare training tensors
y_train_t = torch.tensor(y[:n_train], dtype=torch.float32)
v_train_t = torch.tensor(v[:n_train], dtype=torch.float32)
u_train_t = torch.tensor(u[:n_train], dtype=torch.float32)

print("Starting fast training...")
batch_size = 32
rollout_len = 64
n_epochs = 400

for epoch in range(n_epochs):
    optimizer.zero_grad()
    max_start = n_train - rollout_len - 1
    start_indices = np.random.randint(0, max_start, size=batch_size)
    
    y_batch = y_train_t[start_indices]
    v_batch = v_train_t[start_indices]
    
    segment_loss = 0.0
    for step_i in range(rollout_len):
        curr_indices = start_indices + step_i
        u_curr = u_train_t[curr_indices]
        u_next = u_train_t[curr_indices + 1]
        
        # RK4 step
        a1 = model(y_batch / y_scale, v_batch / v_scale, u_curr)
        dy1 = v_batch
        dv1 = a1
        
        y2 = y_batch + 0.5 * dt * dy1
        v2 = v_batch + 0.5 * dt * dv1
        u2 = 0.5 * (u_curr + u_next)
        a2 = model(y2 / y_scale, v2 / v_scale, u2)
        dy2 = v2
        dv2 = a2
        
        y3 = y_batch + 0.5 * dt * dy2
        v3 = v_batch + 0.5 * dt * dv2
        a3 = model(y3 / y_scale, v3 / v_scale, u2)
        dy3 = v3
        dv3 = a3
        
        y4 = y_batch + dt * dy3
        v4 = v_batch + dt * dv3
        a4 = model(y4 / y_scale, v4 / v_scale, u_next)
        dy4 = v4
        dv4 = a4
        
        y_next = y_batch + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        v_next = v_batch + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        target_y = y_train_t[curr_indices + 1]
        segment_loss = segment_loss + torch.mean((y_next - target_y) ** 2)
        
        y_batch = y_next
        v_batch = v_next
        
    loss = segment_loss / rollout_len
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")

# Full free run evaluation
model.eval()
with torch.no_grad():
    y_pred = np.zeros_like(y)
    v_pred = np.zeros_like(v)
    y_pred[0] = y[0]
    v_pred[0] = v[0]
    
    y_curr = torch.tensor(y[0], dtype=torch.float32)
    v_curr = torch.tensor(v[0], dtype=torch.float32)
    
    for i in range(len(y) - 1):
        u_curr = torch.tensor(u[i], dtype=torch.float32)
        u_next = torch.tensor(u[i+1], dtype=torch.float32)
        
        a1 = model(y_curr / y_scale, v_curr / v_scale, u_curr)
        dy1 = v_curr
        dv1 = a1
        
        y2 = y_curr + 0.5 * dt * dy1
        v2 = v_curr + 0.5 * dt * dv1
        u2 = 0.5 * (u_curr + u_next)
        a2 = model(y2 / y_scale, v2 / v_scale, u2)
        dy2 = v2
        dv2 = a2
        
        y3 = y_curr + 0.5 * dt * dy2
        v3 = v_curr + 0.5 * dt * dv2
        a3 = model(y3 / y_scale, v3 / v_scale, u2)
        dy3 = v3
        dv3 = a3
        
        y4 = y_curr + dt * dy3
        v4 = v_curr + dt * dv3
        a4 = model(y4 / y_scale, v4 / v_scale, u_next)
        dy4 = v4
        dv4 = a4
        
        y_next = y_curr + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        v_next = v_curr + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        y_pred[i+1] = y_next.item()
        v_pred[i+1] = v_next.item()
        y_curr = y_next
        v_curr = v_next

train_rmse = float(np.sqrt(np.mean((y_pred[:n_train] - y[:n_train]) ** 2)))
test_rmse = float(np.sqrt(np.mean((y_pred[n_train:] - y[n_train:]) ** 2)))

print(f"RESULT mlp_fast TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

# Save plot
plt.figure(figsize=(10, 4))
plt.plot(y[n_train:], label='Measured y')
plt.plot(y_pred[n_train:], label='Predicted y (Free Run)')
plt.legend()
plt.title(f"Silverbox Test Segment Free Run (RMSE: {test_rmse:.5f} V)")
plt.xlabel("Test Sample Index")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.savefig("test_plot.png")
plt.close()

res = {"model": "mlp_fast", "train_rmse": train_rmse, "test_rmse": test_rmse}
with open("results_mlp_fast.json", "w") as f:
    json.dump(res, f)
