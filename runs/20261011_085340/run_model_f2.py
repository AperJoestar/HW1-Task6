import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load Data and Preprocess
mat_data = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat_data['V1'].flatten()
y_full = mat_data['V2'].flatten()

# Excerpt slice [42650 : 46746] -> total 4096 samples
start_idx = 42650
end_idx = 46746
u_excerpt = u_full[start_idx:end_idx]
y_excerpt = y_full[start_idx:end_idx]

u_mean = np.mean(u_excerpt)
y_mean = np.mean(y_excerpt)

u = u_excerpt - u_mean
y = y_excerpt - y_mean

fs = 610.35
dt = 1.0 / fs

train_len = 3072
test_len = 1024

y_train = y[:train_len]
u_train = u[:train_len]

y_std = float(np.std(y_train))
v_train_fd = np.diff(y_train) / dt
v_std = float(np.std(v_train_fd))
u_std = float(np.std(u_train))

print(f"Data stats: y_std={y_std:.4f}, v_std={v_std:.4f}, u_std={u_std:.4f}")

y_tensor = torch.tensor(y, dtype=torch.float32)
u_tensor = torch.tensor(u, dtype=torch.float32)

# 2. Neural Network Architecture (Model F2: 3 hidden layers of 32 units with Tanh)
class NeuralAccelerationModel(nn.Module):
    def __init__(self, y_scale, v_scale, u_scale):
        super().__init__()
        self.y_scale = y_scale
        self.v_scale = v_scale
        self.u_scale = u_scale
        
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, y_val, v_val, u_val):
        y_in = y_val / self.y_scale
        v_in = v_val / self.v_scale
        u_in = u_val / self.u_scale
        inputs = torch.stack([y_in, v_in, u_in], dim=-1)
        return self.net(inputs).squeeze(-1)

model = NeuralAccelerationModel(y_std, v_std, u_std)

# 3. RK4 Integration Scheme with Interpolated Input
def rk4_step(y_curr, v_curr, t_indices, u_full_tensor, model, dt):
    def get_u(idx_float):
        idx_lo = torch.clamp(torch.floor(idx_float).long(), 0, len(u_full_tensor) - 1)
        idx_hi = torch.clamp(idx_lo + 1, 0, len(u_full_tensor) - 1)
        frac = idx_float - idx_lo.float()
        return (1.0 - frac) * u_full_tensor[idx_lo] + frac * u_full_tensor[idx_hi]

    u0 = get_u(t_indices)
    a1 = model(y_curr, v_curr, u0)
    dy1 = v_curr
    dv1 = a1

    u1 = get_u(t_indices + 0.5)
    y2 = y_curr + 0.5 * dt * dy1
    v2 = v_curr + 0.5 * dt * dv1
    a2 = model(y2, v2, u1)
    dy2 = v2
    dv2 = a2

    u2 = get_u(t_indices + 0.5)
    y3 = y_curr + 0.5 * dt * dy2
    v3 = v_curr + 0.5 * dt * dv2
    a3 = model(y3, v3, u2)
    dy3 = v3
    dv3 = a3

    u3 = get_u(t_indices + 1.0)
    y4 = y_curr + dt * dy3
    v4 = v_curr + dt * dv3
    a4 = model(y4, v4, u3)
    dy4 = v4
    dv4 = a4

    y_next = y_curr + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
    v_next = v_curr + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
    return y_next, v_next

# 4. Training Strategy via Rollouts (reduced steps and batch size for speed)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
rollout_len = 64
num_steps = 1000
batch_size = 64

print("Starting training Model F2...")
for step in range(1, num_steps + 1):
    max_start = train_len - rollout_len - 1
    start_indices = torch.randint(0, max_start, (batch_size,))
    
    y_batch = y_tensor[start_indices]
    v_batch = (y_tensor[start_indices + 1] - y_tensor[start_indices]) / dt
    
    loss = 0.0
    curr_y = y_batch
    curr_v = v_batch
    
    for i in range(rollout_len):
        t_idx = start_indices.float() + float(i)
        target_y = y_tensor[start_indices + i + 1]
        curr_y, curr_v = rk4_step(curr_y, curr_v, t_idx, u_tensor, model, dt)
        loss = loss + torch.mean((curr_y - target_y) ** 2)
        
    loss = loss / rollout_len
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 250 == 0 or step == 1:
        print(f"Step {step}/{num_steps}, Loss: {loss.item():.6f}")

# 5. Free Run Evaluation over entire excerpt (4096 samples)
model.eval()
with torch.no_grad():
    y_pred = np.zeros(len(y_tensor))
    v_pred = np.zeros(len(y_tensor))
    
    y_pred[0] = y_tensor[0].item()
    v_pred[0] = (y_tensor[1].item() - y_tensor[0].item()) / dt
    
    curr_y = torch.tensor(y_pred[0], dtype=torch.float32)
    curr_v = torch.tensor(v_pred[0], dtype=torch.float32)
    
    for i in range(len(y_tensor) - 1):
        t_idx = torch.tensor(float(i), dtype=torch.float32)
        curr_y, curr_v = rk4_step(curr_y, curr_v, t_idx, u_tensor, model, dt)
        y_pred[i+1] = curr_y.item()
        v_pred[i+1] = curr_v.item()

train_pred = y_pred[:train_len]
test_pred = y_pred[train_len:]

train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
y_test_true = y[train_len:]
test_rmse = np.sqrt(np.mean((test_pred - y_test_true) ** 2))

print(f"RESULT ModelF2 TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

results = {
    "model": "ModelF2",
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse)
}
with open("results_ModelF2.json", "w") as f:
    json.dump(results, f)

plt.figure(figsize=(10, 4))
time_axis = np.arange(test_len) * dt
plt.plot(time_axis, y_test_true, label='Measured $y(t)$', color='black', alpha=0.7)
plt.plot(time_axis, test_pred, label='Predicted Free Run y(t)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Silverbox Test Segment: Measured vs Free-Run Predicted Output (Model F2)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()
