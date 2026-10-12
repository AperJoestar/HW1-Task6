import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Load data and preprocess
mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

# Excerpt [42650 : 46746] (length 4096)
start_idx = 42650
length = 4096
u_excerpt = u_full[start_idx : start_idx + length]
y_excerpt = y_full[start_idx : start_idx + length]

# Subtract mean on excerpt
u_mean = np.mean(u_excerpt)
y_mean = np.mean(y_excerpt)
u = u_excerpt - u_mean
y = y_excerpt - y_mean

# Train/test split
train_len = 3072
test_len = 1024
# train: 0 to 3072, test: 3072 to 4096
dt = 1.0 / 610.35

# Compute scaling factors
sy = np.std(y[:train_len])
# velocity via finite differences
v_train = np.gradient(y[:train_len], dt)
sv = np.std(v_train)
su = np.std(u[:train_len])
sa = np.std(np.gradient(v_train, dt))

print(f"Scaling: sy={sy:.4f}, sv={sv:.4f}, su={su:.4f}, sa={sa:.4f}")

device = torch.device('cpu')

# Convert to tensors
u_t = torch.tensor(u, dtype=torch.float32, device=device)
y_t = torch.tensor(y, dtype=torch.float32, device=device)

# 2. Define Network
class PINNAcceleration(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, y_val, v_val, u_val):
        # inputs scaled
        inp = torch.stack([y_val / sy, v_val / sv, u_val / su], dim=-1)
        out = self.net(inp) * sa
        return out.squeeze(-1)

model = PINNAcceleration().to(device)

# 3. RK4 Integrator
def rk4_step(y_curr, v_curr, u_curr, u_next, dt):
    # u_curr and u_next for half step interpolation
    u_half = 0.5 * (u_curr + u_next)
    
    # k1
    k1_y = v_curr
    k1_v = model(y_curr, v_curr, u_curr)
    
    # k2
    y2 = y_curr + 0.5 * dt * k1_y
    v2 = v_curr + 0.5 * dt * k1_v
    k2_y = v2
    k2_v = model(y2, v2, u_half)
    
    # k3
    y3 = y_curr + 0.5 * dt * k2_y
    v3 = v_curr + 0.5 * dt * k2_v
    k3_y = v3
    k3_v = model(y3, v3, u_half)
    
    # k4
    y4 = y_curr + dt * k3_y
    v4 = v_curr + dt * k3_v
    k4_y = v4
    k4_v = model(y4, v4, u_next)
    
    y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_next = v_curr + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return y_next, v_next

def simulate_rollout(y0, v0, u_seq, rollout_len):
    # u_seq is tensor of shape (batch, rollout_len+1) or similar
    # Here we do batch simulation of trajectories
    batch_size = y0.shape[0]
    y_traj = [y0]
    v_curr = v0
    y_curr = y0
    
    for step in range(rollout_len):
        u_curr = u_seq[:, step]
        u_next = u_seq[:, step + 1]
        y_curr, v_curr = rk4_step(y_curr, v_curr, u_curr, u_next, dt)
        y_traj.append(y_curr)
        
    return torch.stack(y_traj, dim=1) # (batch, rollout_len+1)

# 4. Training
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Initial velocity for start sample
# Estimate initial v using central diff on training data
v_full_est = np.zeros_like(y)
v_full_est[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
v_full_est[0] = (y[1] - y[0]) / dt
v_full_est[-1] = (y[-1] - y[-2]) / dt
v_t_all = torch.tensor(v_full_est, dtype=torch.float32, device=device)

# Curriculum training: gradually increase rollout length
rollout_lengths = [32, 64, 128]
steps_per_stage = 500

torch.manual_seed(42)
np.random.seed(42)

step_count = 0
for L in rollout_lengths:
    print(f"Training with rollout length L={L} for {steps_per_stage} steps...")
    for epoch in range(steps_per_stage):
        step_count += 1
        # Sample random starting indices in training data (ensuring space for L steps)
        max_start = train_len - L - 1
        start_indices = np.random.randint(0, max_start, size=64)
        
        y0_batch = y_t[start_indices]
        v0_batch = v_t_all[start_indices]
        
        # Extract u sequences of length L+1
        u_batch_list = []
        target_y_list = []
        for idx in start_indices:
            u_batch_list.append(u_t[idx : idx + L + 1])
            target_y_list.append(y_t[idx : idx + L + 1])
            
        u_batch = torch.stack(u_batch_list, dim=0)
        target_y = torch.stack(target_y_list, dim=0)
        
        pred_y = simulate_rollout(y0_batch, v0_batch, u_batch, L)
        
        loss = torch.mean((pred_y - target_y) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"  Step {step_count} (epoch {epoch+1}), Loss: {loss.item():.6f}")

# 5. Full Free Run Evaluation
model.eval()
with torch.no_grad():
    # Free run over entire excerpt (train + test)
    y_free = [y_t[0].item()]
    curr_y = y_t[0]
    curr_v = v_t_all[0]
    
    for i in range(length - 1):
        uc = u_t[i]
        un = u_t[i+1]
        curr_y, curr_v = rk4_step(curr_y.unsqueeze(0), curr_v.unsqueeze(0), uc.unsqueeze(0), un.unsqueeze(0), dt)
        curr_y = curr_y.squeeze(0)
        curr_v = curr_v.squeeze(0)
        y_free.append(curr_y.item())
        
    y_free = np.array(y_free)

train_pred = y_free[:train_len]
test_pred = y_free[train_len:]

train_rmse = np.sqrt(np.mean((train_pred - y[:train_len]) ** 2))
test_rmse = np.sqrt(np.mean((test_pred - y[train_len:]) ** 2))

baseline_test = np.sqrt(np.mean(y[train_len:] ** 2))
print(f"Baseline test RMS (zero prediction): {baseline_test:.5f}")
print(f"RESULT arch1 TRAIN_RMSE: {train_rmse:.5f} TEST_RMSE: {test_rmse:.5f}")
print(f"TEST_RMSE: {test_rmse:.5f}")

# Save plot
plt.figure(figsize=(10, 4))
time_axis = np.arange(test_len) * dt
plt.plot(time_axis, y[train_len:], label='Measured (Test)', color='black', alpha=0.7)
plt.plot(time_axis, test_pred, label='Predicted Free Run', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Output Voltage y(t)')
plt.title('Silverbox PINN Free Run on Test Segment (arch1)')
plt.legend()
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()
