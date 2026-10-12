import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

start_idx = 42650
length = 4096
u = u_full[start_idx:start_idx+length]
y = y_full[start_idx:start_idx+length]

u = u - np.mean(u)
y = y - np.mean(y)

train_len = 3072
test_len = 1024

u_train = u[:train_len]
y_train = y[:train_len]

sigma_y = 0.055
sigma_v = 21.5
sigma_u = 0.022
dt = 1.0 / 610.35

v_train = np.zeros_like(y_train)
v_train[0] = (y_train[1] - y_train[0]) / dt
v_train[1:] = (y_train[1:] - y_train[:-1]) / dt

class AccelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, y_s, v_s, u_s):
        if y_s.ndim == 0:
            y_s = y_s.unsqueeze(0)
        if v_s.ndim == 0:
            v_s = v_s.unsqueeze(0)
        if u_s.ndim == 0:
            u_s = u_s.unsqueeze(0)
        inp = torch.stack([y_s, v_s, u_s], dim=-1)
        return self.net(inp).squeeze(-1)

model = AccelMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # smaller LR

def rk4_step_correct(y_t, v_t, u_t, u_next, dt):
    u_mid = 0.5 * (u_t + u_next)
    
    a1 = model(y_t / sigma_y, v_t / sigma_v, u_t / sigma_u)
    k1_y = v_t
    k1_v = a1
    
    y2 = y_t + 0.5 * dt * k1_y
    v2 = v_t + 0.5 * dt * k1_v
    a2 = model(y2 / sigma_y, v2 / sigma_v, u_mid / sigma_u)
    k2_y = v2
    k2_v = a2
    
    y3 = y_t + 0.5 * dt * k2_y
    v3 = v_t + 0.5 * dt * k2_v
    a3 = model(y3 / sigma_y, v3 / sigma_v, u_mid / sigma_u)
    k3_y = v3
    k3_v = a3
    
    y4 = y_t + dt * k3_y
    v4 = v_t + dt * k3_v
    a4 = model(y4 / sigma_y, v4 / sigma_v, u_next / sigma_u)
    k4_y = v4
    k4_v = a4
    
    y_next = y_t + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_next = v_t + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return y_next, v_next

rollout_len = 16 # shorter rollout
y_tr_tensor = torch.tensor(y_train, dtype=torch.float32)
v_tr_tensor = torch.tensor(v_train, dtype=torch.float32)
u_tr_tensor = torch.tensor(u_train, dtype=torch.float32)

start_indices = np.arange(0, train_len - rollout_len, 16)
epochs = 200
batch_size_segments = 64

for epoch in range(epochs):
    model.train()
    np.random.shuffle(start_indices)
    total_loss = 0.0
    n_batches = 0
    
    for i_start in range(0, len(start_indices), batch_size_segments):
        batch_idx = start_indices[i_start:i_start+batch_size_segments]
        if len(batch_idx) == 0: continue
        
        curr_y = y_tr_tensor[batch_idx]
        curr_v = v_tr_tensor[batch_idx]
        
        loss = 0.0
        for step in range(rollout_len):
            idx_t = batch_idx + step
            u_t = u_tr_tensor[idx_t]
            u_next = u_tr_tensor[idx_t + 1]
            
            next_y, next_v = rk4_step_correct(curr_y, curr_v, u_t, u_next, dt)
            target_y = y_tr_tensor[idx_t + 1]
            loss = loss + torch.mean((next_y - target_y) ** 2)
            curr_y, curr_v = next_y, next_v
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.6f}")

def free_run(u_full_seq, y_full_seq, init_y, init_v):
    model.eval()
    preds = [init_y]
    curr_y = torch.tensor(init_y, dtype=torch.float32)
    curr_v = torch.tensor(init_v, dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(len(u_full_seq) - 1):
            u_t = torch.tensor(u_full_seq[i], dtype=torch.float32)
            u_next = torch.tensor(u_full_seq[i+1], dtype=torch.float32)
            curr_y, curr_v = rk4_step_correct(curr_y, curr_v, u_t, u_next, dt)
            preds.append(curr_y.item())
            
    return np.array(preds)

init_y = y[0]
init_v = (y[1] - y[0]) / dt

pred_full = free_run(u, y, init_y, init_v)

train_pred = pred_full[:train_len]
test_pred = pred_full[train_len:]
train_true = y[:train_len]
test_true = y[train_len:]

train_rmse = np.sqrt(np.mean((train_pred - train_true) ** 2))
test_rmse = np.sqrt(np.mean((test_pred - test_true) ** 2))

print(f"RESULT arch3 TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

results = {
    "arch": "arch3",
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse)
}
with open("results_arch3.json", "w") as f:
    json.dump(results, f)

plt.figure(figsize=(10, 4))
plt.plot(test_true, label='Measured (Test)', color='black')
plt.plot(test_pred, label='Predicted (Free Run)', color='red', linestyle='--')
plt.legend()
plt.title(f"Silverbox Test Free Run (RMSE: {test_rmse:.4f})")
plt.xlabel("Test Sample Index")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.savefig("test_plot.png")
plt.close()
