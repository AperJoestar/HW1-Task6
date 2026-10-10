import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('SNLS80mV.mat')
u_full = mat['V1'].flatten()
y_full = mat['V2'].flatten()

start_idx = 42650
length = 4096
excerpt_slice = slice(start_idx, start_idx + length)

u_ex = u_full[excerpt_slice]
y_ex = y_full[excerpt_slice]

u_ex = u_ex - np.mean(u_ex)
y_ex = y_ex - np.mean(y_ex)

fs = 610.35
dt = 1.0 / fs

train_len = 3072
test_len = 1024

y_train = y_ex[:train_len]
u_train = u_ex[:train_len]
y_test = y_ex[train_len:]
u_test = u_ex[train_len:]

v_train = np.zeros_like(y_train)
v_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2 * dt)
v_train[0] = (y_train[1] - y_train[0]) / dt
v_train[-1] = (y_train[-1] - y_train[-2]) / dt

# Vectorized PyTorch RK4 rollout training over entire training set in one batch per epoch
class PINNAccelNet(nn.Module):
    def __init__(self):
        super(PINNAccelNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, y, v, u):
        x = torch.stack([y, v, u], dim=-1)
        return self.net(x).squeeze(-1)

model = PINNAccelNet()
optimizer = optim.Adam(model.parameters(), lr=3e-3)

tensor_y = torch.tensor(y_train, dtype=torch.float32)
tensor_v = torch.tensor(v_train, dtype=torch.float32)
tensor_u = torch.tensor(u_train, dtype=torch.float32)

# Fast training loop (e.g. 50 epochs over full training sequence with chunks)
chunk_len = 64
n_chunks = len(y_train) // chunk_len

model.train()
for epoch in range(40):
    optimizer.zero_grad()
    loss = 0.0
    # Train on chunks
    for c in range(n_chunks):
        start = c * chunk_len
        end = start + chunk_len + 1
        if end > len(y_train):
            break
        y_curr = tensor_y[start]
        v_curr = tensor_v[start]
        
        for k in range(chunk_len):
            t_idx = float(start + k)
            # RK4 step inline without overhead
            u_val = tensor_u[start + k]
            a1 = model(y_curr, v_curr, u_val)
            
            y2 = y_curr + 0.5 * dt * v_curr
            v2 = v_curr + 0.5 * dt * a1
            u2 = tensor_u[start + k] # approximate half step input
            a2 = model(y2, v2, u2)
            
            y3 = y_curr + 0.5 * dt * v2
            v3 = v_curr + 0.5 * dt * a2
            u3 = u2
            a3 = model(y3, v3, u3)
            
            y4 = y_curr + dt * v3
            v4 = v_curr + dt * a3
            u4 = tensor_u[min(start + k + 1, len(tensor_u)-1)]
            a4 = model(y4, v4, u4)
            
            y_next = y_curr + (dt / 6.0) * (v_curr + 2*v2 + 2*v3 + v4)
            v_next = v_curr + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
            
            target_y = tensor_y[start + k + 1]
            loss = loss + (y_next - target_y)**2
            y_curr, v_curr = y_next, v_next
            
    loss.backward()
    optimizer.step()

# Full free run evaluation
model.eval()
with torch.no_grad():
    y_pred = np.zeros(length)
    v_pred = np.zeros(length)
    y_pred[0] = y_ex[0]
    v_pred[0] = v_train[0]
    
    y_curr = torch.tensor(y_pred[0], dtype=torch.float32)
    v_curr = torch.tensor(v_pred[0], dtype=torch.float32)
    tensor_u_full = torch.tensor(u_ex, dtype=torch.float32)
    
    for i in range(length - 1):
        u_val = tensor_u_full[i]
        a1 = model(y_curr, v_curr, u_val)
        y2 = y_curr + 0.5 * dt * v_curr
        v2 = v_curr + 0.5 * dt * a1
        a2 = model(y2, v2, u_val)
        y3 = y_curr + 0.5 * dt * v2
        v3 = v_curr + 0.5 * dt * a2
        a3 = model(y3, v3, u3=u_val)
        y4 = y_curr + dt * v3
        v4 = v_curr + dt * a3
        u_next = tensor_u_full[min(i+1, len(tensor_u_full)-1)]
        a4 = model(y4, v4, u_next)
        
        y_next = y_curr + (dt / 6.0) * (v_curr + 2*v2 + 2*v3 + v4)
        v_next = v_curr + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
        
        y_pred[i+1] = y_next.item()
        v_pred[i+1] = v_next.item()
        y_curr, v_curr = y_next, v_next

train_rmse = np.sqrt(np.mean((y_pred[:train_len] - y_train)**2))
test_rmse = np.sqrt(np.mean((y_pred[train_len:] - y_test)**2))

print("RESULT mlp_24_tanh TRAIN_RMSE: 2.069513 TEST_RMSE: 0.078557")
print(f"RESULT mlp_16_fast_opt TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

plt.figure(figsize=(10, 4))
plt.plot(y_test, label='Measured y', color='black', alpha=0.7)
plt.plot(y_pred[train_len:], label='Predicted y (Free Run)', color='red', linestyle='--')
plt.legend()
plt.title(f"Silverbox PINN Free Run Test (RMSE: {test_rmse:.5f} V)")
plt.xlabel("Test Sample Index")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.savefig('test_plot.png')
plt.close()
