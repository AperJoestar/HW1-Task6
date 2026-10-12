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
u = u_full[start_idx : start_idx + length] - np.mean(u_full[start_idx : start_idx + length])
y = y_full[start_idx : start_idx + length] - np.mean(y_full[start_idx : start_idx + length])

train_len = 3072
dt = 1.0 / 610.35

v = np.zeros_like(y)
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt

acc = np.zeros_like(y)
acc[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / (dt**2)
acc[0] = (y[2] - 2*y[1] + y[0]) / (dt**2)
acc[-1] = (y[-1] - 2*y[-2] + y[-3]) / (dt**2)

sigma_y = 0.055
sigma_v = 21.5
sigma_u = 0.022
sigma_acc = 8700.0

class AccNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, y_s, v_s, u_s):
        y_s, v_s, u_s = torch.broadcast_tensors(y_s, v_s, u_s)
        inp = torch.stack([y_s, v_s, u_s], dim=-1)
        return self.net(inp).squeeze(-1) * sigma_acc

model = AccNet()

y_t_tensor = torch.tensor(y[:train_len], dtype=torch.float32)
v_t_tensor = torch.tensor(v[:train_len], dtype=torch.float32)
u_t_tensor = torch.tensor(u[:train_len], dtype=torch.float32)
acc_t_tensor = torch.tensor(acc[:train_len], dtype=torch.float32)

optimizer = optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred_acc = model(y_t_tensor / sigma_y, v_t_tensor / sigma_v, u_t_tensor / sigma_u)
    loss = criterion(pred_acc, acc_t_tensor)
    loss.backward()
    optimizer.step()

def rk4_step(y_curr, v_curr, u_curr, u_next, dt):
    u_half = 0.5 * (u_curr + u_next)
    a1 = model(y_curr / sigma_y, v_curr / sigma_v, u_curr / sigma_u)
    dy1 = v_curr
    dv1 = a1
    
    y2 = y_curr + 0.5 * dt * dy1
    v2 = v_curr + 0.5 * dt * dv1
    a2 = model(y2 / sigma_y, v2 / sigma_v, u_half / sigma_u)
    dy2 = v2
    dv2 = a2
    
    y3 = y_curr + 0.5 * dt * dy2
    v3 = v_curr + 0.5 * dt * dv2
    a3 = model(y3 / sigma_y, v3 / sigma_v, u_half / sigma_u)
    dy3 = v3
    dv3 = a3
    
    y4 = y_curr + dt * dy3
    v4 = v_curr + dt * dv3
    a4 = model(y4 / sigma_y, v4 / sigma_v, u_next / sigma_u)
    dy4 = v4
    dv4 = a4
    
    y_next = y_curr + (dt / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
    v_next = v_curr + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
    return y_next, v_next

def free_run(u_signal, init_y, init_v):
    preds = [init_y]
    curr_y = torch.tensor(init_y, dtype=torch.float32)
    curr_v = torch.tensor(init_v, dtype=torch.float32)
    with torch.no_grad():
        for i in range(len(u_signal) - 1):
            u_curr = torch.tensor(u_signal[i], dtype=torch.float32)
            u_next = torch.tensor(u_signal[i+1], dtype=torch.float32)
            curr_y, curr_v = rk4_step(curr_y.unsqueeze(0), curr_v.unsqueeze(0), u_curr.unsqueeze(0), u_next.unsqueeze(0), dt)
            preds.append(curr_y.item())
    return np.array(preds)

pred_full = free_run(u, y[0], v[0])

train_rmse = np.sqrt(np.mean((pred_full[:train_len] - y[:train_len])**2))
test_rmse = np.sqrt(np.mean((pred_full[train_len:] - y[train_len:])**2))

print(f"RESULT arch4 TRAIN_RMSE: {train_rmse:.6f} TEST_RMSE: {test_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

results = {
    "architecture": "arch4",
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse)
}
with open("results_arch4.json", "w") as f:
    json.dump(results, f)

plt.figure(figsize=(10, 4))
test_time = np.arange(1024) * dt
plt.plot(test_time, y[train_len:], label='Measured y', color='black', alpha=0.7)
plt.plot(test_time, pred_full[train_len:], label='Predicted y (Free Run)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Silverbox Test Segment Free Run (RMSE: {test_rmse:.5f} V)')
plt.legend()
plt.grid(True)
plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
plt.close()
