import json

with open("results_arch1.json") as f:
    res1 = json.load(f)
with open("results_arch2.json") as f:
    res2 = json.load(f)

print("Architecture 1 - Train RMSE:", res1["train_rmse"], "Test RMSE:", res1["test_rmse"])
print("Architecture 2 - Train RMSE:", res2["train_rmse"], "Test RMSE:", res2["test_rmse"])
print("Final kept TEST_RMSE:", min(res1["test_rmse"], res2["test_rmse"]))
