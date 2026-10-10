from orchestrator import orchestrator_node, has_real_rmse

user = {"role": "user", "text": "Fit dx/dt = -a*x to samples of x(t)=exp(-2t), t in [0,2]."}

# 1. Guard logic, no model call
fake = {"role": "executor", "text": "Done, RMSE is 0.01"}
real = {"role": "executor", "text": "ok\n\n--- script output ---\nRUN fit.py\nexit_code=0\nSTDOUT:\nTEST_RMSE: 1.5e-03\nSTDERR:\n"}
failed = {"role": "executor", "text": "x\n\n--- script output ---\nRUN fit.py\nexit_code=1\nSTDOUT:\nSTDERR:\nTraceback..."}
print("fake:", has_real_rmse([user, fake]), "| real:", has_real_rmse([user, real]), "| failed:", has_real_rmse([user, failed]))

# 2. Fresh problem: expect a sensible first choice
s = {"messages": [user], "instruction": "", "next_role": "", "decisions": 0}
r = orchestrator_node(s)
print(r["messages"][0]["text"])

# 3. Question beside the fit
s2 = {"messages": [user, {"role": "user", "text": "What does the RMSE number mean?"}],
      "instruction": "", "next_role": "", "decisions": 1}
print(orchestrator_node(s2)["messages"][0]["text"])

# 4. Cap: must finish at 8 decisions
s3 = {"messages": [user], "instruction": "", "next_role": "", "decisions": 8}
print(orchestrator_node(s3)["next_role"])