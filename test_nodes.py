from nodes import general_node, scientist_node, executor_node

s = {"messages": [{"role": "user", "text": "Fit dx/dt = -a*x to samples of x(t)=exp(-2t), t in [0,2]."}],
     "instruction": "Write the plan.", "next_role": "", "decisions": 0}
s["messages"] += scientist_node(s)["messages"]
print(s["messages"][-1]["text"][:800])

s["instruction"] = "Implement the plan and report the RMSE."
s["messages"] += executor_node(s)["messages"]
print(s["messages"][-1]["text"])

s["instruction"] = "What does RMSE measure here?"
print(general_node(s)["messages"][-1]["text"][:400])