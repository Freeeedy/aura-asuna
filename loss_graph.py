import matplotlib.pyplot as plt
import re
import numpy as np

log_file = "log_files/loss_logs.txt"

steps = []
train_losses = []
val_losses = []


# example line: [2025-08-28 02:09:28] step: 5000, train loss: 1.101, val loss: 1.108
# \[.*?\] → matches [2025-08-28 02:09:28] (the square brackets and everything inside).
# \s* → optional whitespace after it.
# step: (\d+) → captures the step number.
# train loss: ([0-9.]+) → captures the train loss.
# val loss: ([0-9.]+) → captures the val loss.
pattern = re.compile(r"\[.*?\]\s*step: (\d+), train loss: ([0-9.]+), val loss: ([0-9.]+)")

global_step = 0

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            _, train, val = match.groups()
            steps.append(global_step)
            train_losses.append(float(train))
            val_losses.append(float(val))
            global_step += 1

# convert losses to estimated % correct
train_percent = [np.exp(-l) for l in train_losses]
val_percent   = [np.exp(-l) for l in val_losses]

plt.plot(steps, train_losses, label="Trainloss", color="blue") # train line
plt.plot(steps, val_losses, label="Val loss", color="red") # val line
plt.plot(steps, train_percent, label="Train correct %",color="green")
plt.plot(steps, val_percent, label="Val correct %", color="orange")

plt.title("All losses from most runs") # graph title obv. 
plt.xlabel("Steps (every log entry is one step)") # x-axis label
plt.ylabel("Loss/%") # y-axis label
plt.ylim(0, 4) # makes y-axis 0-1.5
plt.legend() # display the legend (green=val etc.)
plt.grid(True) # display grid to see values better
plt.tight_layout() # adjusts padding so it doesnt overlap
plt.show()
