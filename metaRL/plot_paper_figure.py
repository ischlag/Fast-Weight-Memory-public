import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sns.set(style="darkgrid")

log_folder = sys.argv[1]
ymax = 0
ymin = 9999

# get the rewards from all logs
train_logs = []
with os.scandir(log_folder) as entries:
  for entry in sorted(entries, key=lambda x: x.name):
    if entry.name[:12] == "trainrewards":
      if "lstm" in entry.name and "h512" in entry.name:
        name = "LSTM-512 (x71)"
      elif "lstm" in entry.name and "h1024" in entry.name:
        name = "LSTM-1024 (x307)"
      elif "lstm" in entry.name and "h2048" in entry.name:
        name = "LSTM-2048 (x1207)"
      elif "lstm" in entry.name and "h4096" in entry.name:
        name = "LSTM-4096 (x4814)"
      elif "fwm" in entry.name:
        name = "FWM-32-16"
      train_logs.append((np.loadtxt(entry.path)[:2500], name))

for l in train_logs:
  if l[0].max() > ymax:
    ymax = l[0].max()
  if l[0].min() < ymin:
    ymin = l[0].min()


test_logs = []
with os.scandir(log_folder) as entries:
  for entry in sorted(entries, key=lambda x: x.name):
    if entry.name[:11] == "testrewards":
      if "lstm" in entry.name and "h512" in entry.name:
        name = "LSTM-512 (x71)"
      elif "lstm" in entry.name and "h1024" in entry.name:
        name = "LSTM-1024 (x307)"
      elif "lstm" in entry.name and "h2048" in entry.name:
        name = "LSTM-2048 (x1207)"
      elif "lstm" in entry.name and "h4096" in entry.name:
        name = "LSTM-4096 (x4814)"
      elif "fwm" in entry.name:
        name = "FWM-32-16"
      test_logs.append((np.loadtxt(entry.path)[:2500], name))

for l in test_logs:
  if l[0].max() > ymax:
    ymax = l[0].max()
  if l[0].min() < ymin:
    ymin = l[0].min()

ymax = int((ymax / 10.0) + 0.5) * 10 + 10
ymin = int((ymin / 10.0) - 0.5) * 10
count = (ymax - ymin) / 10 + 1

train_logs.insert(1, train_logs.pop(-1))
test_logs.insert(1, test_logs.pop(-1))

# plot train
def plot_to_file(title, train_logs, test_logs):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
  lines = []
  labels = []
  ax1.set_title("train environments")
  ax1.set(xlabel="steps x 50", ylabel="avg total reward")
  for e in train_logs:
    line_plot = ax1.plot(range(len(e[0])),e[0])[0]
    ax1.set_ylim([ymin, ymax])
    #ax1.set_yticks(np.linspace(ymin, ymax, count))
    lines.append(line_plot)
    labels.append("{}".format(e[1]))

  ax2.set_title("test environments")
  ax2.set(xlabel="steps x 50")

  for e in test_logs:
    line_plot = ax2.plot(range(len(e[0])),e[0])[0]
    ax2.set_ylim([ymin, ymax])
    #ax2.set_yticks(np.linspace(ymin, ymax, count))
    lines.append(line_plot)

  #_ = ax3.axis("off")
  fig.legend(lines, labels)
  plt.savefig("RL_paper_plot.png", bbox_inches='tight')

plot_to_file(log_folder, train_logs, test_logs)
