import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def extract_losses(path):
    losses = []
    with open(path) as f:
        for line in f:
            m = re.search(r"Loss: ([0-9.]+)", line)
            if m:
                losses.append(float(m.group(1)))
    return losses

def extract_accuracies(path):
    accs = []
    with open(path) as f:
        for line in f:
            m = re.search(r"acc = ([0-9.]+)", line)
            if m:
                accs.append(float(m.group(1)))
    return accs

# Task 1 – per-epoch accuracy
task1_epochs   = [1, 2, 3]
task1_accuracy = [0.628158844765343, 0.6498194945848376, 0.6209386281588448]

# Tasks 2a / 2b / 3 – loss curves (node 0 only; all nodes identical)
losses_2a = extract_losses("logs/task2a/task2a-node0.log")
losses_2b = extract_losses("logs/task2b/task2b-node0.log")
losses_3  = extract_losses("logs/task3/task3-node0.log")
steps = list(range(len(losses_2a)))

# Average time per iteration
methods      = ["Gather/Scatter\n(2a)", "All-Reduce\n(2b)", "DDP\n(3)"]
avg_times    = [24.7186, 7.3887, 6.3047]

# Figure 1: Task 1 accuracy per epoch

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(task1_epochs, task1_accuracy, marker="o", linewidth=2, color="#2563eb")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Task 1 – BERT Fine-tuning on RTE (Single Node)")
ax.set_xticks(task1_epochs)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4f}"))
ax.set_ylim(0.60, 0.68)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/task1_accuracy.png", dpi=150)
plt.close()
print("Saved figures/task1_accuracy.png")

# Figure 2: Loss curves Tasks 2a / 2b / 3

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, losses_2a, label="2(a) Gather/Scatter", linewidth=2, color="#2563eb")
ax.plot(steps, losses_2b, label="2(b) All-Reduce",     linewidth=2, color="#dc2626",
        linestyle="--")
ax.plot(steps, losses_3,  label="3 DDP",               linewidth=2, color="#16a34a",
        linestyle=":")
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title("Loss Curve – Node 0 (all methods identical)")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/loss_curve.png", dpi=150)
plt.close()
print("Saved figures/loss_curve.png")

#Figure 3: Average iteration time bar chart

fig, ax = plt.subplots(figsize=(5, 3.5))
colors = ["#2563eb", "#dc2626", "#16a34a"]
bars = ax.bar(methods, avg_times, color=colors, width=0.5)
for bar, t in zip(bars, avg_times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{t:.4f}s", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Avg Time per Iteration (s)")
ax.set_title("Average Iteration Time (excl. first iteration)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/iteration_time.png", dpi=150)
plt.close()
print("Saved figures/iteration_time.png")

# Figure 4: Task 4 – Communication overhead per step (stacked bar)

t4_methods = ["Gather/Scatter", "All-Reduce", "DDP"]

total_step_us  = [78_402_152.192 / 3, 24_896_765.782 / 3, 19_873_801.033 / 3]
total_comm_us  = [67_530_431.503 / 3, 17_485_030.083 / 3, None]  # DDP is overlapped
compute_us     = [t - c for t, c in zip(total_step_us[:2], total_comm_us[:2])]

ar_compute_us  = compute_us[1]  # ~2,470,508 µs
ddp_total_us   = total_step_us[2]
ddp_blocking_comm_us = ddp_total_us - ar_compute_us

comm_us   = [total_comm_us[0], total_comm_us[1], ddp_blocking_comm_us]
comp_us   = [compute_us[0],    compute_us[1],    ar_compute_us]
comm_pct  = [c / t * 100 for c, t in zip(comm_us, total_step_us)]
comp_pct  = [100 - p for p in comm_pct]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(t4_methods))
width = 0.5
bar_comp = ax.bar(x, comp_pct, width, label="Compute", color=["#93c5fd", "#fca5a5", "#86efac"])
bar_comm = ax.bar(x, comm_pct, width, bottom=comp_pct, label="Communication",
                  color=["#2563eb", "#dc2626", "#16a34a"])

for i, (cp, cm) in enumerate(zip(comp_pct, comm_pct)):
    ax.text(i, cp / 2,        f"{cp:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(i, cp + cm / 2,   f"{cm:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(t4_methods)
ax.set_ylabel("Percentage of Step Time (%)")
ax.set_title("Task 4 – Communication vs Compute Overhead per Step\n(DDP: estimated blocking comm after overlap)")
ax.set_ylim(0, 110)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/task4_comm_overhead.png", dpi=150)
plt.close()
print("Saved figures/task4_comm_overhead.png")

# Figure 5: Task 4 – Duration of 3 Profiled Training Steps
gs_steps = [26_357_103.61, 26_138_360.417, 25_906_688.165]
ar_steps = [ 8_448_386.401,  8_114_933.342,  8_333_446.039]
ddp_steps= [ 6_743_720.735,  6_582_409.686,  6_547_670.612]

step_labels = ["Step 1\n(warmup skipped)", "Step 2", "Step 3"]
x = np.arange(len(step_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width, [v/1e6 for v in gs_steps],  width, label="Gather/Scatter", color="#2563eb")
ax.bar(x,         [v/1e6 for v in ar_steps],  width, label="All-Reduce",     color="#dc2626")
ax.bar(x + width, [v/1e6 for v in ddp_steps], width, label="DDP",            color="#16a34a")
ax.set_xticks(x)
ax.set_xticklabels(step_labels)
ax.set_ylabel("Step Duration (s)")
ax.set_title("Task 4 – Duration of 3 Profiled Training Steps")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/task4_step_durations.png", dpi=150)
plt.close()
print("Saved figures/task4_step_durations.png")

print("Done.")
