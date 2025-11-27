# visualize_training.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_FOLDER = "results/logs"
PLOT_FOLDER = "results/plots"
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Load results
def load_eval(alg):
    path = os.path.join(RESULTS_FOLDER, f"{alg}_eval.json")
    with open(path, "r") as f:
        data = json.load(f)
    runs = data.get("runs", [])
    best = data.get("best_of_10", None)
    return runs, best

# Plot cumulative rewards
def plot_cumulative_rewards(all_data):
    plt.figure(figsize=(12, 6))
    for alg, (runs, best) in all_data.items():
        if best is not None:
            plt.plot(np.cumsum([best]*len(runs)), label=f"{alg.upper()} Best")
        for i, r in enumerate(runs):
            if r is not None:
                plt.plot(np.cumsum([r]*len(runs)), linestyle="--", alpha=0.3)
    plt.title("Cumulative Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_FOLDER, "cumulative_rewards.png"))
    plt.show()

# Plot episode-wise rewards
def plot_episode_rewards(all_data):
    plt.figure(figsize=(12, 6))
    for alg, (runs, best) in all_data.items():
        mean_reward_per_run = [r if r is not None else 0 for r in runs]
        plt.plot(range(1, len(runs)+1), mean_reward_per_run, marker='o', label=alg.upper())
    plt.title("Mean Reward per Run")
    plt.xlabel("Run")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_FOLDER, "mean_rewards_per_run.png"))
    plt.show()

# Load data for all methods
methods = ["dqn", "ppo", "a2c", "reinforce"]
all_data = {m: load_eval(m) for m in methods}

# Generate plots
plot_cumulative_rewards(all_data)
plot_episode_rewards(all_data)

# Optional: subplot comparison
def subplot_comparison(all_data):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    for i, (alg, (runs, best)) in enumerate(all_data.items()):
        rewards = [r if r is not None else 0 for r in runs]
        axs[i].plot(range(1, len(runs)+1), rewards, marker='o')
        if best is not None:
            axs[i].axhline(best, color='r', linestyle='--', label="Best-of-10")
        axs[i].set_title(alg.upper())
        axs[i].set_xlabel("Run")
        axs[i].set_ylabel("Mean Reward")
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "subplot_comparison.png"))
    plt.show()

subplot_comparison(all_data)

print(f"Plots saved in '{PLOT_FOLDER}' folder")
