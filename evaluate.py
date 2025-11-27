# evaluate.py
import os
import torch
import numpy as np
import json
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import MastitisDetectionEnv
import torch.nn as nn

# -------------------------------
# REINFORCE Policy Network
# -------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

# -------------------------------
# Evaluate SB3 Models
# -------------------------------
def evaluate_sb3_model(model_path, alg='ppo', episodes=200):
    env = MastitisDetectionEnv()
    if alg.lower() == 'ppo':
        model = PPO.load(model_path)
    elif alg.lower() == 'a2c':
        model = A2C.load(model_path)
    elif alg.lower() == 'dqn':
        model = DQN.load(model_path)
    else:
        raise ValueError("Unsupported algorithm")

    rewards = []
    correct = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(int(action))
        rewards.append(reward)
        if info.get('correct'):
            correct += 1

    avg_reward = np.mean(rewards)
    accuracy = correct / episodes
    print(f"{alg.upper()} -> avg reward: {avg_reward:.3f}, accuracy: {accuracy:.3f}")
    return avg_reward

# -------------------------------
# Evaluate REINFORCE
# -------------------------------
def evaluate_reinforce(model_path, episodes=200):
    env = MastitisDetectionEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNet(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    rewards = []
    correct = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = policy(obs_v)
        action = torch.argmax(probs, dim=1).item()
        _, reward, _, _, info = env.step(action)
        rewards.append(reward)
        if info.get("correct"):
            correct += 1

    avg_reward = np.mean(rewards)
    accuracy = correct / episodes
    print(f"REINFORCE -> avg reward: {avg_reward:.3f}, accuracy: {accuracy:.3f}")
    return avg_reward

# -------------------------------
# Main evaluation
# -------------------------------
if __name__ == "__main__":
    results_folder = "results/logs"
    os.makedirs(results_folder, exist_ok=True)

    # Folder mapping for runs and best-of-10
    models_info = {
        "dqn": {
            "base": "models/dqn",
            "run_prefix": "run_",
            "run_file": "best_model.zip",
            "best_model": "models/dqn/best_model/best_model.zip"
        },
        "ppo": {
            "base": "models/pg/ppo",
            "run_prefix": "ppo_run_",
            "run_file": "final.zip",
            "best_model": "models/pg/ppo/ppo_best_model/best_model.zip"
        },
        "a2c": {
            "base": "models/pg/a2c",
            "run_prefix": "a2c_run_",
            "run_file": "final.zip",
            "best_model": "models/pg/a2c/a2c_best_model/best_model.zip"
        },
        "reinforce": {
            "base": "models/pg/reinforce",
            "run_prefix": "reinforce_run_",
            "run_file": "policy.pt",
            "best_model": "models/pg/reinforce/reinforce_best_model/policy.pt"
        }
    }

    for alg, info in models_info.items():
        avg_rewards_list = []
        print(f"\n--- Evaluating {alg.upper()} 10 runs ---")
        for run in range(1, 11):
            model_path = os.path.join(info["base"], f"{info['run_prefix']}{run}", info["run_file"])
            if os.path.exists(model_path):
                if alg == "reinforce":
                    avg_reward = evaluate_reinforce(model_path, episodes=200)
                else:
                    avg_reward = evaluate_sb3_model(model_path, alg=alg, episodes=200)
                avg_rewards_list.append(avg_reward)
            else:
                print(f"{alg.upper()} run {run} not found at {model_path}")
                avg_rewards_list.append(None)

        # Evaluate best-of-10
        best_model_path = info["best_model"]
        if os.path.exists(best_model_path):
            print(f"\n--- Evaluating {alg.upper()} BEST-OF-10 ---")
            if alg == "reinforce":
                best_reward = evaluate_reinforce(best_model_path, episodes=200)
            else:
                best_reward = evaluate_sb3_model(best_model_path, alg=alg, episodes=200)
        else:
            print(f"{alg.upper()} best-of-10 model not found at {best_model_path}")
            best_reward = None

        # Save results
        save_path = os.path.join(results_folder, f"{alg}_eval.json")
        data = {
            "runs": avg_rewards_list,
            "best_of_10": best_reward
        }
        with open(save_path, "w") as f:
            json.dump(data, f)

    print("\nEvaluation complete. Results saved in 'results/logs/'")
