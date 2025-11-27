# train_pg_multiple_runs.py
import sys, os, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import MastitisDetectionEnv

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
# Training Configuration
# -------------------------------
TOTAL_TIMESTEPS = 20000
REINFORCE_EPISODES = 2000
RESULTS_FOLDER = "models/pg"

# -------------------------------
# 10 Hyperparameter sets
# -------------------------------
PPO_HYPERS = [
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 32},
    {"learning_rate": 3e-4, "n_steps": 256, "batch_size": 64},
    {"learning_rate": 1e-3, "n_steps": 128, "batch_size": 32},
    {"learning_rate": 1e-3, "n_steps": 256, "batch_size": 64},
    {"learning_rate": 5e-4, "n_steps": 128, "batch_size": 32},
    {"learning_rate": 5e-4, "n_steps": 256, "batch_size": 64},
    {"learning_rate": 2e-3, "n_steps": 128, "batch_size": 32},
    {"learning_rate": 2e-3, "n_steps": 256, "batch_size": 64},
    {"learning_rate": 1e-4, "n_steps": 128, "batch_size": 32},
    {"learning_rate": 1e-4, "n_steps": 256, "batch_size": 64},
]

A2C_HYPERS = [
    {"learning_rate": 7e-4},
    {"learning_rate": 5e-4},
    {"learning_rate": 1e-3},
    {"learning_rate": 2e-3},
    {"learning_rate": 3e-4},
    {"learning_rate": 1e-4},
    {"learning_rate": 2e-4},
    {"learning_rate": 8e-4},
    {"learning_rate": 6e-4},
    {"learning_rate": 9e-4},
]

REINFORCE_LRS = [1e-3, 5e-4, 2e-3, 1e-4, 3e-4, 7e-4, 8e-4, 9e-4, 1.5e-3, 2.5e-3]

# -------------------------------
# Train single PPO run
# -------------------------------
def train_ppo_run(run_idx, params):
    run_name = f"ppo_run_{run_idx}"
    save_path = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(save_path, exist_ok=True)

    env = MastitisDetectionEnv()
    env = Monitor(env)

    model = PPO('MlpPolicy', env, verbose=1, **params)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(save_path, "final"))

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    return mean_reward, os.path.join(save_path, "final.zip")

# -------------------------------
# Train single A2C run
# -------------------------------
def train_a2c_run(run_idx, params):
    run_name = f"a2c_run_{run_idx}"
    save_path = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(save_path, exist_ok=True)

    env = MastitisDetectionEnv()
    env = Monitor(env)

    model = A2C('MlpPolicy', env, verbose=1, **params)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(save_path, "final"))

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    return mean_reward, os.path.join(save_path, "final.zip")

# -------------------------------
# Train single REINFORCE run
# -------------------------------
def train_reinforce_run(run_idx, lr):
    run_name = f"reinforce_run_{run_idx}"
    save_path = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(save_path, exist_ok=True)

    env = MastitisDetectionEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNet(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(REINFORCE_EPISODES):
        obs, _ = env.reset()
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = policy(obs_v)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        _, reward, _, _, _ = env.step(int(action.item()))
        loss = -m.log_prob(action) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_path = os.path.join(save_path, "policy.pt")
    torch.save(policy.state_dict(), model_path)

    # Evaluate
    rewards = []
    for _ in range(50):
        obs, _ = env.reset()
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = policy(obs_v)
        action = torch.argmax(probs, dim=1).item()
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)

    mean_reward = sum(rewards)/len(rewards)
    return mean_reward, model_path

# -------------------------------
# Main: Train 10 runs and save best
# -------------------------------
if __name__ == "__main__":
    # PPO
    best_reward = -float("inf")
    best_model_path = None
    PPO_BEST_FOLDER = os.path.join(RESULTS_FOLDER, "ppo_best_model")
    for i, params in enumerate(PPO_HYPERS, start=1):
        mean_reward, model_path = train_ppo_run(i, params)
        print(f"PPO Run {i} mean reward: {mean_reward:.3f}")
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_path = model_path
    if best_model_path:
        os.makedirs(PPO_BEST_FOLDER, exist_ok=True)
        shutil.copy(best_model_path, os.path.join(PPO_BEST_FOLDER, "best_model.zip"))
        print(f"PPO best-of-10 saved to {PPO_BEST_FOLDER} with reward {best_reward:.3f}")

    # A2C
    best_reward = -float("inf")
    best_model_path = None
    A2C_BEST_FOLDER = os.path.join(RESULTS_FOLDER, "a2c_best_model")
    for i, params in enumerate(A2C_HYPERS, start=1):
        mean_reward, model_path = train_a2c_run(i, params)
        print(f"A2C Run {i} mean reward: {mean_reward:.3f}")
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_path = model_path
    if best_model_path:
        os.makedirs(A2C_BEST_FOLDER, exist_ok=True)
        shutil.copy(best_model_path, os.path.join(A2C_BEST_FOLDER, "best_model.zip"))
        print(f"A2C best-of-10 saved to {A2C_BEST_FOLDER} with reward {best_reward:.3f}")

    # REINFORCE
    best_reward = -float("inf")
    best_model_path = None
    REINFORCE_BEST_FOLDER = os.path.join(RESULTS_FOLDER, "reinforce_best_model")
    for i, lr in enumerate(REINFORCE_LRS, start=1):
        mean_reward, model_path = train_reinforce_run(i, lr)
        print(f"REINFORCE Run {i} mean reward: {mean_reward:.3f}")
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_path = model_path
    if best_model_path:
        os.makedirs(REINFORCE_BEST_FOLDER, exist_ok=True)
        shutil.copy(best_model_path, os.path.join(REINFORCE_BEST_FOLDER, "policy.pt"))
        print(f"REINFORCE best-of-10 saved to {REINFORCE_BEST_FOLDER} with reward {best_reward:.3f}")
