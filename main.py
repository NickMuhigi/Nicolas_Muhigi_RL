# main.py
"""
Run a trained RL agent continuously across multiple episodes
on the same GUI window.
"""

import sys
import time
import torch
import torch.nn as nn
from environment.custom_env import MastitisDetectionEnv
from environment.rendering import render_case
from stable_baselines3 import PPO, A2C, DQN

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
# Run a model continuously
# -------------------------------
def run_model(model_path, alg='ppo', episodes=10, wait_time=0.5):
    env = MastitisDetectionEnv()

    if alg.lower() in ['ppo', 'a2c', 'dqn']:
        # Load SB3 model
        if alg.lower() == 'ppo':
            model = PPO.load(model_path)
        elif alg.lower() == 'a2c':
            model = A2C.load(model_path)
        else:
            model = DQN.load(model_path)
        use_sb3 = True
    elif alg.lower() == 'reinforce':
        # Load REINFORCE policy
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        model = PolicyNet(obs_dim, act_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        use_sb3 = False
    else:
        print("Unsupported algorithm. Use 'ppo', 'a2c', 'dqn', or 'reinforce'.")
        return

    obs, _ = env.reset()
    for episode in range(episodes):
        done = False
        step_num = 0
        while not done:
            if use_sb3:
                action, _states = model.predict(obs, deterministic=True)
            else:
                obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    probs = model(obs_v)
                action = torch.argmax(probs, dim=1).item()

            obs2, reward, done, _, info = env.step(int(action))
            render_case(obs2, diagnosis=int(action), ground_truth=info.get('label'))
            time.sleep(wait_time)
            print(f'Episode {episode+1}, Step {step_num}: action={action}, label={info.get("label")}, reward={reward}')
            obs = obs2
            step_num += 1

        # Reset environment for next episode
        obs, _ = env.reset()

    print("Simulation complete.")

# -------------------------------
# Main entry
# -------------------------------
if __name__ == '__main__':
    BEST_MODELS = {
        "dqn": "models/dqn/best_model/best_model.zip",
        "ppo": "models/pg/ppo/ppo_best_model/best_model.zip",
        "a2c": "models/pg/a2c/a2c_best_model/best_model.zip",
        "reinforce": "models/pg/reinforce/reinforce_best_model/policy.pt"
    }

    DEFAULT_ALG = "ppo"
    DEFAULT_EPISODES = 10

    if len(sys.argv) >= 2:
        alg = sys.argv[1].lower()
        model_path = BEST_MODELS.get(alg, BEST_MODELS[DEFAULT_ALG])
    else:
        alg = DEFAULT_ALG
        model_path = BEST_MODELS[DEFAULT_ALG]

    run_model(model_path, alg, episodes=DEFAULT_EPISODES)
