# train_dqn_multiple_runs.py
import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import MastitisDetectionEnv

# -------------------------------
# Define 10 hyperparameter sets
# -------------------------------
HYPERPARAM_SETS = [
    {"learning_rate": 1e-3, "gamma": 0.95, "buffer_size": 5000, "batch_size": 32},
    {"learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 5000, "batch_size": 32},
    {"learning_rate": 5e-4, "gamma": 0.95, "buffer_size": 10000, "batch_size": 64},
    {"learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 10000, "batch_size": 64},
    {"learning_rate": 1e-4, "gamma": 0.95, "buffer_size": 5000, "batch_size": 32},
    {"learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 5000, "batch_size": 32},
    {"learning_rate": 2e-3, "gamma": 0.95, "buffer_size": 10000, "batch_size": 64},
    {"learning_rate": 2e-3, "gamma": 0.99, "buffer_size": 10000, "batch_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.97, "buffer_size": 7500, "batch_size": 48},
    {"learning_rate": 5e-4, "gamma": 0.97, "buffer_size": 7500, "batch_size": 48},
]

TOTAL_TIMESTEPS = 20000
LEARNING_STARTS = 1000
EVAL_FREQ = 2000
N_EVAL_EPISODES = 50
RESULTS_FOLDER = "models/dqn"
BEST_MODEL_FOLDER = "models/dqn/best_model"

# -------------------------------
# Train a single DQN run
# -------------------------------
def train_dqn_run(run_idx, params):
    run_name = f'run_{run_idx}'
    save_path = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n--- Training DQN Run {run_idx} ---")
    print(f"Hyperparameters: {params}")

    env = MastitisDetectionEnv()
    env = Monitor(env)

    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        buffer_size=params["buffer_size"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        learning_starts=LEARNING_STARTS,
        gamma=params["gamma"]
    )

    eval_env = MastitisDetectionEnv()
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,  # <-- will save best_model.zip here
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(save_path, "final"))
    print(f"Saved DQN run {run_idx} to {save_path}/final")

    # Corrected: path to EvalCallback's best_model.zip
    best_model_path = os.path.join(save_path, "best_model.zip")
    if os.path.exists(best_model_path):
        best_model = DQN.load(best_model_path)
        mean_reward, _ = evaluate_policy(best_model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
        return mean_reward, best_model_path
    else:
        return -float("inf"), None

# -------------------------------
# Main: Train all 10 runs and save the overall best
# -------------------------------
if __name__ == '__main__':
    best_reward = -float("inf")
    best_model_path = None

    for i, params in enumerate(HYPERPARAM_SETS, start=1):
        mean_reward, model_path = train_dqn_run(i, params)
        print(f"Run {i} mean reward: {mean_reward:.3f}")
        if mean_reward > best_reward and model_path is not None:
            best_reward = mean_reward
            best_model_path = model_path

    # Copy the best model to a dedicated folder
    if best_model_path:
        os.makedirs(BEST_MODEL_FOLDER, exist_ok=True)
        shutil.copy(best_model_path, os.path.join(BEST_MODEL_FOLDER, "best_model.zip"))
        print(f"\n=== Overall Best DQN Saved to {BEST_MODEL_FOLDER} with mean reward {best_reward:.3f} ===")
