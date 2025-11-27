import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MastitisDetectionEnv(gym.Env):
    """
    Simple diagnostic environment for mastitis vs healthy cow.

    Observation: vector of features [temperature, somatic_cell_count, milk_color_score, appetite, udder_swelling]
      - temperature: float in [36,41] Celsius
      - somatic_cell_count (SCC): float in [0,1000] (thousands)
      - milk_color_score: 0 (normal) - 1 (abnormal)
      - appetite: 0 (low) - 1 (normal)
      - udder_swelling: 0 (no) - 1 (yes)

    Actions: 0 = diagnose Healthy, 1 = diagnose Mastitis
    Reward: +1 correct classification, -1 incorrect.
    Episodes are single-step (one diagnosis per case).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([36.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([41.0, 1000.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.current_obs = None
        self.current_label = None  # 0 healthy, 1 mastitis
        self.render_mode = render_mode

    def _sample_case(self):
        # Generate a case. Mastitis more likely to have higher temp, higher SCC, abnormal milk, low appetite, swelling
        is_mastitis = random.random() < 0.4  # prevalence in sampled cases
        if is_mastitis:
            temp = random.uniform(39.0, 41.0)
            scc = random.uniform(300.0, 1000.0)
            milk_color = 1.0 if random.random() < 0.7 else 0.0
            appetite = 0.0 if random.random() < 0.6 else 1.0
            swelling = 1.0 if random.random() < 0.6 else 0.0
            label = 1
        else:
            temp = random.uniform(36.0, 39.0)
            scc = random.uniform(0.0, 300.0)
            milk_color = 0.0 if random.random() < 0.9 else 1.0
            appetite = 1.0 if random.random() < 0.9 else 0.0
            swelling = 0.0 if random.random() < 0.95 else 1.0
            label = 0
        obs = np.array([temp, scc, milk_color, appetite, swelling], dtype=np.float32)
        return obs, label

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_obs, self.current_label = self._sample_case()
        return self.current_obs, {}

    def step(self, action):
        done = True  # one-step episode
        correct = (action == self.current_label)
        reward = 1.0 if correct else -1.0
        info = {"label": int(self.current_label), "correct": bool(correct)}
        obs = np.copy(self.current_obs)
        # gymnasium step signature: obs, reward, terminated, truncated, info
        return obs, reward, True, False, info

    def render(self):
        # Minimal textual render
        print(f"Observation: temp={self.current_obs[0]:.1f}C, SCC={self.current_obs[1]:.1f}, "
              f"milk_color={int(self.current_obs[2])}, appetite={int(self.current_obs[3])}, "
              f"swelling={int(self.current_obs[4])}")
        print(f"(Hidden) Ground truth label: {'MASTITIS' if self.current_label==1 else 'HEALTHY'}")
