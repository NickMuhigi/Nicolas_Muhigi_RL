# Mastitis Diagnosis Reinforcement Learning Project

---

## Project Overview

This project develops reinforcement learning (RL) agents to automatically diagnose mastitis in cows using a simulated environment. The agents observe features representing the cow's health and decide whether the cow is healthy or diseased.

I implemented **value-based methods (DQN)** and **policy-gradient methods (REINFORCE, PPO, A2C)** to optimize diagnosis accuracy and cumulative rewards. Multiple hyperparameter configurations were tested to identify the best-performing models.

---

## Environment

The custom environment (`MastitisDetectionEnv`) is built using `gymnasium` and `pygame` for visualization.

- **Agent:** A diagnostic system that observes cow health indicators and selects a diagnosis
- **Action Space:** Discrete
  - `0` – Cow is healthy
  - `1` – Cow is diseased
- **Observation Space:** Vector of numerical features representing the cow's condition (temperature, somatic cell count, etc.)
- **Reward:**
  - `+1` for correct diagnosis
  - `0` for incorrect diagnosis
- **Termination:** Each episode ends after a single cow diagnosis or a fixed number of steps

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/student_name_rl_summative.git
cd student_name_rl_summative
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py          # Custom Gymnasium environment
│   ├── rendering.py           # Visualization components (pygame)
├── training/
│   ├── dqn_training.py        # DQN training script
│   ├── pg_training.py         # Policy Gradient training (REINFORCE/PPO/A2C)
├── models/
│   ├── dqn/                   # Saved DQN models
│   └── pg/                    # Saved Policy Gradient models
├── main.py                    # Run best-performing model
├── evaluate.py                # Evaluate all trained models
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## Usage

### 1. Train Models

**DQN:**

```bash
python training/dqn_training.py
```

**PPO / A2C / REINFORCE:**

```bash
python training/pg_training.py --algorithm ppo
```

### 2. Evaluate Models

```bash
python evaluate.py
```

This will evaluate all runs and save results in `results/logs/`.

### 3. Run Best Model

```bash
python main.py <model_path> <algorithm>
```

**Example:**

```bash
python main.py models/ppo/ppo_best_model/best_model.zip ppo
```

This runs the trained agent in the GUI and prints terminal output for each step.

---

## Visualization

The environment is visualized in pygame with a real-time display of cow indicators, agent decisions, and reward feedback.

- Green highlights indicate correct diagnoses
- Red indicates incorrect ones

Videos and plots of training curves can be found in `results/logs/`.

---

## Results

- **DQN** and **PPO** achieved the highest accuracy and stable rewards
- **A2C** showed slightly more variance
- **REINFORCE** converged slower and had lower generalization

Plots of cumulative rewards, training stability, and convergence curves are included in the PDF report.

---

## Requirements

- Python 3.9+
- gymnasium
- pygame
- torch
- stable-baselines3
- numpy
- matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## License

This project is for academic purposes. Unauthorized commercial use is prohibited.