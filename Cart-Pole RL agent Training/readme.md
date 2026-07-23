# CartPole-v1 — Deep Q-Network (DQN) Agent

A self-contained, single-file implementation of a Deep Q-Network (DQN) agent that solves the classic **CartPole-v1** environment from Gymnasium. 

## Project Overview

CartPole is a classic control problem: a pole is attached to a cart moving along a frictionless track, and the agent must apply left/right forces to the cart to keep the pole balanced upright for as long as possible.

- **Observation space:** 4 continuous values (cart position, cart velocity, pole angle, pole angular velocity)
- **Action space:** 2 discrete actions (push cart left, push cart right)
- **Reward:** +1 for every timestep the pole remains upright
- **Solved condition:** average reward ≥ 475 over 100 consecutive episodes (max episode length is 500)

This project trains a DQN agent — a neural network that approximates the Q-value function — to solve this environment using experience replay and a target network for stable learning.

## Files

| File | Description |
|---|---|
| `cartpole_dqn.py` | Complete training script: installs dependencies, defines the DQN agent, trains it, evaluates it, plots results, and saves outputs |
| `README.md` | This file |

## How It Works

1. **Q-Network** — A small feed-forward neural network (4 → 128 → 128 → 2) that maps a state to Q-values for each action.
2. **Replay Buffer** — Stores past transitions `(state, action, reward, next_state, done)` and samples random mini-batches to break correlation between consecutive experiences.
3. **Target Network** — A slowly-updated copy of the Q-network used to compute stable targets, reducing training instability.
4. **Epsilon-Greedy Exploration** — The agent starts by acting mostly randomly (epsilon ≈ 1.0) and exponentially decays toward mostly-greedy behavior (epsilon ≈ 0.02) as training progresses.
5. **Training Loop** — Runs up to 400 episodes, performing a gradient update after every environment step once enough experience has been collected, and stops early once the environment is considered solved.
6. **Evaluation** — After training, the agent is run greedily (no exploration) for 20 episodes to measure final performance.
7. **Results Saving** — Model weights, reward logs, evaluation scores, a training curve image, and a text summary are all saved to a timestamped folder.



## Output / Results

At the end of a run you'll see:
- A **reward-per-episode plot** with a 20-episode moving average and the solve threshold marked
- Printed **evaluation scores** (list of 20 episode rewards) plus their mean, standard deviation, and max
- A saved **results folder** (`cartpole_results/run_<timestamp>/`) containing:
  - `dqn_cartpole.pth` — trained model weights
  - `episode_rewards.csv` — reward for every training episode
  - `eval_scores.csv` — reward for every evaluation episode
  - `training_curve.png` — the reward plot as an image
  - `summary.txt` — final performance summary


## Key Hyperparameters

| Parameter | Value |
|---|---|
| Discount factor (gamma) | 0.99 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Replay buffer size | 50,000 |
| Epsilon start / end | 1.0 / 0.02 |
| Epsilon decay | ~1500 steps |
| Target network update frequency | every 500 gradient steps |
| Max episodes | 400 |
| Max steps per episode | 500 |

These can be tuned directly in the `train()` and `DQNAgent` sections of `cartpole_dqn.py`.

## Possible Extensions

- Swap DQN for Double DQN or Dueling DQN for improved stability
- Try Stable-Baselines3's PPO for a comparison baseline
- Record a video of the trained agent using `gymnasium`'s `RecordVideo` wrapper
- Sweep hyperparameters (learning rate, epsilon decay, network size) to compare convergence speed
