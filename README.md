# Finding the Fastest Lap: Deep Reinforcement Learning for F1 Race Strategy

A reinforcement learning project that models Formula 1 pit-stop strategy as a Markov Decision Process and trains agents to outperform a heuristic opponent using tabular and deep RL methods.

---

## Overview

In Formula 1, deciding **when to pit and which tire compound to fit** is as critical as raw car speed. This project frames that decision as an MDP and applies five RL algorithms from scratch to learn optimal race strategies.

The project runs in two phases:

- **Phase 1**: Tabular RL (SARSA, Q-Learning, Monte Carlo) on a discrete 1,440-state MDP
- **Phase 2**: Deep RL (DQN, DDQN, PPO, Actor-Critic) on a continuous 8-dimensional race environment

All algorithms were implemented from scratch in Python without relying on external RL libraries.

---

## Environments

### Discrete MDP (Phase 1)

A 10-lap race modeled as a finite-horizon MDP with fully discretized state space.

| Component | Details |
|-----------|---------|
| State Space | 1,440 states: lap, tire compound, tire condition, opponent tire condition, gap, pit count, mandatory compound status |
| Action Space | Stay, Pit-Soft, Pit-Medium |
| Transitions | Deterministic for agent mechanics; stochastic for opponent behavior |
| Reward | Dense per-lap time delta + sparse terminal win/loss bonus (+8000 / -6000) |

### Continuous Environment (Phase 2)

A 50-lap race with real-valued state features and realistic tire degradation dynamics.

| Component | Details |
|-----------|---------|
| State Space | 8-dimensional: lap, tire age, compound, gap to opponent, lap-time delta, safety-car flag, stint length, wear rate |
| Action Space | Stay, Pit-Soft, Pit-Medium, Pit-Hard |
| Transitions | Deterministic simulation with lap-time noise for race variability |
| Reward | Negative lap-time delta per step (faster than opponent = positive reward) |

---

## Algorithms

### Phase 1 - Tabular Methods

| Algorithm | Type | Final Win Rate |
|-----------|------|---------------|
| SARSA | On-policy TD | 100% |
| Q-Learning | Off-policy TD | 100% |
| Monte Carlo (First-Visit) | On-policy MC | 100% |

All three converged to the same optimal 2-stop strategy: **Medium (Start) -> Soft (Lap 5) -> Medium (Lap 7) -> Finish**

### Phase 2 - Deep RL Methods

| Algorithm | Win Rate | Avg. Gap | Avg. Pits |
|-----------|----------|----------|-----------|
| DQN | 100% | +12.4s | 1.0 |
| Double DQN | 100% | +23.7s | 2.0 |
| PPO | 100% | +6.9s | 2.0 |
| Actor-Critic | 0% | -33.9s | 0.0 |

**DDQN achieved the strongest performance**, learning an aggressive early-undercut two-stop strategy (pit laps 13-15 and 32) that produced the largest winning margin.

---

## Key Findings

- **DDQN outperforms DQN** by decoupling action selection from evaluation, reducing Q-value overestimation and producing more stable convergence
- **PPO is the most stable policy-gradient method**, with clipped objectives preventing policy collapse and GAE reducing variance
- **One-step Actor-Critic fails** in long-horizon noisy environments without a replay buffer, collapsing to always selecting "Stay"
- **Tabular methods are surprisingly effective** for the discrete domain: all three converge to identical optimal policies at 100% win rate
- **Exploring Starts was critical** for Monte Carlo to ensure sufficient coverage of the sparse reward state space

---

## Project Structure

```
.
├──Code
    ├── F1Phase1.py                  # Discrete 1440-state MDP environment
    ├── F1Phase2.py         # Continuous 8-dimensional race environment + deep RL agents
├── Project_Report.pdf            # Full project report with learning curves and analysis
└── README.md
```

---

## Tech Stack

- **Python** - core implementation
- **PyTorch** - neural networks for DQN, DDQN, PPO, Actor-Critic
- **NumPy** - state management and matrix operations
- **Matplotlib** - learning curves, win rate plots, action frequency analysis

---

## Setup and Usage

```bash
pip install torch numpy matplotlib
```

**Run discrete MDP (Phase 1):**
```bash
python F1Phase1.py
```

**Run continuous environment with deep RL (Phase 2):**
```bash
python F1Phase2.py
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Discount factor (gamma) | 0.99 |
| Learning rate (DQN/DDQN) | 1e-3 |
| Learning rate (PPO) | 3e-4 |
| Replay buffer size | 10,000 |
| Batch size | 64 |
| Target network update | every 200 steps |
| PPO clip epsilon | 0.2 |
| GAE lambda | 0.95 |
| Hidden layers | 2 x 128, ReLU |
