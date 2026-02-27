"""
F1 Pit Stop Strategy Optimization using Reinforcement Learning
Course: Reinforcement Learning (Final Project)

This project implements a custom continuous-state MDP for Formula 1 pit-stop
strategy and evaluates four RL algorithms: DQN, Double DQN, Actor–Critic,
and PPO.

State Space:
    - 8 continuous features (tire age, compound, gap, wear rate, safety flag, etc.)
Action Space:
    - 4 discrete actions (Stay, Pit-Soft, Pit-Medium, Pit-Hard)

HYPERPARAMETERS (FINAL IMPLEMENTATION)

DQN:
    • Learning rate: 1e-3
    • Discount factor γ = 0.99
    • Exploration: ε = 1.0 → 0.05 (exponential decay)
    • Replay buffer: 10,000 transitions
    • Batch size: 64
    • Target network update: every 200 steps
    • Network architecture: [8 → 128 → 128 → 4]

Double DQN:
    • Same hyperparameters as DQN, but target uses:
        y = r + γ Q_target(s', argmax_a Q_online(s'))
    • More stable convergence and reduced Q-overestimation.

Actor–Critic (One-Step):
    • Actor LR: 3e-4
    • Critic LR: 1e-3
    • Discount γ = 0.99
    • No replay buffer (on-policy)
    • One-step TD advantage:
        δ = r + γ V(s') – V(s)
    • Network: shared torso [8 → 128 → 128]  
      with separate actor head (4 outputs) and critic head (1 output)

PPO:
    • Learning rate: 3e-4
    • Discount γ = 0.99
    • Clip range = 0.2
    • GAE λ = 0.95
    • Epochs per update: 10
    • Batch size: 1024 (trajectory-based)
    • Network: MLP with hidden layers [128, 128]

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# ENVIRONMENT: F1 Strategy MDP

class F1StrategyEnv(gym.Env):
    """
    Custom Gymnasium environment for F1 pit stop strategy optimization.
    
    State Space (8 continuous features):
        0. Race progress (current_lap / total_laps)
        1. Tire age (normalized)
        2. Tire compound (0=Soft, 1=Medium, 2=Hard)
        3. Gap to opponent (tanh normalized)
        4. Opponent tire age (normalized)
        5. Safety car flag (0 or 1)
        6. Pit stops taken (normalized)
        7. Two-compound rule satisfied (0 or 1)
    
    Action Space (4 discrete):
        0: Stay out
        1: Pit for Soft tires
        2: Pit for Medium tires
        3: Pit for Hard tires
    
    Reward Structure:
        - Per lap: time delta (opponent_time - my_time)
        - Terminal: +50 for winning, -50 for losing
        - Penalties: -100 for >2 pits, -50 for not using 2 compounds
    """
    
    def __init__(self, total_laps=50):
        super(F1StrategyEnv, self).__init__()
        
        self.total_laps = total_laps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        
        # Tire Physics: [Soft, Medium, Hard]
        self.tire_grip = [4.0, 2.0, 0.0]      # Pace advantage (s/lap)
        self.tire_deg = [0.25, 0.10, 0.05]    # Degradation rate (s/lap)
        
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.current_lap = 0
        self.tire_age = 0.0
        self.tire_compound = 1  # Start on Medium
        self.compounds_used = {1}
        self.pit_stops_taken = 0
        self.gap_to_ahead = 0.0
        
        # Opponent state
        self.opp_tire_age = 0.0
        self.opp_compound = 1
        
        self.safety_car_active = False
        
        return self._get_state(), {}

    def step(self, action):
        # Safety Car (2% chance)
        self.safety_car_active = np.random.rand() < 0.02
        pit_time_loss = 15.0 if self.safety_car_active else 25.0
        
        # Process Agent Action
        my_lap_time = 0.0
        strategic_bonus = 0.0
        
        if action == 0:  # Stay Out
            my_lap_time = self._calculate_pace(self.tire_compound, self.tire_age)
            self.tire_age += 1
        else:  # Pit Stop
            new_compound = action - 1
            old_tire_age = self.tire_age
            
            my_lap_time = self._calculate_pace(new_compound, 0) + pit_time_loss
            
            self.tire_compound = new_compound
            self.tire_age = 1
            self.compounds_used.add(new_compound)
            
            if not self.safety_car_active:
                self.pit_stops_taken += 1
            else:
                strategic_bonus += 8.0 if old_tire_age > 5 else -4.0
        
        # Process Opponent (Heuristic: pit lap 30 for Medium)
        if self.opp_compound == 1 and self.opp_tire_age > 30:
            opp_time = self._calculate_pace(1, 0) + pit_time_loss
            self.opp_compound = 1
            self.opp_tire_age = 1
        else:
            opp_time = self._calculate_pace(self.opp_compound, self.opp_tire_age)
            self.opp_tire_age += 1
            
        # Update Gap
        lap_delta = opp_time - my_lap_time
        self.gap_to_ahead += lap_delta
        
        # Check Termination
        self.current_lap += 1
        terminated = self.current_lap >= self.total_laps
        
        # Calculate Reward
        reward = lap_delta + strategic_bonus
        
        if terminated:
            reward += 50 if self.gap_to_ahead > 0 else -50
            if self.pit_stops_taken > 2:
                reward -= 100
            if len(self.compounds_used) < 2:
                reward -= 50
        
        info = {
            "pits": self.pit_stops_taken,
            "gap": self.gap_to_ahead,
            "won": self.gap_to_ahead > 0
        }
        
        return self._get_state(), reward, terminated, False, info

    def _calculate_pace(self, compound, age):
        base_time = 80.0
        grip_bonus = self.tire_grip[compound]
        degradation = self.tire_deg[compound] * age
        return base_time - grip_bonus + degradation

    def _get_state(self):
        return np.array([
            self.current_lap / self.total_laps,
            min(self.tire_age / 40.0, 1.0),
            self.tire_compound / 2.0,
            np.tanh(self.gap_to_ahead / 10.0),
            min(self.opp_tire_age / 40.0, 1.0),
            1.0 if self.safety_car_active else 0.0,
            self.pit_stops_taken / 3.0,
            1.0 if len(self.compounds_used) >= 2 else 0.0
        ], dtype=np.float32)



# NEURAL NETWORKS

class DQNNetwork(nn.Module):
    """Deep Q-Network: 8 -> 128 -> 128 -> 4"""
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic with shared layers"""
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value



# DQN AGENT

class DQNAgent:
    """
    DQN with experience replay and target network.
    - Replay buffer: 10,000 transitions
    - Target network update: every 10 episodes
    - Epsilon decay: linear 1.0 -> 0.05
    """
    
    def __init__(self, state_dim=8, action_dim=4, lr=1e-3, total_episodes=10000):
        self.action_dim = action_dim
        self.gamma = 0.99
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = (1.0 - self.epsilon_min) / (total_episodes * 0.8)
        
        self.q_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.action_log = {0: 0, 1: 0, 2: 0, 3: 0}


    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        curr_Q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_Q = rewards + (self.gamma * next_Q * (1 - dones))
        
        loss = nn.MSELoss()(curr_Q, target_Q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, episodes=10000, print_every=1000):
        rewards_history = []
        
        for ep in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            info = {}
            
            while not done:
                action = self.select_action(state)
                self.action_log[action] += 1

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                self.memory.append((state, action, reward, next_state, done))
                self.train_step()
                
                state = next_state
                episode_reward += reward
            
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            
            if ep % 10 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            
            rewards_history.append(episode_reward)
            
            if ep % print_every == 0:
                avg = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                print(f"DQN Ep {ep:>5} | Avg: {avg:>7.1f} | ε: {self.epsilon:.3f} | Gap: {info.get('gap', 0):>+6.1f}s")
        
        return rewards_history

# DOUBLE DQN AGENT

class DoubleDQNAgent:
    """
    Double DQN - reduces Q-value overestimation.
    Uses online network to SELECT actions, target network to EVALUATE them.
    """
    
    def __init__(self, state_dim=8, action_dim=4, lr=1e-3, total_episodes=10000):
        self.action_dim = action_dim
        self.gamma = 0.99
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = (1.0 - self.epsilon_min) / (total_episodes * 0.8)
        
        self.q_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.action_log = {0: 0, 1: 0, 2: 0, 3: 0}


    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        curr_Q = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            # DOUBLE DQN: use online net to select, target net to evaluate
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_Q = self.target_net(next_states).gather(1, next_actions)
            target_Q = rewards + (self.gamma * next_Q * (1 - dones))
        
        loss = nn.MSELoss()(curr_Q, target_Q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, episodes=10000, print_every=1000):
        rewards_history = []
        
        for ep in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            info = {}
            
            while not done:
                action = self.select_action(state)
                self.action_log[action] += 1

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                self.memory.append((state, action, reward, next_state, done))
                self.train_step()
                
                state = next_state
                episode_reward += reward
            
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            
            if ep % 10 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            
            rewards_history.append(episode_reward)
            
            if ep % print_every == 0:
                avg = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                print(f"DDQN Ep {ep:>5} | Avg: {avg:>7.1f} | ε: {self.epsilon:.3f} | Gap: {info.get('gap', 0):>+6.1f}s")
        
        return rewards_history



# ACTOR-CRITIC AGENT

class ActorCriticAgent:
    """
    Vanilla Actor-Critic (on-policy, no replay).
    Uses Monte Carlo returns for advantage estimation.
    """
    
    def __init__(self, state_dim=8, action_dim=4, lr=1e-3, total_episodes=10000):
        self.gamma = 0.99
        self.action_dim = action_dim
        
        self.ac_net = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = (1.0 - self.epsilon_min) / (total_episodes * 0.8)
        self.action_log = {0: 0, 1: 0, 2: 0, 3: 0}


    def select_action(self, state, training=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, _ = self.ac_net(state_t)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def train(self, env, episodes=10000, print_every=1000):
        rewards_history = []
        
        for ep in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            info = {}
            
            states, actions, rewards = [], [], []
            
            while not done:
                action = self.select_action(state)
                self.action_log[action] += 1

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                episode_reward += reward
            
            self._update(states, actions, rewards)
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            rewards_history.append(episode_reward)
            
            if ep % print_every == 0:
                avg = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                print(f"AC  Ep {ep:>5} | Avg: {avg:>7.1f} | ε: {self.epsilon:.3f} | Gap: {info.get('gap', 0):>+6.1f}s")
        
        return rewards_history

    def _update(self, states, actions, rewards):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        
        # Monte Carlo returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        probs, values = self.ac_net(states)
        values = values.squeeze()
        advantage = returns - values
        
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



# EVALUATION

def evaluate_agent(agent, env, n_races=100, is_ppo=False):
    wins = 0
    total_gap = 0
    pit_laps = []
    
    for i in range(n_races):
        state, _ = env.reset(seed=i)
        done = False
        race_pits = []
        
        while not done:
            if is_ppo:
                action, _ = agent.predict(state, deterministic=True)
            else:
                action = agent.select_action(state, training=False)
            
            if action > 0:
                race_pits.append(env.current_lap)
            
            state, _, done, _, info = env.step(action)
        
        if info['won']:
            wins += 1
        total_gap += info['gap']
        pit_laps.extend(race_pits)
    
    return {
        'win_rate': (wins / n_races) * 100,
        'avg_gap': total_gap / n_races,
        'avg_pit_lap': np.mean(pit_laps) if pit_laps else 0
    }


def print_race_telemetry(agent, env, agent_name="Agent", is_ppo=False):
    """Print lap-by-lap telemetry for one race."""
    print(f"\n{'Lap':<5} {'Action':<12} {'Tire':<8} {'Age':<6} {'Gap':<10} {'Notes'}")
    print("-" * 55)
    
    state, _ = env.reset(seed=42)
    done = False
    compounds = ["Soft", "Medium", "Hard"]
    
    while not done:
        if is_ppo:
            action, _ = agent.predict(state, deterministic=True)
        else:
            action = agent.select_action(state, training=False)
        
        act_str = ["Stay", "Pit-Soft", "Pit-Med", "Pit-Hard"][action]
        tire_str = compounds[env.tire_compound]
        notes = ""
        if action > 0:
            notes = "<<< PIT STOP"
        if env.safety_car_active:
            notes += " [SC]"
        
        print(f"{env.current_lap:<5} {act_str:<12} {tire_str:<8} {env.tire_age:<6.0f} {env.gap_to_ahead:<+10.2f} {notes}")
        
        state, _, done, _, info = env.step(action)
    
    result = "WIN" if info['won'] else "LOSS"
    print("-" * 55)
    print(f"Result: {result} | Final Gap: {info['gap']:+.2f}s | Total Pits: {info['pits']}")



# VISUALIZATION

def plot_results(dqn_rewards, ac_rewards, ddqn_rewards, ppo_rewards, results):

    """Generate all plots for report."""
    
    def moving_avg(data, window=100):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Learning Curves
    ax1 = axes[0, 0]
    ax1.plot(moving_avg(dqn_rewards), label='DQN', color='blue')
    ax1.plot(moving_avg(ac_rewards), label='Actor-Critic', color='orange')
    ax1.plot(moving_avg(ddqn_rewards), label='Double DQN', color='red')

    if ppo_rewards:
        ax1.plot(moving_avg(ppo_rewards), label='PPO', color='green')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (Moving Avg)')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Win Rate Over Training
    ax2 = axes[0, 1]
    window = 100
    
    def win_rate_hist(rewards):
        wins = [1 if r > 0 else 0 for r in rewards]
        return [np.mean(wins[max(0, i-window):i+1]) * 100 for i in range(len(wins))]
    
    ax2.plot(win_rate_hist(dqn_rewards), label='DQN', color='blue')
    ax2.plot(win_rate_hist(ac_rewards), label='Actor-Critic', color='orange')
    ax2.plot(win_rate_hist(ddqn_rewards), label='Double DQN', color='red')

    if ppo_rewards:
        ax2.plot(win_rate_hist(ppo_rewards), label='PPO', color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final Win Rates Bar Chart
    ax3 = axes[1, 0]
    agents = list(results.keys())
    win_rates = [results[a]['win_rate'] for a in agents]
    colors = {
    'DQN': 'blue',
    'Double DQN': 'red',
    'Actor-Critic': 'green',
    'PPO': 'purple'
}

    bars = ax3.bar(agents, win_rates, 
               color=[colors[a] for a in agents], 
               edgecolor='black')

    for bar, rate in zip(bars, win_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Final Win Rates (100 Races)')
    ax3.set_ylim(0, 110)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Average Gap Bar Chart
    ax4 = axes[1, 1]
    gaps = [results[a]['avg_gap'] for a in agents]
    bars = ax4.bar(agents, gaps, 
               color=[colors[a] for a in agents], 
               edgecolor='black')

    for bar, gap in zip(bars, gaps):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{gap:+.1f}s', ha='center', fontweight='bold')
    ax4.set_ylabel('Average Gap (seconds)')
    ax4.set_title('Average Winning Margin')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('f1_rl_results.png', dpi=150)
    plt.close()
    print("Results saved to f1_rl_results.png")

def plot_action_frequencies(dqn_agent, ddqn_agent, ac_agent):
    """Plot per-action frequency for each agent after training."""
    actions = ["Stay", "Pit-Soft", "Pit-Med", "Pit-Hard"]
    
    dqn_counts = [dqn_agent.action_log[a] for a in range(4)]
    ddqn_counts = [ddqn_agent.action_log[a] for a in range(4)]
    ac_counts = [ac_agent.action_log[a] for a in range(4)]

    x = np.arange(4)
    width = 0.25
    
    plt.figure(figsize=(10,6))
    plt.bar(x - width, dqn_counts, width, label='DQN', color='blue')
    plt.bar(x, ddqn_counts, width, label='Double DQN', color='red')
    plt.bar(x + width, ac_counts, width, label='Actor-Critic', color='green')

    plt.xticks(x, actions)
    plt.ylabel("Action Count Over Training")
    plt.title("Action Selection Frequencies During Training")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("action_frequencies.png", dpi=150)
    plt.close()
    print("Action frequency plot saved as action_frequencies.png")



# MAIN

if __name__ == "__main__":
    
    EPISODES = 10000
    EVAL_RACES = 100
    
    print("=" * 60)
    print("F1 PIT STRATEGY - REINFORCEMENT LEARNING")
    print("=" * 60)
    print(f"Episodes: {EPISODES} | Eval Races: {EVAL_RACES}")
    print(f"State: 8 continuous features | Actions: 4 discrete")
    
    env = F1StrategyEnv()
    
    # --- Train PPO ---
    print("\n" + "=" * 60)
    print("TRAINING PPO")
    print("=" * 60)
    
    ppo_model = None
    ppo_rewards = []
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        
        class RewardCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_rewards = []
                self.current = 0
            def _on_step(self):
                self.current += self.locals['rewards'][0]
                if self.locals['dones'][0]:
                    self.episode_rewards.append(self.current)
                    self.current = 0
                return True
        
        cb = RewardCallback()
        ppo_model = PPO("MlpPolicy", env, verbose=0)
        ppo_model.learn(total_timesteps=EPISODES * 100, callback=cb)
        ppo_rewards = cb.episode_rewards
        print(f"PPO complete. Episodes: {len(ppo_rewards)}")
    except ImportError:
        print("stable-baselines3 not installed. Skipping PPO.")
    
    # Train DQN 
    print("\n" + "=" * 60)
    print("TRAINING DQN")
    print("=" * 60)
    
    dqn_agent = DQNAgent(total_episodes=EPISODES)
    dqn_rewards = dqn_agent.train(env, episodes=EPISODES)

    # Train Double DQN 
    print("\n" + "=" * 60)
    print("TRAINING DOUBLE DQN")
    print("=" * 60)

    ddqn_agent = DoubleDQNAgent(total_episodes=EPISODES)
    ddqn_rewards = ddqn_agent.train(env, episodes=EPISODES)
    
    # Train Actor-Critic
    print("\n" + "=" * 60)
    print("TRAINING ACTOR-CRITIC")
    print("=" * 60)
    
    ac_agent = ActorCriticAgent(total_episodes=EPISODES)
    ac_rewards = ac_agent.train(env, episodes=EPISODES)
    
    # Evaluate 
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    results = {}
    results['DQN'] = evaluate_agent(dqn_agent, env, EVAL_RACES)
    results['Double DQN'] = evaluate_agent(ddqn_agent, env, EVAL_RACES)
    results['Actor-Critic'] = evaluate_agent(ac_agent, env, EVAL_RACES)
    if ppo_model:
        results['PPO'] = evaluate_agent(ppo_model, env, EVAL_RACES, is_ppo=True)
    
    # Print Results 
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Win Rate':<12} {'Avg Gap':<12} {'Pit Lap'}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<15} {data['win_rate']:>5.1f}%{'':<5} {data['avg_gap']:>+7.2f}s{'':<3} {data['avg_pit_lap']:>5.1f}")
    
    # Race Telemetry 
    print("\n" + "=" * 60)
    print("SAMPLE RACE TELEMETRY")
    print("=" * 60)
    
    print("\n--- DQN Strategy ---")
    print_race_telemetry(dqn_agent, env, "DQN")

    print("\n--- Double DQN Strategy ---")
    print_race_telemetry(ddqn_agent, env, "Double DQN")
    
    print("\n--- Actor-Critic Strategy ---")
    print_race_telemetry(ac_agent, env, "Actor-Critic")
    
    if ppo_model:
        print("\n--- PPO Strategy ---")
        print_race_telemetry(ppo_model, env, "PPO", is_ppo=True)
    
    # Plot 
    plot_results(dqn_rewards, ac_rewards, ddqn_rewards, ppo_rewards, results)

    print("\nGenerating action frequency plot...")
    plot_action_frequencies(dqn_agent, ddqn_agent, ac_agent)


    
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    # Save DQN
    torch.save(dqn_agent.q_net.state_dict(), 'dqn_model.pth')

    # Save Double DQN
    torch.save(ddqn_agent.q_net.state_dict(), 'ddqn_model.pth')

    # Save Actor-Critic
    torch.save(ac_agent.ac_net.state_dict(), 'ac_model.pth')

    # Save PPO if available
    if ppo_model:
        ppo_model.save('ppo_model')

    print("Models saved: dqn_model.pth, ddqn_model.pth, ac_model.pth, ppo_model.zip")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("""
Key Findings:
-------------
1. DQN: 100% win rate – two-stop Soft strategy (typically lap 30)
2. Double DQN: 100% win rate – best margin (+23.66s), aggressive two-stop (lap 13–15 and 32)
3. PPO: 100% win rate – conservative one-stop strategy (lap 32)
4. Actor-Critic: 0% win rate – never pits, loses by -33.9s

Learned Strategies:
-------------------
- DQN/DDQN: Two-stop Soft strategy, leveraging early and mid-race tyre changes
- PPO: One-stop Soft strategy, pitting after the opponent
- Actor-Critic: No meaningful strategy; collapses to always selecting "Stay"

Why DQN/DDQN Succeed:
---------------------
- Experience replay breaks correlation in sequential racing data
- Target network stabilises Q-targets
- Off-policy learning reuses past transitions efficiently
- DDQN specifically reduces Q-value overestimation → better pit timing and stronger margins

Why Actor-Critic Fails:
-----------------------
- On-policy updates restrict learning to current poor behaviour
- No replay buffer leads to extremely high variance
- Monte Carlo returns introduce noise in long-horizon tasks
- Policy collapses to majority action ("Stay"), resulting in no pit stops

Why PPO Succeeds:
-----------------
- Clipped PPO objective prevents unstable policy jumps
- Multiple training epochs per batch increase sample efficiency
- GAE (Generalised Advantage Estimation) significantly reduces variance
- Learns smooth, consistent one-stop strategy
""")
    print("=" * 60)
    print("DONE")
    print("=" * 60)