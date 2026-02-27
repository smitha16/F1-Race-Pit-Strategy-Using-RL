import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os

ACTION_STAY = 0
ACTION_SOFT = 1
ACTION_MED = 2
ACTIONS = [ACTION_STAY, ACTION_SOFT, ACTION_MED]
ACTION_SYMBOLS = ['Stay', 'Soft', 'Med']
PLOT_WINDOW = 2000
PLOT_SUBSAMPLE = 50


TOTAL_RACE_LAPS = 10
NUM_COMPOUNDS = 2  # Soft, Medium
NUM_MY_TIRES = 3  # Fresh, Worn, Dead
NUM_OPP_TIRES = 2  # Fresh, Old
NUM_GAPS = 2  # Behind, Ahead
NUM_MY_PITS = 3  # 0, 1, 2+ stops
NUM_MANDATORY = 2
NUM_SC = 2

MY_PIT_LABELS = ["0 stops", "1 stop", "2+ stops"]
LAP_LABELS = [f"Lap {i}" for i in range(1, TOTAL_RACE_LAPS + 1)]
COMPOUND_LABELS = ["Soft", "Med"]
MY_TIRE_LABELS = ["Fresh", "Worn", "DEAD"]
OPP_TIRE_LABELS = ["Fresh", "Old"]
GAP_LABELS = ["Behind", "Ahead"]

BASE_LAP_TIME = 80.0
PIT_LOSS = 35.0
PUNCTURE_PENALTY = -7000.0
MANDATORY_PENALTY = -7000.0
EXTRA_PIT_PENALTY = -800.0

COMP_SOFT = 1
COMP_MED = 2

TIRE_GRIP = [0.0, 50.0, 30.0]
TIRE_DEG = [0.0, 40.0, 14.0]
TIRE_MAX_LIFE = [0, 3, 7]

VALID_STATES = [
    (l, c, t, ot, g, mp, mand)
    for l in range(1, TOTAL_RACE_LAPS + 1)
    for c in range(NUM_COMPOUNDS)
    for t in range(NUM_MY_TIRES)
    for ot in range(NUM_OPP_TIRES)
    for g in range(NUM_GAPS)
    for mp in range(NUM_MY_PITS)
    for mand in range(NUM_MANDATORY)
]

STATE_LIST = VALID_STATES[:]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}

NUM_STATES = len(STATE_LIST)
NUM_ACTIONS = len(ACTIONS)


def state_to_index(state):
    return STATE_INDEX[state]

def index_to_state(idx):
    return STATE_LIST[idx]

def calculate_pace(c, age):
    if c == 0 or c >= len(TIRE_GRIP):
        return BASE_LAP_TIME
    grip = TIRE_GRIP[c]
    deg = TIRE_DEG[c] * age
    if age > TIRE_MAX_LIFE[c]:
        deg += 6.0
    return BASE_LAP_TIME - grip + deg


def sample_opponent_params():
    #Randomising opponent behaviour each episode
    return {
        "med_life_pct_pit": random.uniform(0.55, 0.8),
        "soft_life_pct_pit": random.uniform(0.6, 0.9),
        "attack_gap": random.uniform(-8.0, -3.0),
        "defend_gap": random.uniform(2.0, 8.0)
    }


class F1SimpleEnv:
    def __init__(self):
        self.opp_params = None
        self.opp_compound = None
        self.opp_tire_age = None
        self.dnf = None
        self.gap = None
        self.compound = None
        self.lap = None
        self.tire_age = None
        self.my_pit_count = None
        self.used_soft = None
        self.used_med = None
        self.mandatory_ok = None
        self.mandatory_before = None
        self.reset()

    def reset(self):
        #Standard Race Start
        self.opp_params = sample_opponent_params()
        self.lap = 1
        self.tire_age = 0
        self.compound = COMP_MED
        self.gap = 0.0
        self.dnf = False
        self.opp_tire_age = 0
        self.opp_compound = COMP_MED
        self.my_pit_count = 0
        self.used_soft = (self.compound == COMP_SOFT)
        self.used_med = (self.compound == COMP_MED)
        self.mandatory_ok = int(self.used_soft and self.used_med)
        self.mandatory_before = self.mandatory_ok
        return self.get_state()

    def reset_exploring_start(self):
        self.opp_params = sample_opponent_params()

        # Sample lap with bias toward mid‑race
        self.lap = random.choices(
            population=list(range(1, TOTAL_RACE_LAPS + 1)),
            weights=[1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
            k=1
        )[0]

        # Sample pit count
        if self.lap == 1:
            self.my_pit_count = 0
        elif self.lap <= 4:
            self.my_pit_count = random.choice([0, 1])
        elif self.lap <= 7:
            self.my_pit_count = random.choice([0, 1, 2])
        else:
            self.my_pit_count = random.choice([1, 2])

        # Sample compound & tyre age
        self.compound = random.choice([COMP_SOFT, COMP_MED])
        max_life = TIRE_MAX_LIFE[self.compound]

        # On lap 1-2 always fresh tire. Later laps can be fresh / worn / dead
        if self.lap <= 2:
            self.tire_age = 0
        else:
            # age cannot exceed laps completed
            max_age = min(self.lap - 1, max_life + 4)
            age_candidates = list(range(0, max_age + 1))
            weights = [2 if a <= max_life else 1 for a in age_candidates]
            self.tire_age = random.choices(age_candidates, weights=weights, k=1)[0]

        # Gap
        self.gap = random.uniform(-15.0, 15.0)

        # Opponent compound & age
        self.opp_compound = random.choice([COMP_SOFT, COMP_MED])
        opp_max = TIRE_MAX_LIFE[self.opp_compound]
        if self.lap <= 2:
            self.opp_tire_age = 0
        else:
            opp_max_age = min(self.lap - 1, opp_max + 4)
            age_candidates = list(range(0, opp_max_age + 1))
            weights = [2 if a <= opp_max else 1 for a in age_candidates]
            self.opp_tire_age = random.choices(age_candidates, weights=weights, k=1)[0]

        # Sample mandatory compound flag
        if self.my_pit_count == 0:
            if self.compound == COMP_SOFT:
                self.used_soft, self.used_med = True, False
            else:
                self.used_soft, self.used_med = False, True
        else:
            if random.random() < 0.5:
                self.used_soft = True
                self.used_med = True
            else:
                if self.compound == COMP_SOFT:
                    self.used_soft, self.used_med = True, False
                else:
                    self.used_soft, self.used_med = False, True

        self.mandatory_ok = int(self.used_soft and self.used_med)

        self.dnf = False
        self.mandatory_before = self.mandatory_ok

        return self.get_state()

    def get_state(self):

        c = self.compound - 1

        max_life = TIRE_MAX_LIFE[self.compound]
        life_pct = self.tire_age / max_life if max_life > 0 else 0

        if self.tire_age > max_life:
            t = 2
        elif life_pct <= 0.5:
            t = 0
        else:
            t = 1

        opp_max = TIRE_MAX_LIFE[self.opp_compound]
        opp_pct = self.opp_tire_age / opp_max if opp_max > 0 else 0
        ot = 0 if opp_pct < 0.6 else 1

        g = 0 if self.gap <= 0.0 else 1

        mp = min(self.my_pit_count, NUM_MY_PITS - 1)

        mand = self.mandatory_ok

        return self.lap, c, t, ot, g, mp, mand

    def step(self, action):

        if self.dnf:
            return self.get_state(), 0, True, 0, 0

        if action == ACTION_STAY:
            max_life = TIRE_MAX_LIFE[self.compound]
            if self.tire_age > max_life:
                if np.random.random() < 0.5:
                    self.dnf = True
                    return self.get_state(), PUNCTURE_PENALTY, True, 0, 0
            my_time = calculate_pace(self.compound, self.tire_age)
            self.tire_age += 1
        else:
            my_time = calculate_pace(action, 0) + PIT_LOSS
            self.compound = action
            self.tire_age = 1
            if self.my_pit_count < NUM_MY_PITS - 1:
                self.my_pit_count += 1
            if self.compound == COMP_SOFT:
                self.used_soft = True
            elif self.compound == COMP_MED:
                self.used_med = True
            self.mandatory_ok = int(self.used_soft and self.used_med)

        # Opponent Move
        opp_action = self._opponent_strategy()
        if opp_action == ACTION_STAY:
            opp_time = calculate_pace(self.opp_compound, self.opp_tire_age)
            self.opp_tire_age += 1
        else:
            opp_time = calculate_pace(opp_action, 0) + PIT_LOSS
            self.opp_compound = opp_action
            self.opp_tire_age = 1

        delta = opp_time - my_time
        self.gap += delta

        current_reward = delta*30

        newly_satisfied = (self.mandatory_ok == 1 and not self.mandatory_before)
        if newly_satisfied and self.lap <= 8:
            # reward for finishing the compound requirement early
            current_reward += 500.0

        if action != ACTION_STAY and action == self.compound:
            current_reward -= 2000.0  # Same compound pit

        if action == ACTION_STAY and self.tire_age > TIRE_MAX_LIFE[self.compound]:
            current_reward -= 5000.0  # Staying on DEAD tires

        self.lap += 1
        race_done = (self.lap > TOTAL_RACE_LAPS)

        if race_done:
            if not self.mandatory_ok:
                current_reward += MANDATORY_PENALTY
            if self.gap > 0:
                current_reward += 8000.0  # Win bonus
            else:
                current_reward -= 6000.0  # Loss penalty

        if action != ACTION_STAY:
            if self.my_pit_count >= 1:  # already had at least one stop
                current_reward += EXTRA_PIT_PENALTY
            if self.my_pit_count >= 9:
                current_reward += (EXTRA_PIT_PENALTY*10)

        over = max(0, self.tire_age - TIRE_MAX_LIFE[self.compound])
        current_reward -= 50.0 * over

        return self.get_state(), current_reward, race_done, my_time, opp_time

    def _opponent_strategy(self):
        #Stochastic opponent strategy per episode
        opp_action = ACTION_STAY
        max_life = TIRE_MAX_LIFE[self.opp_compound]
        life_pct = self.opp_tire_age / max_life if max_life > 0 else 0

        p = self.opp_params

        if self.opp_compound == COMP_MED:
            if self.tire_age <= 3 and self.opp_tire_age > 10 and self.gap > p["defend_gap"]:
                opp_action = COMP_SOFT
            elif self.gap < p["attack_gap"] and self.opp_tire_age > 11:
                opp_action = COMP_SOFT
            elif life_pct > p["med_life_pct_pit"]:
                opp_action = COMP_SOFT
            elif abs(self.gap) < 2.0 and self.opp_tire_age > 12:
                opp_action = COMP_SOFT
        elif self.opp_compound == COMP_SOFT:
            if life_pct > p["soft_life_pct_pit"]:
                opp_action = COMP_MED
            elif self.opp_tire_age > 9:
                opp_action = COMP_MED

        return opp_action

# ALGORITHM IMPLEMENTATIONS

def select_action_epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        s_idx = state_to_index(state)
        q_values = Q[s_idx, :]
        max_val = np.max(q_values)
        ties = np.where(q_values == max_val)[0]
        return np.random.choice(ties)


def calculate_max_norm_change(Q_old, Q_new):
    #Calculate max-norm (L-infinity) between two Q-tables
    return np.max(np.abs(Q_new - Q_old))

def run_sarsa_exploring_starts(env, gamma, alpha, num_episodes, log_every=5000):
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))

    epsilon = 1.0
    min_epsilon = 0.05
    decay = 0.99998

    episode_rewards = []
    win_history = []
    pits_history = []
    q_changes = []
    Q_old = np.copy(Q)

    print("Running SARSA with Exploring Starts...")

    for ep in range(num_episodes):
        state = env.reset_exploring_start() if random.random() < 0.8 else env.reset()
        action = select_action_epsilon_greedy(Q, state, epsilon)

        total_reward = 0.0
        pits_this_ep = 0
        steps = 0
        race_done = False
        max_steps = TOTAL_RACE_LAPS + 10

        while not race_done and steps < max_steps:
            next_state, reward, race_done, my_time, opp_time = env.step(action)
            total_reward += reward
            if action != ACTION_STAY:
                pits_this_ep += 1

            if race_done:
                target = reward
                next_action = None
            else:
                next_action = select_action_epsilon_greedy(Q, next_state, epsilon)
                ns_idx = state_to_index(next_state)
                target = reward + gamma * Q[ns_idx, next_action]

            s_idx = state_to_index(state)
            td_error = target - Q[s_idx, action]
            Q[s_idx, action] += alpha * td_error

            state = next_state
            action = next_action if next_action is not None else 0
            steps += 1

        episode_rewards.append(total_reward)
        win_history.append(1 if env.gap > 0 else 0)
        pits_history.append(pits_this_ep)

        if (ep + 1) % 500 == 0:
            q_change = calculate_max_norm_change(Q_old, Q)
            q_changes.append(q_change)
            Q_old = np.copy(Q)

        if (ep + 1) % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            win_rate = np.mean(win_history[-log_every:])
            avg_pits = int(np.mean(pits_history[-log_every:]))
            # recent_change = q_changes[-1] if q_changes else 0.0
            print(f" Episode {ep+1}: Eps {epsilon:.3f}, "
                  f"Reward {avg_reward:.1f}, "
                  f"WinRate {win_rate:.2%}, "
                  f"Pits {avg_pits:.2f}")

        epsilon = max(min_epsilon, epsilon * decay)

    return Q, episode_rewards, win_history, pits_history, q_changes



def run_q_learning_exploring_starts(env, gamma, alpha, num_episodes, log_every=1000):
    #Off-policy Q-Learning with Exploring Starts (flat Q-table).
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))

    epsilon = 1.0
    min_epsilon = 0.05
    decay = 0.99997

    episode_rewards = []
    win_history = []
    pits_history = []
    q_changes = []
    Q_old = np.copy(Q)

    print("Running Q-Learning with Exploring Starts...")

    for ep in range(num_episodes):
        state = env.reset_exploring_start() if random.random() < 0.8 else env.reset()

        total_reward = 0.0
        pits_this_ep = 0
        steps = 0
        race_done = False
        max_steps = TOTAL_RACE_LAPS + 10

        while not race_done and steps < max_steps:
            # epsilon-greedy behaviour policy
            s_idx = state_to_index(state)
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = int(np.argmax(Q[s_idx, :]))

            if action != ACTION_STAY:
                pits_this_ep += 1

            next_state, reward, race_done, _, _ = env.step(action)
            total_reward += reward

            ns_idx = state_to_index(next_state) if not race_done else None
            best_next = 0.0 if race_done else np.max(Q[ns_idx, :])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[s_idx, action]
            Q[s_idx, action] += alpha * td_error

            state = next_state
            steps += 1

        episode_rewards.append(total_reward)
        win_history.append(1 if env.gap > 0 else 0)
        pits_history.append(pits_this_ep)

        if (ep + 1) % 500 == 0:
            q_change = calculate_max_norm_change(Q_old, Q)
            q_changes.append(q_change)
            Q_old = np.copy(Q)

        if (ep + 1) % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            win_rate = np.mean(win_history[-log_every:])
            avg_pits = int(np.mean(pits_history[-log_every:]))

            recent_change = q_changes[-1] if q_changes else 0.0
            print(f" Episode {ep+1}: Eps {epsilon:.3f}, Reward {avg_reward:.1f}, "
                  f"WinRate {win_rate:.2%}, Pits {avg_pits:.2f}")

        epsilon = max(min_epsilon, epsilon * decay)

    return Q, episode_rewards, win_history, pits_history, q_changes


def run_monte_carlo_exploring_starts(env, gamma, num_episodes, log_every=1000):
    #First-visit Monte Carlo control with Exploring Starts (flat Q-table).
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    N = np.zeros_like(Q)

    epsilon = 1.0
    min_epsilon = 0.05
    decay = 0.99996

    episode_rewards = []
    win_history = []
    pits_history = []
    q_changes = []
    Q_old = np.copy(Q)

    print("Running Monte Carlo with Exploring Starts...")

    for ep in range(num_episodes):
        state = env.reset_exploring_start() if random.random() < 0.8 else env.reset()

        episode = []
        done = False
        steps = 0
        pits_this_ep = 0
        max_steps = TOTAL_RACE_LAPS + 10

        while not done and steps < max_steps:
            s_idx = state_to_index(state)
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = int(np.argmax(Q[s_idx, :]))

            if action != ACTION_STAY:
                pits_this_ep += 1

            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        # compute returns
        G = 0.0
        episode_returns = []
        for (s, a, r) in reversed(episode):
            G = r + gamma * G
            episode_returns.insert(0, (s, a, G))

        # first-visit updates
        seen = set()
        for (s, a, G_t) in episode_returns:
            key = (s, a)
            if key in seen:
                continue
            seen.add(key)

            s_idx = state_to_index(s)
            N[s_idx, a] += 1
            Q[s_idx, a] += (G_t - Q[s_idx, a]) / N[s_idx, a]

        total_reward = sum(r for _, _, r in episode)
        episode_rewards.append(total_reward)
        win_history.append(1 if env.gap > 0 else 0)
        pits_history.append(pits_this_ep)

        if (ep + 1) % 500 == 0:
            q_change = calculate_max_norm_change(Q_old, Q)
            q_changes.append(q_change)
            Q_old = np.copy(Q)

        if (ep + 1) % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            win_rate = np.mean(win_history[-log_every:])
            avg_pits = int(np.mean(pits_history[-log_every:]))
            print(f" Episode {ep+1}: Eps {epsilon:.3f}, Reward {avg_reward:.1f}, "
                  f"WinRate {win_rate:.2%}, Pits {avg_pits:.2f}")

        epsilon = max(min_epsilon, epsilon * decay)

    return Q, episode_rewards, win_history, pits_history, q_changes

#Policy Evaluation

def save_policy_to_csv(Q, filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        # header
        writer.writerow([
            "lap", "compound", "my_tyre", "opp_tyre",
            "gap", "my_pits", "mandatory_compound",
            "action", "is_valid_or_reachable_state"
        ])

        for idx, state in enumerate(STATE_LIST):
            lap, c, t, ot, g, mp, mand = state
            q_vals = Q[idx, :]
            best_a = int(np.argmax(q_vals))
            best_action_name = ACTION_SYMBOLS[best_a]

            valid = 1 if is_valid_state(state) else 0

            writer.writerow([
                lap,
                COMPOUND_LABELS[c],
                MY_TIRE_LABELS[t],
                OPP_TIRE_LABELS[ot],
                GAP_LABELS[g],
                mp,
                mand,
                best_action_name,
                valid
            ])

    print(f"Policy written to {filepath}")


def evaluate_policy_vs_opponent(env, Q, num_races=500):
    # Evaluate learned policy against opponent
    wins = 0
    total_gap = 0.0
    total_pits = 0

    for _ in range(num_races):
        state = env.reset()
        done = False
        steps = 0
        max_steps = TOTAL_RACE_LAPS + 10
        pits_this_race = 0

        while not done and steps < max_steps:
            s_idx = state_to_index(state)
            action = int(np.argmax(Q[s_idx, :]))

            next_state, reward, done, my_time, opp_time = env.step(action)
            if action != ACTION_STAY:
                pits_this_race += 1

            state = next_state
            steps += 1

        total_pits += pits_this_race
        if env.gap > 0:
            wins += 1
        total_gap += env.gap

    win_rate = wins / num_races
    avg_gap = total_gap / num_races
    avg_pits = int(total_pits / num_races)
    return win_rate, avg_gap, avg_pits


# VISUALIZATION

def run_logged_race(env, Q):

    state = env.reset()
    total_reward = 0.0
    my_pits = 0
    header_fmt = "{:<3} {:<10} {:<6} {:<14} {:<12} {:>8} {:<6} {:<18} {:<6}"
    row_fmt = "{:<3} {:<10} {:<6} {:<14} {:<12} {:+8.2f} {:<6} {:<18} {:<6}"

    print(header_fmt.format("Lap", "Compound", "MyAge", "OpponentComp", "OpponentAge", "Gaps", "Pits", "MandatoryCompound", "Action"))

    done = False
    while not done:
        s_idx = state_to_index(state)

        action = int(np.argmax(Q[s_idx, :]))

        action_name = ACTION_SYMBOLS[action]

        lap, comp, my_tire, opp_tire, gap_disc, mp, mand = state

        print(row_fmt.format(
            lap,
              COMPOUND_LABELS[comp],
              env.tire_age,
              COMPOUND_LABELS[env.opp_compound-1],
              env.opp_tire_age,
              env.gap,
              mp,
              mand,
              action_name))

        next_state, reward, done, my_time, opp_time = env.step(action)
        total_reward += reward

        if action != ACTION_STAY:
            my_pits += 1

        state = next_state

    print(f"\nRace finished. Total pits: {my_pits}, total reward: {total_reward:.1f}")
    print(f"Final gap: {env.gap:+.2f}s, mandatory_ok={env.mandatory_ok}, dnf={env.dnf}")
    return my_pits, total_reward, env.gap


def is_valid_state(state):

    lap, c, t, ot, g, mp, mand = state

    # Cannot have more pit stops than completed laps
    if mp > lap:
        return False

    # mandatory_ok == 1 requires at least one stop
    if mp == 0 and mand == 1:
        return False

    # 0 = Fresh, 1 = Worn, 2 = DEAD
    # At race start (lap 0) tyres must be fresh
    # At race start (lap 1) tyres must be fresh
    if lap == 1 and t != 0:
        return False

    # Very early laps (1–2) with DEAD tyres are artefacts
    if lap <= 2 and t == 2:
        return False

    # Opponent:
    if lap == 1 and ot != 0:
        return False
    if lap <= 2 and ot == 2:
        return False

    # 2+ pits extremely early (laps 1–2)
    if lap <= 2 <= mp:
        return False

    if mand == 1 and lap <= 2 and t == 2:
        return False

    return True


def moving_mean_std(x, window):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    means = np.empty_like(x, dtype=float)
    stds = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window)
        w = x[start:i+1]
        means[i] = w.mean()
        stds[i] = w.std()
    idx = np.arange(0, len(x), PLOT_SUBSAMPLE)
    return means[idx], stds[idx], idx

def plot_learning_curves_separate_with_std(results_dict, window=PLOT_WINDOW):

    for name, (rewards, wins, pits, q_changes) in results_dict.items():
        # Reward
        mean_r, std_r, idx_r = moving_mean_std(rewards, window)
        if len(idx_r) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(idx_r, mean_r, linewidth=2)
            plt.fill_between(idx_r, mean_r - std_r, mean_r + std_r, alpha=0.2)
            plt.title(f"{name} – Reward (mean ± std, w={window})")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Win rate
        mean_w, std_w, idx_w = moving_mean_std(wins, window)
        if len(idx_w) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(idx_w, mean_w, linewidth=2)
            plt.fill_between(idx_w, mean_w - std_w, mean_w + std_w, alpha=0.2)
            plt.axhline(0.5, color="red", linestyle="--", linewidth=1.5)
            plt.title(f"{name} – Win rate (mean ± std, w={window})")
            plt.xlabel("Episode")
            plt.ylabel("Win rate")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Pits
        mean_p, std_p, idx_p = moving_mean_std(pits, window)
        if len(idx_p) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(idx_p, mean_p, linewidth=2)
            plt.fill_between(idx_p, mean_p - std_p, mean_p + std_p, alpha=0.2)
            plt.title(f"{name} – Pits (mean ± std, w={window})")
            plt.xlabel("Episode")
            plt.ylabel("Pits per episode")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


#MAIN BLOCK

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Hyperparameters
    GAMMA = 0.997
    ALPHA = 0.01
    NUM_EPISODES = 500000
    LOG_EVERY = 5000

    env = F1SimpleEnv()

    print("Configuration:")
    print(f"  Num states: {NUM_STATES}")
    print(f"  Num actions: {NUM_ACTIONS}")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Gamma: {GAMMA}, Alpha: {ALPHA}")
    print()

    results_learning = {}   # name -> (rewards, wins, q_changes)
    Q_tables = {}           # name -> Q

    # ---------- SARSA ----------
    print("[1/3] SARSA with Exploring Starts")
    print("-" * 80)
    Q_sarsa, rew_sarsa, wins_sarsa, pits_sarsa, qchg_sarsa = run_sarsa_exploring_starts(env, GAMMA, ALPHA, NUM_EPISODES, LOG_EVERY)
    results_learning["SARSA"] = (rew_sarsa, wins_sarsa, pits_sarsa, qchg_sarsa)
    Q_tables["SARSA"] = Q_sarsa

    # ---------- Q-Learning ----------
    print("\n[2/3] Q-Learning with Exploring Starts")
    print("-" * 80)
    Q_q, rew_q, wins_q, pits_q, qchg_q = run_q_learning_exploring_starts(env, GAMMA, ALPHA, NUM_EPISODES, LOG_EVERY)
    results_learning["Q-Learning"] = (rew_q, wins_q, pits_q, qchg_q)
    Q_tables["Q-Learning"] = Q_q

    # ---------- Monte Carlo ----------
    print("\n[3/3] Monte Carlo with Exploring Starts")
    print("-" * 80)
    Q_mc, rew_mc, wins_mc, pits_mc, qchg_mc = run_monte_carlo_exploring_starts(env, GAMMA, NUM_EPISODES, LOG_EVERY)
    results_learning["Monte Carlo"] = (rew_mc, wins_mc, pits_mc, qchg_mc)
    Q_tables["Monte Carlo"] = Q_mc

    # ---------- Evaluation vs opponent ----------
    print("\nEVALUATION PHASE")
    print("-" * 80 + "\n")

    results_eval = {}

    for name, Q in Q_tables.items():
        print(f"Evaluating {name}...")
        win_rate, avg_gap, avg_pits = evaluate_policy_vs_opponent(env, Q, num_races=500)
        final_reward = np.mean(results_learning[name][0][-1000:])

        results_eval[name] = {
            "win_rate": win_rate,
            "avg_gap": avg_gap,
            "avg_pits": avg_pits,
            "final_reward": final_reward,
        }

        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Gap: {avg_gap:.2f}s")
        print(f"  Avg Pits: {avg_pits:.2f}")
        print(f"  Final Avg Reward (last 1000): {final_reward:.1f}")
        print()

    # ---------- Learning curves ----------
    plot_learning_curves_separate_with_std(results_learning, window=500)

    save_policy_to_csv(Q_tables["SARSA"], "policy_sarsa.csv")
    save_policy_to_csv(Q_tables["Q-Learning"], "policy_qlearning.csv")
    save_policy_to_csv(Q_tables["Monte Carlo"], "policy_montecarlo.csv")

    # ---------- Logged race for each algorithm ----------
    for name, Q in Q_tables.items():
        print(f"\nLOGGED RACE – {name}")
        env = F1SimpleEnv()
        run_logged_race(env, Q)


