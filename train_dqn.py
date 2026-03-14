import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import random
import matplotlib.pyplot as plt
import env_fpl_simulator
from tqdm import tqdm

from dqn_agent import DQN, ReplayBuffer, to_tensor
from env_fpl_simulator import SoloFPLTransferEnv
from data_loader import load_fpl_data

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EPISODES = 100  # Reduced for CPU
BATCH_SIZE = 16
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE_FREQ = 10
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
BUFFER_CAPACITY = 5000

# Load data and environment
df = load_fpl_data("cleaned_fpl_data_2021-22.csv")

if "value" not in df.columns and "now_cost" in df.columns:
    df["value"] = df["now_cost"] / 10.0

env = SoloFPLTransferEnv(df)
input_dim = 15 * 6  # 15 players, 6 stats each
output_dim = 1

# Initialize policy and target networks
policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_CAPACITY)
epsilon = EPS_START

def generate_trade_options(df, current_player, gw, top_k=3):
    if "round" in df.columns:
        player_rows = df[(df["name"] == current_player) & (df["round"] == gw)]
        available_pool = df[(df["round"] == gw)]
    else:
        player_rows = df[df["name"] == current_player]
        available_pool = df.copy()

    if player_rows.empty:
        return []

    current_pos = player_rows.iloc[0]["position"]

    # Only trade within same position
    same_pos_pool = available_pool[available_pool["position"] == current_pos]
    candidate_names = same_pos_pool["name"].unique().tolist()

    return [ (current_player, name) for name in candidate_names if name != current_player ]

def state_to_tensor(state, df, gw):
    player_rows = []
    for name in state:
        if "round" in df.columns:
            row = df[(df["name"] == name) & (df["round"] == gw)].head(1)
        else:
            row = df[df["name"] == name].head(1)

        if row.empty:
            player_rows.append(np.zeros(6))  # fallback if player missing
        else:
            player_rows.append(row[["minutes", "goals_scored", "assists", "influence", "ict_index", "value"]].values[0])
    
    return np.array(player_rows).flatten()
def flatten_squad(squad):
    return [player for pos in squad for player in squad[pos]]

def sample_random_action(env):
    trades = []
    for player in flatten_squad(env.squad):
        options = generate_trade_options(env.df, player, env.current_gw)
        trades.extend(options)  # options are already (out, in) tuples
    return random.choice(trades) if trades else None

# === Training Loop ===
all_rewards = []
all_losses = []

for episode in tqdm(range(EPISODES)):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_vec = state_to_tensor(state, df, env.current_gw)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        if random.random() < epsilon:
            action = sample_random_action(env)
        else:
            with torch.no_grad():
                q_val = policy_net(state_tensor).item()
            action = env.select_best_action_based_on_state(q_val)

        if action is not None:
            next_state, reward, done, _ = env.step(action)
        else:
            print("No valid action found — skipping env.step()")
            next_state, reward, done = env.state, 0, True  # fallback or default values
        next_state_vec = state_to_tensor(next_state, df, env.current_gw)

        buffer.push(state_vec, 0, reward, next_state_vec, done)
        total_reward += reward

        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_vals = policy_net(states)
            next_q_vals = target_net(next_states).detach()
            targets = rewards + GAMMA * next_q_vals * (1 - dones)

            loss = nn.MSELoss()(q_vals, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())

        state = next_state

    all_rewards.append(total_reward)
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode+1} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")