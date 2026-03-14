import pandas as pd
import numpy as np
import random
from copy import deepcopy


def build_valid_fpl_squad(df, gw, min_minutes=30):
    if "round" in df.columns:
        gw_df = df[(df["round"] == gw) & (df["minutes"] >= min_minutes)].copy()
    else:
        gw_df = df[df["minutes"] >= min_minutes].copy()

    def select_players(pos, count):
        pool = gw_df[gw_df["position"] == pos]
        return list(pool.sample(n=count, replace=False)["name"])

    squad = {
        "GK": select_players("GK", 2),
        "DEF": select_players("DEF", 5),
        "MID": select_players("MID", 5),
        "FWD": select_players("FWD", 3)
    }
    return squad


def flatten_squad(squad):
    return [player for pos in squad for player in squad[pos]]


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


class SoloFPLTransferEnv:
    def __init__(self, df, gw_start=1):
        self.df = df
        self.current_gw = gw_start
        self.squad = build_valid_fpl_squad(df, self.current_gw)
        self.state = flatten_squad(self.squad)
        self.history = []

    def reset(self):
        self.current_gw = 1
        self.squad = build_valid_fpl_squad(self.df, self.current_gw)
        self.state = flatten_squad(self.squad)
        self.history = []
        return self.state

    def step(self, action):
        player_out, player_in = action
        squad_flat = flatten_squad(self.squad)

        if "round" in self.df.columns:
            curr_df = self.df[self.df["round"] == self.current_gw]
        else:
            curr_df = self.df.copy()  # fallback to full-season stats

        original_points = curr_df[curr_df["name"].isin(squad_flat)]["total_points"].sum()

        for pos in self.squad:
            if player_out in self.squad[pos]:
                self.squad[pos].remove(player_out)
                self.squad[pos].append(player_in)
                break

        new_flat = flatten_squad(self.squad)
        new_points = curr_df[curr_df["name"].isin(new_flat)]["total_points"].sum()
        reward = new_points - original_points

        self.history.append((self.current_gw, player_out, player_in, reward))
        self.current_gw += 1

        if "round" in self.df.columns:
            done = self.current_gw > self.df["round"].max()
        else:
            done = self.current_gw > 1  # Only one step when using full-season data

        self.state = new_flat
        return self.state, reward, done, {}

    def valid_trade_actions(self):
        trades = []
        for player in flatten_squad(self.squad):
            options = generate_trade_options(self.df, player, self.current_gw)
            trades.extend(options)  # options is already a list of (out, in) pairs
        return trades
    
    def select_best_action_based_on_state(self, q_value):
        """
        Placeholder strategy: return the first valid trade.
        Later, replace with logic that uses q_value to select best trade.
        """
        valid_actions = self.valid_trade_actions()
        if valid_actions:
            return valid_actions[0]  # Replace with smarter selection if needed
        else:
            return None