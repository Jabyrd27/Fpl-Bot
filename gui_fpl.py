import streamlit as st
import pandas as pd
import requests
from itertools import combinations

from main_fplBot import team_rosters, manager_names
from fpl_predict_model import train_model

# === Load models and predictions ===
@st.cache_resource
def load_models_and_data():
    short_model, short_df = train_model(mode="short")
    long_model, long_df = train_model(mode="long")
    return short_model, short_df, long_model, long_df

short_model, short_df, long_model, long_df = load_models_and_data()
current_gw = 34

# === Load bootstrap data for teams and players ===
bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
players_df = pd.DataFrame(bootstrap["elements"])
teams_df = pd.DataFrame(bootstrap["teams"])

# Map team ID → team name
team_id_to_name = dict(zip(teams_df["id"], teams_df["name"]))

# Player enrichment (already exists in your code)
players_df["position"] = players_df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
players_df["name"] = players_df["web_name"]

# Make sure team and position info are merged
def enrich_with_static_info(df_input):
    df_input = df_input.copy()
    df_input["name"] = df_input.get("web_name", df_input.get("name"))
    df_input = df_input.loc[:, ~df_input.columns.duplicated()]
    merged = df_input.merge(players_df[["name", "team", "position"]], on="name", how="left")
    merged["team_name"] = merged["team"].map(team_id_to_name)  
    return merged

short_df = enrich_with_static_info(short_df)
long_df = enrich_with_static_info(long_df)

# === Merge long-term and short-term predictions ===
df = long_df.copy()
# bring in the real total_points from the bootstrap snapshot
df = df.merge(
     players_df[["name", "total_points"]],
     on="name",
     how="left"
)
# coerce any missing back to zero, and make sure it's an int
df["total_points"] = (
    pd.to_numeric(df["total_points"], errors="coerce")
    .fillna(0)
    .astype(int)
)

next_pred_df = short_df.copy()

# standardize the name column
if "web_name" in next_pred_df.columns:
    next_pred_df = next_pred_df.rename(columns={"web_name": "name"})

# pick whichever column holds your next-GW forecasts
pred_col = "predicted_next" if "predicted_next" in next_pred_df.columns else "predicted_points"
next_pred_df = next_pred_df[["name", pred_col]].rename(columns={pred_col: "predicted_next"})

# Remove duplicates before merging
df = df.loc[:, ~df.columns.duplicated()]
next_pred_df = next_pred_df.loc[:, ~next_pred_df.columns.duplicated()]

df = df.merge(next_pred_df, on="name", how="left")
df["predicted_next"] = pd.to_numeric(df["predicted_next"], errors="coerce").fillna(0.0)
df = df.drop_duplicates(subset=["name"], keep="first")

# === Sidebar: Select team ===
st.sidebar.title("FPL Assistant")
selected_team = st.sidebar.selectbox("Select your team:", list(team_rosters.keys()),
                                     format_func=lambda k: manager_names.get(k, f"Team {k}"))
user_roster_names = team_rosters[selected_team]
st.sidebar.markdown(f"**My Team:** *{manager_names.get(selected_team, selected_team)}*")

st.title("My FPL Team Dashboard")

# === Tabs ===
tab_my_team, tab_free_agents, tab_trade_suggestions, tab_trade_analyzer = st.tabs(
    ["My Team", "Free Agents", "Trade Suggestions", "Trade Analyzer"]
)

# --- TAB 1: My Team ---
with tab_my_team:
    st.subheader("Current Roster and Projections")

    user_team_df = df[df["name"].isin(user_roster_names)].copy()
    user_team_df["total_predicted_points"] = user_team_df["total_points"] + user_team_df["predicted_points"]

    if not user_team_df.empty:
        st.dataframe(user_team_df[["name", "team_name", "position",
                                   "total_points", "predicted_next", "predicted_points", "total_predicted_points"]]
                     .rename(columns={
                         "name": "Name", "team_name": "Team", "position": "Pos",
                         "total_points": "Points So Far",
                         "predicted_next": f"Predicted GW{current_gw + 1}",
                         "predicted_points": "Predicted ROS",
                         "total_predicted_points": "Projected Total"
                     }))
        st.metric(label=f"Predicted GW{current_gw + 1} Team Score",
                  value=f"{user_team_df['predicted_next'].sum():.1f} pts")
    else:
        st.warning("No players found for this team.")

# --- TAB 2: Free Agents ---
with tab_free_agents:
    st.subheader(f"Top Free Agents for GW{current_gw + 1}")

    owned_players = set(player for roster in team_rosters.values() for player in roster)
    free_agents_df = df[~df["name"].isin(owned_players)].copy()

    pos_options = ["DEF", "MID", "FWD"]
    selected_pos = st.selectbox("Select Position Group", pos_options)

    pos_free_agents = free_agents_df[free_agents_df["position"] == selected_pos]
    top_pos_free_agents = pos_free_agents.sort_values(by="predicted_next", ascending=False).head(10)

    if top_pos_free_agents.empty:
        st.warning("No free agents found for this position group.")
    else:
        st.table(top_pos_free_agents[["name", "team_name", "predicted_next", "total_points"]]
                 .rename(columns={
                     "name": "Player", "team_name": "Team",
                     "predicted_next": f"GW{current_gw + 1} Predicted",
                     "total_points": "Points So Far"
                 }))
        # --- Free Agent Comparison Tool ---
    st.markdown("---")
    st.subheader("Waiver Comparison")


    # Step 1: Select your own player first
    my_selected = st.selectbox("Select a Player From Your Team", options=user_roster_names)

    if my_selected:
        my_row = df[df["name"] == my_selected].iloc[0]
        my_pos = my_row["position"]
        my_team_name = my_row.get("team_name", "N/A")

        # Step 2: Filter free agents by same position
        matching_free_agents = free_agents_df[free_agents_df["position"] == my_pos]
        sorted_match_fa = matching_free_agents.sort_values(by="predicted_next", ascending=False).head(10)
        fa_selected = st.selectbox(f"Compare With Free Agent ({my_pos})", options=sorted_match_fa["name"].unique())

        if fa_selected:
            fa_row = df[df["name"] == fa_selected].iloc[0]

            # Display stacked comparison
            st.markdown("### My Player")
            st.table(pd.DataFrame([{
                "Player": my_selected,
                "Team": my_team_name,
                "Position": my_pos,
                f"GW{current_gw + 1} Pred": my_row["predicted_next"],
                "ROS Pred": my_row["predicted_points"]
            }]))

            st.markdown("### Free Agent")
            st.table(pd.DataFrame([{
                "Player": fa_selected,
                "Team": fa_row.get("team_name", "N/A"),
                "Position": fa_row["position"],
                f"GW{current_gw + 1} Pred": fa_row["predicted_next"],
                "ROS Pred": fa_row["predicted_points"]
            }]))

            # Net differences
            st.markdown("### Net Gain if You Swap")
            net_short = fa_row["predicted_next"] - my_row["predicted_next"]
            net_long = fa_row["predicted_points"] - my_row["predicted_points"]

            st.metric(label="Short-Term Gain", value=f"{net_short:+.1f} pts")
            st.metric(label="Long-Term Gain", value=f"{net_long:+.1f} pts")


# --- Trade Suggestions Tab ---
with tab_trade_suggestions:
    st.subheader("Trade Proposal: Package Deal for a Premium")

    if not user_roster_names:
        st.warning("Please select your team.")
    
    else:
        user_team = df[df["name"].isin(user_roster_names) & ~df["position"].isin(["GK"])].copy()

        best_trade = None
        best_gain = -float("inf")

        for manager_id, their_roster in team_rosters.items():
            if manager_id == selected_team:
                continue  # Skip your own team

            their_team = df[df["name"].isin(their_roster) & ~df["position"].isin(["GK"])].copy()
            if their_team.empty or user_team.empty:
                continue

            # Try their top 1–3 players as trade targets
            for _, star in their_team.sort_values(by="predicted_points", ascending=False).head(3).iterrows():
                star_name = star["name"]
                star_ros = float(star["predicted_points"])
                star_gw = float(star.get("predicted_next", 0.0))

                fillers_pool = their_team[their_team["name"] != star_name].nsmallest(2, "predicted_points")
                if fillers_pool.shape[0] < 2:
                    continue

                fillers_list = [
                    {
                        "name": row["name"],
                        "ros":  float(row["predicted_points"]),
                        "gw":   float(row.get("predicted_next", 0.0))
                    }
                    for _, row in fillers_pool.iterrows()
                ]

                sterm_fill = sum(player['gw'] for player in fillers_list)

                for out_combo in combinations(user_team.index, 3):
                    give_df = user_team.loc[list(out_combo)]
                    give_total = give_df["predicted_points"].sum()

                    get_total = star_ros + fillers_pool["predicted_points"].sum()
                    net_gain = get_total - give_total
                    

                    if net_gain <= 10.0 and net_gain > best_gain:

                        give_list = [
                            {
                                "name": row["name"],
                                "ros":  float(row["predicted_points"]),
                                "gw":   float(row.get("predicted_next", 0.0))
                            }
                            for _, row in give_df.iterrows()
                        ]

                        sterm_give = sum(player['gw'] for player in give_list)
                        net_sterm = (sterm_fill + star_gw) - sterm_give
                        best_gain = net_gain
                        best_trade = {
                            "Target": star_name,
                            "Target ROS": star_ros,
                            "Target GW": star_gw,
                            "You Give": give_list,
                            "Give ROS": round(give_total, 1),
                            "Get ROS": star_ros + round(fillers_pool["predicted_points"].sum(), 1),
                            "Get Fillers": fillers_list,
                            "Net Gain": round(net_gain, 2),
                            "Net Sterm" : round(net_sterm, 2),
                            "From Manager": manager_names.get(str(manager_id), f"Manager {manager_id}")
                        }
        

        if not best_trade:
            st.info("No fair and realistic trade found. Try again later.")
        else:
            st.markdown(f"### Target: **{best_trade['Target']}** (owned by {best_trade['From Manager']})")
            st.write(f"- Projected ROS: {best_trade['Target ROS']:.1f} pts")
            st.write(f"- Projected GW{current_gw + 1}: {best_trade['Target GW']:.1f} pts")

            st.markdown("### You Give (3 Players)")
            give_md = ""
            for p in best_trade["You Give"]:
                give_md += (
                    f"- {p['name']}\n"
                    f"    - ROS: {p['ros']:.1f} pts\n"
                    f"    - GW{current_gw + 1}: {p['gw']:.1f} pts\n"
                )
            st.write(f"##### Total ROS Given: *{best_trade['Give ROS']}*")
            st.markdown(give_md)

            st.markdown("### You Receive (2 Fillers)")

            get_md = ""
            for p in best_trade["Get Fillers"]:
                get_md += (
                    f"- {p['name']}\n"
                    f"    - ROS: {p['ros']:.1f} pts\n"
                    f"    - GW{current_gw + 1}: {p['gw']:.1f} pts\n"
                )
            st.write(f"##### Total ROS Recieved: *{best_trade['Get ROS']}*")
            st.markdown(get_md)
            
            if net_sterm >= 0 :
                st.success(f"Short-Term Gain: {best_trade['Net Sterm']} points")
            else:
                st.error(f"Short-Term Loss: {best_trade['Net Sterm']} points")

            st.success(f"Net Long-Term Gain: {best_trade['Net Gain']} points")

# --- Trade Analyzer Tab ---
with tab_trade_analyzer:
    st.subheader("Trade Analyzer – Compare Hypothetical Trades")
    if not user_roster_names:
        st.write("Please select your team to use the trade analyzer.")
    else:
        # Select an opposing team to trade with
        opponent_options = [tid for tid in team_rosters.keys() if tid != selected_team]
        opponent_team = st.selectbox("Select another manager's team:", opponent_options, 
                                     format_func=lambda k: manager_names.get(k, f"Team {k}"))
        opponent_roster_names = team_rosters[opponent_team]
        # Multi-select players for the trade scenario
        out_players = st.multiselect(f"Players to Trade Away (from {manager_names.get(selected_team, 'User')}):", user_roster_names)
        in_players = st.multiselect(f"Players to Trade For (from {manager_names.get(opponent_team, 'Opponent')}):", opponent_roster_names)
        # Ensure both sides selected
        if not out_players or not in_players:
            st.info("Select at least one player from **your team** and **the other team** to analyze the trade.")
        else:
            # Calculate short-term and long-term projections for the selected players
            out_df = df[df["name"].isin(out_players)].copy()
            in_df = df[df["name"].isin(in_players)].copy()
            # Sum up projections for each side
            out_short_sum = out_df["predicted_next"].sum()
            out_long_sum = out_df["predicted_points"].sum()
            in_short_sum = in_df["predicted_next"].sum()
            in_long_sum = in_df["predicted_points"].sum()
            net_short = float(in_short_sum - out_short_sum)
            net_long = float(in_long_sum - out_long_sum)
            # Display side-by-side comparison of players involved
            col1, col2 = st.columns(2)
            col1.markdown(f"**Players Out ({manager_names.get(selected_team, 'User')})**")
            col1.table(out_df[["name", "position", "predicted_next", "predicted_points"]]
                       .rename(columns={
                           "name": "Player", "position": "Pos", 
                           "predicted_next": f"GW{current_gw+1} Pred", 
                           "predicted_points": "ROS Pred"
                       }))
            col2.markdown(f"**Players In ({manager_names.get(opponent_team, 'Other Team')})**")
            col2.table(in_df[["name", "position", "predicted_next", "predicted_points"]]
                       .rename(columns={
                           "name": "Player", "position": "Pos", 
                           "predicted_next": f"GW{current_gw+1} Pred", 
                           "predicted_points": "ROS Pred"
                       }))
            # Show net gains/losses in short-term and long-term
            net_col1, net_col2 = st.columns(2)
            if net_short >= 0:
                net_col1.success(f"Short-Term Net: +{net_short:.1f} pts")
            else:
                net_col1.error(f"Short-Term Net: {net_short:.1f} pts")
            if net_long >= 0:
                net_col2.success(f"Long-Term Net: +{net_long:.1f} pts")
            else:
                net_col2.error(f"Long-Term Net: {net_long:.1f} pts")
            st.caption("Net values indicate how many points you gain or lose by making this trade. Short-term = next gameweek; Long-term = rest of season.")