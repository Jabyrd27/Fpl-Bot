import requests
import pandas as pd
from pprint import pprint
import json

team_rosters = {
    "41451": ["Arrizabalaga", "Rúben", "Digne", "Livramento",
              "De Bruyne", "Savinho", "Sancho", "Enzo", "Semenyo",
              "Welbeck", "Watkins", "Vicario", "Ferguson", "Van Hecke", "N.Semedo"],
    
    "41624": ["Raya", "Virgil", "J.Timber", "Aït-Nouri", "Todibo",
              "Kulusevski", "Luis Díaz", "Tielemans", "Kudus",
              "N.Jackson", "Isak", "Kelleher", "Minteh", "James", "Awoniyi"],

    "41651": ["Areola", "Konaté", "Kerkez", "Spence",
              "Madueke", "Martinelli", "Barnes", "Johnson",
              "Cunha", "Evanilson", "João Pedro",
              "Verbruggen", "Damsgaard", "Tosin", "Zabarnyi"],

    "41739": ["A.Becker", "Guéhi", "Cash", "Mings", "Milenković",
              "Rice", "Thomas", "Garnacho", "J.Murphy", "Diogo J.",
              "Hirst", "Sels", "Doherty", "Haaland", "Wilson"],
    
    "42481": ["Flekken", "Bradley", "Muñoz", "Konsa", "Kiwior",
              "M.Asensio", "Rogers", "Bowen", "Eze",
              "Wissa", "Mateta", "Palmer", "Elanga", "Beto", "N.Williams"],

    "42848": ["Pickford", "Robertson", "C.Richards", "Robinson",
              "Smith Rowe", "Palmer", "Mbeumo", "Mitoma", "Souček",
              "Solanke", "Wood", "J.Virginia", "Darwin", "Trippier", "Branthwaite"],

    "45840": ["King", "Pond", "Pedro Lima", "Meupiyou",
              "Gonzalez", "Cundle", "B.Traore", "Miley", "Kuol",
              "Kalajdžić", "Chiwome", "Whiteman", "Lankshear", "Casey", "Simpson-Pusey"],

    "57356": ["Henderson", "Pedro Porro", "Schär", "Huijsen",
              "Saka", "Trossard", "I.Sarr", "O.Dango", "Rashford",
              "Gakpo", "Onuachu", "Sánchez", "Ndiaye", "Aina", "Davis"],

    "64060": ["Martínez", "Cucurella", "Kilman", "Chalobah",
              "Anderson", "Schade", "Gibbs-White", "B.Fernandes", "Maddison",
              "Nketiah", "Raúl", "José Sá", "Gvardiol", "Dalot", "Daka"],

    "64668": ["Leno", "Alexander-Arnold", "Emerson", "White",
              "Odobert", "Doku", "Kluivert", "M.Salah", "Nwaneri",
              "Füllkrug", "Delap", "Pope", "Colwill", "Burn", "Muniz"],

    "129846": ["Ederson M.", "Murillo", "Saliba", "Lewis", "Mitchell",
               "Iwobi", "Foden", "Hudson-Odoi", "Son", "Gordon",
               "Strand Larsen", "Ortega Moreno", "Archer", "Højlund", "Mykolenko"],

    "186268": ["Ramsdale", "Estupiñan", "Wan-Bissaka", "Lacroix",
               "Ødegaard", "Mac Allister", "Bruno G.", "Szoboszlai", "Merino",
               "Marmoush", "Vardy", "Onana", "Mazraoui", "Ings", "Mavropanos"]
}

LEAGUE_ID = "31131"  # e.g., "123456"
BASE_URL = "https://draft.premierleague.com/api"

# You need to grab your session cookie manually from your browser session with FPL
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 OPR/103.0.0.0",
    "Cookie": "pl_profile=eyJzIjogIld6SXNORFV3TkRNeE16VmQ6MXU2dURQOktXTUZTQzdXTG43Z2VCYUdGRFZLVk1zc3hXaDJhZTY4em1id1k1dUNxaVUiLCAidSI6IHsiaWQiOiA0NTA0MzEzNSwgImZuIjogIkpvaG4iLCAibG4iOiAiQnlyZCIsICJmYyI6IDF9fQ==;sessionid=.eJxVijsOwjAMQO-SGVX52PmwcQKGijlyakdBIIQaOiHuTrrB-D5vlWl7tbx1WfOV1VEBanDGoTr8pkLLTR57f96nXU-Xofs8n08D_t9GvY0R_cJUnLehiiBbD1YY2NTIBoyOkCBEwoDRWorBF4MJfaLK4grpoD5f07swzA:1u6uJw:SNl8NaB4mSpsdQiEA34i05eawma6kJRkJiNk9yXNu3s"
}

# Get league details to retrieve team entry IDs
league_url = f"{BASE_URL}/league/{LEAGUE_ID}/details"
response = requests.get(league_url, headers=HEADERS)
#print("Status code:", response.status_code)
#print("Raw text:", response.text[:300])
league_data = response.json()
#print(json.dumps(league_data, indent=2))

# Extract entries (teams in the league)
entries = league_data['league_entries']
entry_ids = [entry['entry_id'] for entry in entries]

manager_names = {
    str(entry['entry_id']): f"{entry['player_first_name']} {entry['player_last_name']}"
    for entry in league_data['league_entries']
}

print(f"Found {len(entry_ids)} teams.")


# Get full player dataset
bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
res = requests.get(bootstrap_url)
data = res.json()

# Create DataFrame with relevant player info
all_players_df = pd.DataFrame(data["elements"])

# Add player display name
all_players_df["web_name"] = all_players_df["web_name"].astype(str)

position_map = {
    1: "Goalkeeper",
    2: "Defender",
    3: "Midfielder",
    4: "Forward"
}

owned_players = set(player for roster in team_rosters.values() for player in roster)

# Get free agents by filtering out owned players
free_agents_df = all_players_df[~all_players_df["web_name"].isin(owned_players)]

free_agents_df = free_agents_df[free_agents_df["status"] == "a"]  # 'a' = available

free_agents_df["position"] = free_agents_df["element_type"].map(position_map)

top_by_position = {}

for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
    pos_df = free_agents_df[free_agents_df["position"] == pos]
    top_players = pos_df.sort_values(by="total_points", ascending=False).head(5)
    top_by_position[pos] = top_players[["web_name", "total_points", "form", "ict_index", "minutes"]]

    
'''# Sort by total points (or try 'form', 'ict_index', etc.)
top_free_agents = free_agents_df.sort_values(by="total_points", ascending=False)

# Show top 15
top_free_agents[["web_name", "total_points", "form", "ict_index", "minutes"]].head(15)'''


'''from IPython.display import display, Markdown

for pos, df in top_by_position.items():
    display(Markdown(f"### Top Free Agent {pos}s"))
    display(df.reset_index(drop=True))'''