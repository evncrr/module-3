import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

batting = pd.read_csv("mlb/Batting.csv")
master = pd.read_csv("mlb/Master.csv")

master['player_Name'] = master['nameFirst'] + " " + master['nameLast']
df = batting.merge(master[['playerID', 'player_Name']], on='playerID', how='left')
stats = df.groupby('playerID')[["AB", "H", "2B", "3B", "HR", "BB", "SO", "SB", "RBI"]].sum().reset_index()
stats = stats.merge(master[['playerID', 'player_Name']], on='playerID', how='left')

stats = stats[stats['AB'] > 1000]

stats['H_rate'] = stats['H'] / stats['AB']
stats['2B_rate'] = stats['2B'] / stats['AB']
stats['HR_rate'] = stats['HR'] / stats['AB']
stats['BB_rate'] = stats['BB'] / stats['AB']
stats['SO_rate'] = stats['SO'] / stats['AB']
stats['SB_rate'] = stats['SB'] / stats['AB']

features = ['H_rate', '2B_rate', 'HR_rate', 'BB_rate', 'SO_rate','SB_rate']
X = stats[features].fillna(0).values  #dense matrix

#standardize features for euclidean distance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#calc pairwise euclidean distance
dist_matrix = euclidean_distances(X_scaled)

#convert distance to similarity
similarity_matrix = 1 / (1 + dist_matrix)

#label similarity matrix with player names
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=stats['player_Name'],
    columns=stats['player_Name']
)

#get top most similar players to target players
target_player = "Mike Trout"
target_player2 = "Chris Davis"
target_player3 = "Andrew McCutchen"
similar_players = similarity_df[target_player].sort_values(ascending=False).head(10)
similar_players2 = similarity_df[target_player2].sort_values(ascending=False).head(10)
similar_players3 = similarity_df[target_player3].sort_values(ascending=False).head(10)


print(f"Players most similar to {target_player}:")
print(similar_players)
print(f"Players most similar to {target_player2}:")
print(similar_players2)
print(f"Players most similar to {target_player3}:")
print(similar_players3)

