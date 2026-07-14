import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

GOAL_X = 120
POST_A_Y = 36
POST_B_Y = 44
columns_to_drop = {
"player_id",
    "body_Right Foot",
    "technique_Overhead Kick",
    "technique_Backheel",
    "second",
    "period",
    "match_id",
    "minute",
    "technique_Diving Header",
    "type_Open Play",
    "technique_Lob"
}

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\Users\giuse\PycharmProjects\xG_models\serie_a_shots.csv')

print(df.describe())

print(df["shot_first_time"].value_counts(dropna=False))

# handle Penalty shot type
df = df[df["shot_type"] != "Penalty"].copy()

# binary target: 1 gol, 0 otherwise
df["is_goal"] = (df["shot_outcome"] == "Goal").astype(int)
df["is_one_on_one"] = df["shot_one_on_one"].notna().astype(int)
df["is_first_shot"] = df["shot_first_time"].notna().astype(int)
df["is_under_pressure"] = df["under_pressure"].notna().astype(int)
# Creating: body_Head, body_Right Foot, body_Left Foot, body_Other
dummies = pd.get_dummies(df["shot_body_part"], prefix="body")
shot_types = pd.get_dummies(df["shot_type"], prefix="type")
techniques = pd.get_dummies(df["shot_technique"], prefix="technique")
df = pd.concat([df, dummies, shot_types, techniques], axis=1)

#df.drop(["type_Penalty"], inplace=True)

# vectors to compute wide angel between shooter and goal
dx = GOAL_X - df["shot_x"]
dy_a = POST_A_Y - df["shot_y"]
dy_b = POST_B_Y - df["shot_y"]

angle_a = np.arctan2(dy_a, dx)
angle_b = np.arctan2(dy_b, dx)

print(f"Tiri totali: {len(df)}")
print(f"Goal rate complessivo: {df['is_goal'].mean():.3f}")

# is the data distributed evenly, or are some categories rare?
df["shot_body_part"].value_counts().plot(kind='bar')

# --- Categoriche ---
for col in ["shot_body_part", "shot_type", "shot_one_on_one",
            "shot_first_time", "under_pressure"]:
    print(f"\n{'='*40}")
    print(f"{col}")
    print(f"{'='*40}")
    print(df.groupby(col)["is_goal"].agg(["count", "mean"]))

# --- Numeriche ---
for col in ["shot_x", "shot_y"]:
    print(f"\n{'='*40}")
    print(f"{col}")
    print(f"{'='*40}")
    print(df[col].describe())

df["distance"] = np.sqrt((120 - df["shot_x"])**2 + (40 - df["shot_y"])**2)
# angles between shooters and spot
df["angle"] = np.abs(np.degrees(angle_b - angle_a))
df["log_distance"] = np.log(df["distance"])
df["log_angle"] = np.log(df["angle"] + 1)

# interactions features - combined features to match football situations
df["angle_x_header"] = df["angle"] * df["body_Head"]
df["distance_x_pressure"] = df["distance"] * df["is_under_pressure"]

# measure to frame how centrale the shot is --- LOWER CENTRALITY = More central shot = better chance!
df["centrality"] = abs(df["shot_y"] - 40)

df.drop(columns=columns_to_drop, inplace=True)
plt.figure(figsize=(12, 10))
corr_matrix = df.corr(numeric_only=True)
sn.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.show()

print(df.groupby(pd.cut(df["angle"], bins=[0, 5, 10, 20, 30, 90]))["is_goal"].agg(["count", "mean"]))
print(df.groupby(pd.cut(df["distance"], bins=[0, 8, 12, 18, 25, 45]))["is_goal"].agg(["count", "mean"]))
print(df.groupby("shot_technique")["is_goal"].agg(["count", "mean"]))

train = df[df["match_week"] <= 31].copy()
test = df[df["match_week"] > 31].copy()

print(f"Train: {len(train)} tiri (matchweek 1-31)")
print(f"Test:  {len(test)} tiri (matchweek 32-38)")
print(f"Goal rate train: {train['is_goal'].mean():.3f}")
print(f"Goal rate test:  {test['is_goal'].mean():.3f}")

df.to_csv("shots_featured.csv", index=False)
print(f"Salvato: shots_featured.csv ({len(df)} righe, {len(df.columns)} colonne)")