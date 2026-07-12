import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

GOAL_X = 120
POST_A_Y = 36
POST_B_Y = 44
pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\Users\giuse\PycharmProjects\xG_models\serie_a_shots.csv')

print(df.describe())

print(df["shot_first_time"].value_counts(dropna=False))

# binary target: 1 gol, 0 otherwise
df["is_goal"] = (df["shot_outcome"] == "Goal").astype(int)
df["is_one_on_one"] = df["shot_one_on_one"].notna().astype(int)
df["is_first_shot"] = df["shot_first_time"].notna().astype(int)
df["is_under_pressure"] = df["under_pressure"].notna().astype(int)
# Creating: body_Head, body_Right Foot, body_Left Foot, body_Other
dummies = pd.get_dummies(df["shot_body_part"], prefix="body")
shot_types = pd.get_dummies(df["shot_type"], prefix="type")

# vectors to compute wide angel between shooter and goal
dx = GOAL_X - df["shot_x"]
dy_a = POST_A_Y - df["shot_y"]
dy_b = POST_B_Y - df["shot_y"]

angle_a = np.arctan2(dy_a, dx)
angle_b = np.arctan2(dy_b, dx)

# angles between shooters and spot
df["angle"] = np.abs(np.degrees(angle_b - angle_a))

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

corr_matrix = df.corr(numeric_only=True)
sn.heatmap(corr_matrix, annot=True)
plt.show()

print(df.groupby(pd.cut(df["angle"], bins=[0, 5, 10, 20, 30, 90]))["is_goal"].agg(["count", "mean"]))
print(df.groupby(pd.cut(df["distance"], bins=[0, 8, 12, 18, 25, 45]))["is_goal"].agg(["count", "mean"]))