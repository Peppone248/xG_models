'''Backheel           30  0.100000
Diving Header      34  0.205882
Half Volley      1458  0.085734
Lob                72  0.138889
Normal           7523  0.084275
Overhead Kick      55  0.072727
Volley            705  0.106383
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("shots_featured.csv")
print(f"Caricato: {len(df)} tiri, {len(df.columns)} colonne")

FEATURES = [
    #"distance",
    "log_distance",
    #"angle",
    "log_angle",
    "is_one_on_one",
    "is_first_shot",
    "is_under_pressure",
    "body_Head",
    #"body_Right Foot",
    "body_Left Foot",
    "body_Other",
    #"type_Open Play",
    "type_Free Kick",
    "technique_Normal",
    #"technique_Half Volley",
    "technique_Volley",
    #"technique_Backheel",
    "centrality"
]


# Split temporale (già discusso)
train = df[df["match_week"] <= 31].copy()
test = df[df["match_week"] > 31].copy()

X_train = train[FEATURES]
y_train = train["is_goal"]
X_test = test[FEATURES]
y_test = test["is_goal"]

# Fit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predizioni: predict_proba restituisce [P(no goal), P(goal)]
# La colonna [:, 1] è la probabilità di gol — il nostro xG
y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]

# Valutazione
print(f"Log loss train: {log_loss(y_train, y_pred_train):.4f}")
print(f"Log loss test:  {log_loss(y_test, y_pred_test):.4f}")

# Baseline: prevedere la media per ogni tiro
naive_pred = np.full(len(y_test), y_train.mean())
print(f"Log loss naive:    {log_loss(y_test, naive_pred):.4f}")
print(f"Log loss modello:  {log_loss(y_test, y_pred_test):.4f}")

print(f"Log loss StatsBomb: {log_loss(y_test, test['shot_statsbomb_xg']):.4f}")

# --- Calibration plot ---
# Usiamo 10 bin basati sui quantili invece che su intervalli uguali.
# Perché? Con intervalli uguali, i bin alti hanno pochissimi tiri.
# I quantili garantiscono circa lo stesso numero di tiri per bin.

test["xg_pred"] = y_pred_test
test["xg_bin"] = pd.qcut(test["xg_pred"], q=10, duplicates="drop")

calibration = test.groupby("xg_bin", observed=True).agg(
    predicted_xg=("xg_pred", "mean"),    # media delle predizioni nel bin
    actual_goal_rate=("is_goal", "mean"), # % gol reali nel bin
    n_shots=("is_goal", "count")          # quanti tiri nel bin
)

fig, ax = plt.subplots(figsize=(10, 8))

# Linea diagonale = calibrazione perfetta
ax.plot([0, 1], [0, 1], "k--", label="Calibrazione perfetta")

# I nostri punti
ax.scatter(
    calibration["predicted_xg"],
    calibration["actual_goal_rate"],
    s=calibration["n_shots"] / 2,  # dimensione proporzionale ai tiri
    zorder=5,
    label="Modello"
)

ax.set_xlabel("xG predetto (media del bin)")
ax.set_ylabel("% gol effettivi")
ax.set_title("Calibration Plot — Logistic Regression v1")
ax.legend()
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.6)
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=150)
plt.show()

calibration.columns = ["predicted_xg", "actual_goal_rate", "n_shots"]
print(calibration)

# --- Coefficienti ---
# Ogni coefficiente dice: "se questa feature aumenta di 1 unità,
# di quanto cambia il log-odds di segnare?"

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)

print("Intercetta:", model.intercept_[0])
print()
print(coef_df.to_string(index=False))

calibration.to_csv('calibration_centrality.txt', sep='\t', index=False)
coef_df.to_csv('coef_centrality.txt', sep='\t', index=False)