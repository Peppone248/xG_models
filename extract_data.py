from datetime import time

from statsbombpy import sb
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
comps = sb.competitions()
print(f"Competizioni totali: {comps[['competition_id', 'competition_name']]}")

champions_league = comps[comps["competition_name"] == "Champions League"]
print(f"\nChampions League disponibile:")
print(champions_league[["competition_id", "season_id", "season_name"]].to_string(index=False))

COMPETITION_ID = 2
SEASON_ID = 27

# ============================================================
# LIVELLO 2: MATCHES
# ============================================================
# sb.matches() prende competition_id e season_id come input
# e restituisce un DataFrame con una riga per partita.
# Il campo chiave qui è match_id: ci servirà per scaricare gli eventi.

matches = sb.matches(COMPETITION_ID, SEASON_ID)
print(f"\nPartite nella stagione: {len(matches)}")

first = matches.iloc[0]
print(f"  match_id:  {first['match_id']}")
print(f"  data:      {first['match_date']}")
print(f"  casa:      {first['home_team']}")
print(f"  trasferta: {first['away_team']}")
print(f"  risultato: {first['home_score']}-{first['away_score']}")
print(f"  giornata:  {first['match_week']}")

# ============================================================
# LIVELLO 3: EVENTS (prima una sola partita)
# ============================================================

events = sb.events(match_id=first["match_id"])
print(f"\nEventi totali nella partita: {len(events)}")

shots = events[events["type"] == "Shot"]
print(f"Di cui tiri: {len(shots)}")

print(f"\n--- Anatomia di un tiro ---")
one_shot = shots.iloc[0]
print(f"  Giocatore:      {one_shot['player']}")
print(f"  Squadra:        {one_shot['team']}")
print(f"  Minuto:         {one_shot['minute']}'")
print(f"  Posizione:      {one_shot['location']}")
print(f"  Parte del corpo:{one_shot['shot_body_part']}")
print(f"  Tipo:           {one_shot['shot_type']}")
print(f"  Esito:          {one_shot['shot_outcome']}")
print(f"  xG StatsBomb:   {one_shot['shot_statsbomb_xg']:.4f}")

print(f"\n  → x = {one_shot['location'][0]} (distanza dalla propria porta)")
print(f"  → y = {one_shot['location'][1]} (posizione laterale, centro = 40)")

# ============================================================
# 4. ESTRARRE TUTTI I TIRI DELLA STAGIONE
# ============================================================
# PROBLEMA PRATICO: statsbombpy scarica i dati da GitHub.
# Con 380 richieste consecutive, alcune possono andare in timeout.
# Soluzione: retry con attesa crescente (exponential backoff).
# Questo è un pattern standard in data engineering per qualsiasi
# pipeline che dipende da un servizio esterno.

MAX_RETRIES = 5  # Quanti tentativi prima di arrendersi
BASE_WAIT = 3  # Secondi di attesa al primo retry

print(f"\n{'=' * 50}")
print(f"Estrazione tiri da {len(matches)} partite...")
print(f"{'=' * 50}")

all_shots = []
failed_matches = []

for i, (_, match) in enumerate(matches.iterrows()):

    # 1. Caricare eventi CON RETRY
    #    Se la richiesta fallisce, aspettiamo e riproviamo.
    #    L'attesa raddoppia ad ogni tentativo (3s, 6s, 12s, 24s, 48s)
    #    per dare tempo al server di riprendersi.
    events = None
    for attempt in range(MAX_RETRIES):
        try:
            events = sb.events(match_id=match["match_id"])
            break  # Successo, usciamo dal loop di retry
        except Exception as e:
            wait = BASE_WAIT * (2 ** attempt)
            print(f"\n  ⚠ Timeout partita {match['match_id']} "
                  f"(tentativo {attempt + 1}/{MAX_RETRIES}). "
                  f"Riprovo tra {wait}s...")
            time.sleep(wait)

    # Se tutti i tentativi falliscono, segniamo la partita e continuiamo
    if events is None:
        print(f"\n  ✗ Partita {match['match_id']} saltata dopo {MAX_RETRIES} tentativi")
        failed_matches.append(match["match_id"])
        continue

    # 2. Filtrare tiri
    match_shots = events[events["type"] == "Shot"].copy()

    # 3. Aggiungere contesto dalla tabella matches
    match_shots["match_date"] = match["match_date"]
    match_shots["home_team"] = match["home_team"]
    match_shots["away_team"] = match["away_team"]
    match_shots["match_week"] = match["match_week"]

    # 4. Accumulare
    all_shots.append(match_shots)

    # Stampa progresso ogni 50 partite
    if (i + 1) % 50 == 0 or (i + 1) == len(matches):
        n_shots = sum(len(s) for s in all_shots)
        print(f"  {i + 1}/{len(matches)} partite → {n_shots} tiri")

if failed_matches:
    print(f"\n⚠ {len(failed_matches)} partite non scaricate: {failed_matches}")
    print(f"  Puoi rieseguire lo script — le altre partite verranno riscaricate.")

# Unire tutto
df = pd.concat(all_shots, ignore_index=True)
print(f"\nEstrazione completata: {len(df)} tiri da {df['match_id'].nunique()} partite")

# ============================================================
# 5. SELEZIONARE LE COLONNE RILEVANTI
# ============================================================
# Il DataFrame contiene colonne per TUTTI i tipi di evento (passaggi, ecc.)
# La maggior parte sono NaN per i tiri. Teniamo solo quelle utili.
KEEP = [
    # Contesto partita
    "match_id", "match_date", "match_week", "home_team", "away_team",
    # Chi e quando
    "player", "player_id", "team", "position", "minute", "second", "period",
    # Dove (coordinate grezze)
    "location",
    # Caratteristiche del tiro
    "shot_body_part", "shot_type", "shot_technique",
    "shot_first_time", "shot_one_on_one", "under_pressure",
    "play_pattern",
    # Esito e benchmark
    "shot_outcome", "shot_statsbomb_xg",
    # Per v2: posizioni giocatori al momento del tiro
    "shot_freeze_frame",
]

# Manteniamo solo le colonne che esistono (alcune potrebbero non esserci)
keep_existing = [c for c in KEEP if c in df.columns]
df = df[keep_existing].copy()

# ============================================================
# 6. ESTRARRE X e Y dalla colonna location
# ============================================================
# location è una lista Python [x, y]. Per lavorarci in pandas,
# la spacchettiamo in due colonne numeriche separate.

df["shot_x"] = df["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
df["shot_y"] = df["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else None)

# ============================================================
# 7. SALVARE
# ============================================================
csv_cols = [c for c in df.columns if c not in ["location", "shot_freeze_frame"]]
df[csv_cols].to_csv("serie_a_shots.csv", index=False)

df.to_pickle("serie_a_shots.pkl")

print(f"\nFile salvati:")
print(f"  serie_a_shots.csv  — {len(df)} righe (senza freeze frame)")
print(f"  serie_a_shots.pkl  — {len(df)} righe (completo)")

print(f"\n{'=' * 50}")
print(f"RIEPILOGO DATASET")
print(f"{'=' * 50}")
print(f"Tiri totali:     {len(df)}")
print(f"Partite:         {df['match_id'].nunique()}")
print(f"Giocatori:       {df['player'].nunique()}")
print(f"Squadre:         {df['team'].nunique()}")
goal_rate = (df["shot_outcome"] == "Goal").mean()
print(f"Gol:             {(df['shot_outcome'] == 'Goal').sum()}")
print(f"% conversione:   {goal_rate:.1%}")

print(f"\nDistribuzione esiti:")
print(df["shot_outcome"].value_counts().to_string())