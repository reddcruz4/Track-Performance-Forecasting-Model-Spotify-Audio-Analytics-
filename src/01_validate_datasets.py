from pathlib import Path
import pandas as pd
import sqlite3

BASE_DIR = Path(__file__).parent

first = pd.read_csv(BASE_DIR / "spotify_data_clean.csv")
second = pd.read_csv(BASE_DIR / "track_data_final.csv")

audio_df = pd.read_csv(BASE_DIR / "spotify_songs.csv")

artist_df = pd.concat([first, second], ignore_index=True)

#####
print(artist_df["artist_name"].sample(20, random_state=42).tolist())