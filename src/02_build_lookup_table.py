from pathlib import Path
import pandas as pd
import sqlite3

BASE_DIR = Path(__file__).parent

first = pd.read_csv(BASE_DIR / "spotify_data_clean.csv")
second = pd.read_csv(BASE_DIR / "track_data_final.csv")

audio_df = pd.read_csv(BASE_DIR / "spotify_songs.csv")
artist_df = pd.concat([first, second], ignore_index=True)


##clean artist names
artist_df["artist_name"] = (
    artist_df["artist_name"]
    .str.lower()
    .str.strip()
)
### build lookup table
artist_lookup = (
    artist_df
    .groupby("artist_name")
    .agg({
        "artist_popularity": "mean",
        "artist_followers": "max"
    })
    .reset_index()
)

from pathlib import Path

data_dir = BASE_DIR / "data"
data_dir.mkdir(exist_ok=True)

artist_lookup.to_csv(
    data_dir / "artist_lookup.csv",
    index=False
)

print("Artist lookup saved successfully!")