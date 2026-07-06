from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 1. Load your original 5,000 song dataset and the artist lookup
# (Make sure to replace "spotify_5k_dataset.csv" with your actual filename!)


high = pd.read_csv(BASE_DIR / "high_popularity_spotify_data.csv")
low = pd.read_csv(BASE_DIR / "low_popularity_spotify_data.csv")

audio_df = pd.concat([high, low], ignore_index=True)
artist_lookup = pd.read_csv(BASE_DIR / "artist_lookup.csv")

# 2. DEDUPLICATE: Remove duplicate tracks to prevent data leakage (Our Phase 1 fix)
audio_df = audio_df.drop_duplicates(subset=["track_id"]).copy()

# 3. Extract the Primary Artist (splitting by comma) and clean the text
audio_df["primary_artist"] = (
    audio_df["track_artist"]
    .astype(str)
    .apply(lambda x: x.split(',')[0])  # Keeps only the first artist
    .str.lower()
    .str.strip()
)

artist_lookup["artist_name_clean"] = (
    artist_lookup["artist_name"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# 4. Perform the Left Join using the cleaned primary artist
df_merged = audio_df.merge(
    artist_lookup,
    how="left",
    left_on="primary_artist",
    right_on="artist_name_clean"
)

# 5. Create the metadata flag (This was a great idea in your original code, keep it!)
df_merged["has_artist_metadata"] = (
    df_merged["artist_popularity"].notna().astype(int)
)

# NOTE: We purposely REMOVED the fillna(0) steps here!
# We want XGBoost to see the missing values as actual NaNs.

# 6. Drop rows where core track info is missing entirely
df_merged = df_merged.dropna(subset=[
    "track_name",
    "track_artist"
])

# 7. Save the enriched dataset to a new file so you don't overwrite anything
df_merged.to_csv(
    BASE_DIR / "spotify_enriched_5k.csv",
    index=False
)

print("Merged 5k dataset saved successfully!")

# 8. Print out the stats to see our true match rate
df = pd.read_csv(BASE_DIR / "spotify_enriched_5k.csv")

print("\n--- Match Statistics ---")
print(df["has_artist_metadata"].value_counts())
match_rate = df["has_artist_metadata"].mean() * 100
print(f"Overall Match Rate: {match_rate:.2f}%")

print("\n--- Artist Popularity Stats (NaNs excluded) ---")
print(df["artist_popularity"].describe())

print("\n--- Artist Followers Stats (NaNs excluded) ---")
print(df["artist_followers"].describe())