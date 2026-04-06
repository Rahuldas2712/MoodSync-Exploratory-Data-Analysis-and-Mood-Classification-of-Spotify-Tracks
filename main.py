import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("SpotifyFeatures.csv", encoding='latin1')

# Strip any leading/trailing spaces in column names
df.columns = df.columns.str.strip()
df.rename(columns={'ï»¿genre': 'genre'}, inplace=True)
# Check column names
print("Columns in Dataset:", df.columns.tolist())

# -----------------------------
# 2. Select Relevant Columns
# -----------------------------
df = df[['track_name', 'artist_name', 'genre', 'danceability', 'energy',
         'valence', 'acousticness', 'popularity']]

# Drop rows with missing values
df.dropna(inplace=True)

# -----------------------------
# 3. Define Mood Classification
# -----------------------------
def classify_mood(row):
    if row['valence'] > 0.6 and row['energy'] > 0.5:
        return "Happy"
    elif row['valence'] < 0.4 and row['energy'] < 0.5:
        return "Sad"
    elif row['energy'] > 0.7 and row['danceability'] > 0.6:
        return "Energetic"
    elif row['acousticness'] > 0.6 and row['energy'] < 0.5:
        return "Calm"
    else:
        return "Neutral"

df['Mood'] = df.apply(classify_mood, axis=1)

print("\nMood Distribution:")
print(df['Mood'].value_counts())

# -----------------------------
# 4. Mood Distribution Plot
# -----------------------------
plt.figure(figsize=(8,5))
df['Mood'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Songs by Mood")
plt.xlabel("Mood")
plt.ylabel("Number of Songs")
plt.show()

# -----------------------------
# 5. Average Popularity by Mood
# -----------------------------
mood_popularity = df.groupby("Mood")['popularity'].mean().sort_values()
mood_popularity.plot(kind='barh', color='orange', figsize=(7,4))
plt.title("Average Popularity by Mood")
plt.xlabel("Average Popularity")
plt.ylabel("Mood")
plt.show()

# -----------------------------
# 6. Danceability vs Energy Scatter Plot
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='danceability', y='energy', hue='Mood', alpha=0.6)
plt.title("Danceability vs Energy (Mood-wise Distribution)")
plt.show()

# -----------------------------
# 7. Top 10 Artists by Popularity
# -----------------------------
top_artists = df.groupby('artist_name')['popularity'].mean().sort_values(ascending=False).head(10)
top_artists.plot(kind='bar', color='green', figsize=(8,4))
plt.title("Top 10 Artists by Average Popularity")
plt.xlabel("Artist")
plt.ylabel("Average Popularity")
plt.xticks(rotation=45)
plt.show()
