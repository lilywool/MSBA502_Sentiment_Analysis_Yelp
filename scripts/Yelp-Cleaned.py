"""
Data Cleaning - Yelp Reviews Dataset

Author:       Lillian Wool
Email:        lwool@sandiego.edu
Class:        MSBA502

Description:
    This Python script performs data cleaning and column standardization 
    on the Yelp reviews dataset with sentiment and emotion analysis. Key tasks include:
        - Calculating descriptive statistics for meaningful columns
        - Printing head() and basic stats
        - Renaming columns to snake_case
        - Calculating stats and .isna() counts after renaming
        - Saving the full cleaned dataset (all reviews retained) to CSV

Input:
    - Yelp_reviews_sentiment_emotion.csv

Output:
    - yelp.cleaned.csv
"""

# Import necessary libraries
import pandas as pd
import numpy as np

# Configure file paths
INPUT_CSV = "Yelp_reviews_sentiment_emotion.csv"
OUTPUT_CSV = "yelp.cleaned.csv"

# Load CSV
print("Loading reviews CSV with sentiment and emotion columns...")
df = pd.read_csv(INPUT_CSV)

# Rename columns to snake_case
df.rename(columns={
    "review_id": "review_id",
    "user_id": "user_id",
    "business_id": "business_id",
    "stars_x": "business_star_rating",
    "date": "review_date",
    "text": "review_text",
    "name_x": "business_name",
    "city": "business_city",
    "state": "business_state",
    "stars_y": "user_star_rating",
    "review_count_x": "business_review_count",
    "is_open": "business_is_open",
    "categories": "business_categories",
    "name_y": "user_name",
    "review_count_y": "user_review_count",
    "average_stars": "user_average_stars",
    "Sentiment_Label": "sentiment_label",
    "Sentiment_Confidence": "sentiment_confidence",
    "Emotion_Label": "emotion_label",
    "Emotion_Confidence": "emotion_confidence"
}, inplace=True)

# Print table header
print("\n--- Column Names After Renaming ---")
print(df.columns.tolist())

# Numeric descriptive stats
numeric_cols = [
    "business_star_rating", "user_star_rating", 
    "business_review_count", "user_review_count", 
    "user_average_stars", "sentiment_confidence", "emotion_confidence"
]

numeric_stats = df[numeric_cols].describe().transpose()
numeric_stats["mode"] = df[numeric_cols].mode().iloc[0]
numeric_stats["na_count"] = df[numeric_cols].isna().sum()

# Categorical stats (exclude IDs, names, review text, date)
categorical_cols = [
    "business_is_open", "business_categories", 
    "business_city", "business_state", 
    "sentiment_label", "emotion_label"
]

categorical_stats = pd.DataFrame(index=categorical_cols)
for col in categorical_cols:
    categorical_stats.loc[col, "count"] = df[col].count()
    categorical_stats.loc[col, "unique"] = df[col].nunique()
    categorical_stats.loc[col, "mode"] = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
    categorical_stats.loc[col, "na_count"] = df[col].isna().sum()

# Print descriptive stats
print("\n--- Numeric Column Stats ---")
print(numeric_stats)

print("\n--- Categorical Column Stats ---")
print(categorical_stats)

# Save cleaned CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n Full cleaned dataset saved to '{OUTPUT_CSV}'")
