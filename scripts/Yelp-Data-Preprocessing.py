
"""
Data Pre-Processing - Yelp Reviews Dataset

Author:       Lillian Wool
Email:        lwool@sandiego.edu
Class:        MSBA502

Description:
    This Python script performs data pre-processing on the Yelp Academic Dataset 
    to create a manageable dataset suitable for downstream tasks such as 
    sentiment analysis and emotion detection. Key tasks include:
        - Random sampling of reviews to create a smaller dataset (~15,000 reviews)
        - Selection of relevant columns from review, business, and user data
        - Memory-efficient streaming of large JSON files
        - Merging reviews with corresponding business and user information
        - Outputting a cleaned CSV dataset ready for analysis

Original Dataset:
    - Source: https://business.yelp.com/data/resources/open-dataset/
    - Reviews: 8,662,090 observations
    - Businesses: 229,907 observations
    - Users: 2,904,385 observations
    - Size: ~6.8 GB compressed JSON files
    - Files used in this script:
        * yelp_academic_dataset_review.json
        * yelp_academic_dataset_business.json
        * yelp_academic_dataset_user.json
    - Files NOT used in this script:
        * yelp_academic_dataset_tip.json
        * yelp_academic_dataset_checkin.json
        * photos and other auxiliary JSON files

Required Libraries:
    - pandas
    - numpy
    - tqdm
    - orjson

Usage:
    1. Activate the Python virtual environment.
    2. Ensure all required libraries are installed.
    3. Run the script in the terminal or IDE:
           python Yelp-Data-Preprocessing.py
    4. The processed dataset will be saved as 'sampled_yelp_reviews.csv'.

Notes:
    - Designed to avoid memory errors when handling large JSON files.
    - The script preserves only the necessary columns for analysis.
    - Sentiment analysis and emotion detection are intended to be applied 
      on the output dataset in subsequent steps.
"""




# Import necessary libraries
import pandas as pd
import numpy as np
import orjson
from tqdm import tqdm

# Configure file paths and sample parameters
review_file = "yelp_academic_dataset_review.json"
business_file = "yelp_academic_dataset_business.json"
user_file = "yelp_academic_dataset_user.json"

SAMPLE_SIZE = 15000
SEED = 42
np.random.seed(SEED)

# Define columns to keep from each JSON file
review_cols = ["review_id", "user_id", "business_id", "stars", "date", "text"]
business_cols = ["business_id", "name", "city", "state", "stars", "review_count", "is_open", "categories"]
user_cols = ["user_id", "name", "review_count", "average_stars"]

# Count total reviews in the dataset
print("Counting reviews...")
with open(review_file, "rb") as f:
    total_reviews = sum(1 for _ in f)
print(f"Total reviews in dataset: {total_reviews:,}")

# Select random line numbers to sample
chosen_idx = set(np.random.choice(total_reviews, SAMPLE_SIZE, replace=False))

# Stream reviews and collect only sampled ones
sampled_reviews = []

with open(review_file, "rb") as f:
    for i, line in tqdm(enumerate(f), total=total_reviews, desc="Sampling reviews"):
        if i in chosen_idx:
            review_json = orjson.loads(line)
            sampled_reviews.append({k: review_json[k] for k in review_cols})

# Convert sampled reviews to DataFrame
sampled_reviews_df = pd.DataFrame(sampled_reviews)
print(f"Sampled {len(sampled_reviews_df)} reviews successfully!")

# Load business and user data
business_df = pd.read_json(business_file, lines=True)[business_cols]
user_df = pd.read_json(user_file, lines=True)[user_cols]

# Merge reviews with corresponding business and user information
merged_df = sampled_reviews_df.merge(business_df, on="business_id", how="left") \
                              .merge(user_df, on="user_id", how="left")
print("Final merged dataset shape:", merged_df.shape)

# Save processed dataset to CSV
merged_df.to_csv("sampled_yelp_reviews.csv", index=False)
print("Saved sampled dataset to 'sampled_yelp_reviews.csv'")
