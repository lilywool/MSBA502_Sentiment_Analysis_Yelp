
# Yelp Academic Dataset Analysis

Python scripts for preprocessing, sentiment analysis, emotion detection, and cleaning of the Yelp Academic Dataset. Includes tools for sampling reviews, merging business and user information, analyzing sentiment/emotion with HuggingFace transformers, and producing a cleaned dataset ready for downstream analysis.

## Repository Structure

```

Yelp-Project/
│
│
├─ output/                              # CSV outputs included
│   ├─ sampled_yelp_reviews.csv         # 15,000 review sample (Seed = 42)
│   ├─ Yelp_reviews_sentiment_emotion.csv
│   └─ yelp.cleaned.csv
│
├─ Yelp-Data-Preprocessing.py           # Preprocessing script
├─ Yelp-Emotional-Sentiment.py          # Sentiment & emotion analysis
├─ Yelp-Data-Cleaning.py                # Cleaning & stats script
└─ README.md                            # This file

````

## Setup

1. Clone the repository:

```bash
git clone https://github.com/username/Yelp-Project.git
cd Yelp-Project
````

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

3. Install required Python libraries:

```bash
pip install pandas numpy tqdm transformers torch orjson
```

4. (Optional) If you want to recreate the sampled CSV yourself, download the **Yelp Academic Dataset** from:

[Yelp Open Dataset](https://www.yelp.com/dataset)

Place the files `yelp_academic_dataset_review.json`, `yelp_academic_dataset_business.json`, and `yelp_academic_dataset_user.json` in a folder named `Yelp-JSON` inside the project root.

## Usage

### 1. Preprocessing

```bash
python Yelp-Data-Preprocessing.py
```

* Randomly samples **15,000 reviews** using **Seed = 42**.
* Merges review data with business and user information.
* Saves output as `output/sampled_yelp_reviews.csv`.

### 2. Sentiment & Emotion Analysis

```bash
python Yelp-Emotional-Sentiment.py
```

* Performs sentiment analysis (positive/negative) and emotion detection on reviews.
* Saves enriched dataset as `output/Yelp_reviews_sentiment_emotion.csv`.

### 3. Data Cleaning

```bash
python Yelp-Data-Cleaning.py
```

* Standardizes column names.
* Computes descriptive statistics for numeric and categorical columns.
* Saves cleaned dataset as `output/yelp.cleaned.csv`.

## Notes

* The included CSVs (`sampled_yelp_reviews.csv`, `Yelp_reviews_sentiment_emotion.csv`, `yelp.cleaned.csv`) are **derived from the Yelp Academic Dataset** and contain a **subset of 15,000 reviews** for academic purposes.
* The original JSON files are **not included** due to size and licensing.
* All scripts are designed for **academic use only** and comply with Yelp’s Data Agreement.
* HuggingFace transformer models may truncate reviews exceeding 512 tokens.

## License

This repository’s Python scripts are licensed under the Apache License 2.0.  
The Yelp dataset is **not included** and is governed by Yelp’s [Data Agreement](https://www.yelp.com/dataset).

---

# Apache License 2.0

Copyright 2025 Lillian Wool

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.
```

