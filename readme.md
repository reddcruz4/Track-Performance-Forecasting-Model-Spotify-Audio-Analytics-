# Track Performance Forecasting Model (Spotify Audio Analytics)

## 📌 Overview

### Problem Statement
In the highly competitive music and digital content landscape, predicting how consumers will engage with a track before it scales is notoriously difficult. A&R teams, marketers, and algorithmic curators often rely on subjective listening or trailing indicators (like post-release streams), which are not scalable or proactive. Additionally, machine learning models that attempt to solve this are often treated as "black boxes," making it difficult for non-technical stakeholders to trust or act upon the predictions.

### Intended Solution
This project delivers an end-to-end data science solution that mathematically maps 15 quantitative audio features to consumer engagement metrics. By combining an automated data processing pipeline with ensemble machine learning models (XGBoost/Random Forest), this system forecasts track performance with 72% accuracy ($R^2 = 0.72$). To bridge the gap between technical output and business utility, the project also integrates SHAP (Explainable AI) for transparent insights and a zero-latency **Audio Lookup Table** for instant querying by business teams.

---

## 🛠️ Key Features & Architecture (Challenges & Solutions)

### 1. Automated Data Processing Pipeline
To eliminate manual data wrangling and ensure reproducibility across 10,000+ tracks, I built a modular Python pipeline.
* **The Challenge:** The raw audio data contained highly skewed distributions (e.g., features like `Acousticness` and `Instrumentalness` heavily clustered at 0 or 1) and varying scales (e.g., `Tempo` in BPM vs. `Valence` strictly bound between 0 and 1).
* **The Solution:** Engineered a robust transformation sequence. I implemented automated feature scaling (using Standard/MinMax Scalers) to normalize variances and applied targeted encoding for categorical metadata, ensuring the model received mathematically stable inputs.

### 2. High-Performance Audio Lookup Table
* **The Challenge:** Running live machine learning inference scripts is computationally expensive and inaccessible for non-technical teams who need quick answers during curation meetings.
* **The Solution:** Engineered a pre-computed **Lookup Table** architecture. Instead of processing complex ML queries on demand, the pipeline automatically structures historical and forecasted benchmarks into an indexed dataframe. Users can instantly retrieve a track's predicted engagement tier, risk factors, and audio benchmarks with zero computational latency.

### 3. Predictive Modeling & Hyperparameter Tuning
* **The Challenge:** Early iterations of the model suffered from overfitting due to class imbalances and dominant genres heavily swaying the decision trees.
* **The Solution:** Implemented robust cross-validation techniques (Stratified K-Fold) and utilized GridSearchCV to fine-tune the hyperparameters of the XGBoost and Random Forest algorithms. This stabilized the model variance and maximized generalization, securing the final 72% predictive accuracy.

### 4. Explainable AI (SHAP Integration)
* **The Challenge:** Stakeholders need to know *why* a track is predicted to succeed, not just *that* it will succeed. 
* **The Solution:** Integrated SHAP (SHapley Additive exPlanations) to deconstruct the model's predictions. This framework isolates exactly how much individual metrics (e.g., high valence vs. low acousticness) push a track's projected score up or down, effectively revealing the primary drivers of consumer traction.

---

## 📈 Data Visualizations & Dashboards
*(Note: Visualizing the data distribution and model explainability was a core component of this analysis. Below are the key visual insights generated from the pipeline.)*

### 1. Feature Importance (SHAP Summary)
*This chart visualizes which audio features had the strongest impact on predicting high consumer engagement.*

<img width="792" height="940" alt="shap" src="https://github.com/user-attachments/assets/3e08eef7-d467-456f-87b1-983f38f2af93" />


### 2. Correlation Matrix of Audio Features
*A heatmap identifying the multi-collinearity between structural audio features (e.g., the negative correlation between Energy and Acousticness).*

<img width="725" height="627" alt="correlation" src="https://github.com/user-attachments/assets/5d343b3d-faa5-48f3-8279-bfd8384c79f5" />


### 3. Predicted vs. Actual Engagement Distributions
*A visualization tracking the model's forecasting accuracy against the testing dataset.*

<img width="1383" height="583" alt="accuracy" src="https://github.com/user-attachments/assets/9747b3d8-7a98-4fb1-be42-6bb0c4290d96" />


---

## 📊 Dataset & Audio Features
The dataset consists of **10,000+ tracks** embedded with **15 core audio dimensions**. Key variables include:
* `Danceability` / `Energy` / `Loudness`: Structural audio intensity vectors.
* `Valence`: The emotional positivity expressed by a track.
* `Acousticness` / `Instrumentalness`: Textural composition metrics.
* `Tempo`: Beats per minute (BPM) tracking rhythm flow.

---

## 🚧 Limitations & Future Work
While the current model effectively captures acoustic-driven engagement, consumer behavior is influenced by external contextual factors. Future iterations of this project will focus on:
1. **Lyrical Sentiment Analysis:** Pulling textual data via external lyrical APIs (e.g., Genius or Musixmatch) to apply Natural Language Processing (NLP) alongside the audio features.
2. **Virality & Social Context:** Integrating external "virality" metrics (e.g., TikTok trend velocity, social media shares) to capture algorithmic spikes that audio features alone cannot predict.
3. **Time-Series Decay:** Tracking how quickly a track's engagement drops off over a 6-month period to forecast longevity rather than just peak success.

---

## 📂 Repository Structure
```text
├── data/
│   ├── raw_tracks.csv          # Source dataset (10,000+ records)
│   └── audio_lookup_table.csv  # Generated business lookup array
├── notebooks/
│   ├── 1_data_cleaning.ipynb   # Exploration & manipulation
│   ├── 2_modeling_shap.ipynb   # Model training & explanation
├── src/
│   ├── pipeline.py             # Automated data preprocessing script
│   └── model_inference.py      # Forecasting execution module
├── outputs/
│   ├── shap_summary.png        # SHAP visual dashboard
│   └── correlation_matrix.png  # Feature EDA visuals
├── README.md
└── requirements.txt
