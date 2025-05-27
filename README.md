# Earthquake_ML_RF_Model
This file deals with the development of Earthquake_Detection and Predection model. Below is the summary of the achieved goals

# Key Insights and Analysis of the Earthquake Prediction Model
## Exceptional Model Performance
The model achieved perfect classification metrics, including 100% precision, recall, and F1-score, along with an ROC AUC and PR AUC of 1.0. The confusion matrix confirmed no misclassifications, indicating flawless separation between earthquake and non-earthquake days. While such results are rare in real-world scenarios, they suggest either:

1. A perfectly separable dataset (possibly synthetic or preprocessed in a way that leaks target information).

2. Overfitting, where the model memorizes training patterns but may fail on unseen data.

3. Feature engineering artifacts, such as lag features that may indirectly reveal future earthquake occurrences.

4. Further validation on independent datasets is necessary to confirm generalization.

## Robust Data Preprocessing & Feature Engineering
The pipeline effectively handles:

1. Temporal feature extraction (hour, day of week, month) to capture periodic trends.

2. Daily aggregation of seismic metrics (mean magnitude, depth, coordinates).

3. Lag features (1, 7, and 14 days) to model short and mid-term dependencies.

4. Automated preprocessing (missing value imputation, scaling, categorical encoding).

However, the lack of spatial features (e.g., regional clustering, distance-based metrics) limits location-specific predictions. Future iterations could incorporate geospatial analysis for more granular forecasting.

## Visualization & Interpretability
The system generates insightful plots:

1. Daily earthquake trends to observe temporal patterns.

2. PCA projections to assess feature separability.

3. Predicted vs. actual probabilities to validate model alignment.

4. Feature importance rankings, revealing which factors (e.g., lagged occurrences, depth, magnitude) most influence predictions.

These visualizations enhance model transparency and help identify potential biases or overfitting.

## Limitations & Future Improvements
While the model performs exceptionally, key limitations exist:

1. No probability calibration—perfect confidence scores may not reflect real-world uncertainty.

2. Binary classification only—does not predict earthquake magnitude or location.

3. Possible data leakage—lag features may inadvertently expose future events.


# Images of Results
## Daily Earthquake Occurance
![Daily_Earthquake_Occurance](https://github.com/user-attachments/assets/08998fec-9753-460e-adc7-cf739cd059c7)


## PCA
![PCA](https://github.com/user-attachments/assets/d4766d48-c802-49e9-a03d-53d2889aac0a)


## Feature Importance
![PCA](https://github.com/user-attachments/assets/f7b3df52-d97e-41e8-a736-6f04119da5cf)


## Predicted Earthquake Probability vs Actual Events
![Predicted_Earthquake_Probability_vs_Actual_Events](https://github.com/user-attachments/assets/e56af1ac-51e7-4fcc-88ee-2ea521db4bc2)

