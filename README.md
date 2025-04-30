# Fraud Detection System

Sistem cerdas untuk mendeteksi transaksi mencurigakan pada data perbankan dan kartu kredit menggunakan teknik machine learning.

## Background

Kasus fraud dalam transaksi perbankan dan kartu kredit terus meningkat dan menimbulkan kerugian finansial yang signifikan bagi lembaga keuangan maupun nasabah. Seiring berkembangnya teknologi, metode penipuan juga semakin canggih dan sulit dideteksi dengan pendekatan konvensional.

Machine learning offers a more effective approach because it can:
- Analyze millions of transactions and find hidden patterns
- Recognize suspicious behaviors that might be missed by traditional systems
- Adapt more quickly to the latest fraud trends
- Reduce false alarms that often disturb customers

Project ini dikembangkan untuk membantu lembaga keuangan mengidentifikasi transaksi fraud secara akurat dan real-time, dengan meminimalkan gangguan pada transaksi normal nasabah.

## Dataset

Data yang digunakan adalah `fraudTrain.csv` yang berisi catatan transaksi dengan berbagai informasi seperti:
- Transaction amount (`amt`)
- Purchase category (`category`)
- User and merchant location (`lat`, `long`, `merch_lat`, `merch_long`)
- Transaction time (`trans_date_trans_time`)
- User demographic information
- Transaction label (`is_fraud`) yang menunjukkan apakah transaksi tersebut fraud atau tidak

## Analysis Steps

Kode dibagi menjadi 12 tahapan utama:

1. **Data Loading**
   - Reading dataset from Google Drive
   - Exploring basic dataset information

2. **Exploratory Data Analysis (EDA)**
   - Analyzing data structure and distribution
   - Identifying missing values
   - Visualizing fraud vs non-fraud proportions

3. **Data Preprocessing**
   - Cleaning data and handling irrelevant values
   - Converting date formats

4. **Feature Engineering**
   - Creating time features (hour, day, month of transaction)
   - Calculating user age
   - Computing distance between user and merchant
   - Converting categorical variables to numeric

5. **Advanced Analysis & Visualization**
   - Visualizing transaction amount distribution
   - Analyzing patterns based on time
   - Analyzing correlations between features

6. **Data Preparation for Modeling**
   - Removing redundant features
   - Standardizing numeric features

7. **Train-Test Split**
   - Splitting data into 80% training and 20% testing

8. **Advanced Feature Engineering**
   - Calculating fraud rate per category based on training data
   - Applying safe transformations without data leakage

9. **Model Training & Evaluation**
   - Random Forest with default parameters
   - XGBoost with hyperparameter optimization
   - Model performance evaluation

10. **Model Stacking**
    - Combining the best models
    - Evaluating ensemble model performance

11. **Feature Importance Analysis**
    - Identifying the most influential features
    - Visualizing important features

12. **Model Comparison & Conclusion**
    - Comparing performance between models
    - Determining the best model

## Results

Perbandingan performa model (ROC AUC):

| Model | ROC AUC |
|-------|---------|
| Random Forest | 0.9925 |
| XGBoost (Tuned) | 0.9973 |
| Stacking | 0.9979 |

Model Stacking memberikan hasil terbaik dengan ROC AUC 0.9979, menunjukkan kemampuan yang sangat baik dalam membedakan transaksi normal dan fraud.

## How to Use

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

### Running the Code

1. Upload dataset `fraudTrain.csv` ke Google Drive
2. Mount Google Drive di Google Colab
3. Run the code sequentially from step 1 to 12

## Hyperparameter Tuning

For the XGBoost model, parameter optimization was done with RandomizedSearchCV using:
- n_estimators: [50, 100]
- learning_rate: [0.01, 0.1]
- max_depth: [3, 5]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]
- scale_pos_weight: [1, weight_ratio] (to handle class imbalance)

## Important Notes

1. **Handling Worker Timeout Warning**
   - If you see the warning "A worker stopped while some jobs were given to the executor", try:
     - Reducing dataset size for tuning
     - Reducing iterations and parameters
     - Reducing parallelism with n_jobs=2 
     - Using more memory-efficient XGBoost parameters

2. **Managing Class Imbalance**
   - Fraud data is typically much less common than non-fraud
   - Stratified sampling is used to maintain class distribution
   - The scale_pos_weight parameter is adjusted in XGBoost to give more weight to the minority class

## Next Steps

Beberapa pengembangan yang bisa dilakukan:
- Threshold optimization to balance precision and recall
- Implementing the model into a production system
- Developing a model performance monitoring dashboard
- In-depth analysis of factors contributing to fraud
