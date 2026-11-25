# Fraud Detection Feature Engineering Documentation

## Overview
This document explains all the engineered features created to enhance the fraud detection dataset. Each feature is designed to capture specific behavioral patterns and risk indicators associated with fraudulent transactions.

## Feature Categories

### 1. Amount-Based Features

| Feature Name | Type | Fraud Detection Rationale |
|--------------|------|---------------------------|
| `High_Amount_Flag` | Binary (0/1) | Flags transactions above 75th percentile. Fraudsters often use high amounts to maximize gains. |
| `Very_High_Amount_Flag` | Binary (0/1) | Flags transactions above 95th percentile. Extremely high amounts are highly suspicious. |
| `Amount_to_Age_Ratio` | Continuous | Ratio of transaction amount to account age. High ratios for new accounts indicate potential fraud. |
| `High_Amount_Age_Ratio` | Binary (0/1) | Flags accounts with unusually high amount-to-age ratios. |
| `Log_Transaction_Amount` | Continuous | Log-transformed amount to normalize skewed distribution for better model performance. |

### 2. Time-Based Features

| Feature Name | Type | Fraud Detection Rationale |
|--------------|------|---------------------------|
| `Is_Night_Time` | Binary (0/1) | Flags transactions between 12 AM - 6 AM. Fraud often occurs during off-hours when monitoring is reduced. |
| `Is_Off_Hours` | Binary (0/1) | Flags transactions during late night (10 PM - 6 AM). Extended off-hours detection. |
| `Time_Category` | Categorical | Categorizes time into Night/Morning/Afternoon/Evening. Different risk levels for different periods. |

### 3. Behavioral Features

| Feature Name | Type | Fraud Detection Rationale |
|--------------|------|---------------------------|
| `Transaction_Velocity` | Continuous | Number of transactions per hour. High velocity indicates potential automated fraud. |
| `High_Velocity_Flag` | Binary (0/1) | Flags accounts with unusually high transaction frequency. |
| `Fraud_History_Score` | Continuous (0-1) | Normalized score based on previous fraudulent transactions. Users with fraud history are higher risk. |
| `Has_Fraud_History` | Binary (0/1) | Simple flag for users with any previous fraudulent transactions. |
| `New_Account_Flag` | Binary (0/1) | Flags accounts less than 30 days old. New accounts are common fraud targets. |
| `Very_New_Account_Flag` | Binary (0/1) | Flags accounts less than 7 days old. Very new accounts are extremely high risk. |
| `Frequency_Anomaly` | Continuous | Z-score of transaction frequency deviation from mean. Identifies unusual patterns. |
| `High_Frequency_Flag` | Binary (0/1) | Flags accounts with transaction frequency above 75th percentile. |

### 4. Device and Location Features

| Feature Name | Type | Fraud Detection Rationale |
|--------------|------|---------------------------|
| `Device_Risk_Score` | Continuous (0-1) | Risk score based on device type. Certain devices are more associated with fraud. |
| `Rare_Location_Flag` | Binary (0/1) | Flags transactions from rarely used locations. Unusual locations may indicate account takeover. |

### 5. Payment Method Features

| Feature Name | Type | Fraud Detection Rationale |
|--------------|------|---------------------------|
| `Payment_Method_Risk` | Continuous (0-1) | Risk score based on payment method. Some methods are more vulnerable to fraud. |

### 6. Composite Risk Features

| Feature Name | Type | Fraud Detection Rationale |
|--------------|------|---------------------------|
| `Composite_Risk_Score` | Continuous (0-1) | Weighted combination of multiple risk factors. Provides overall risk assessment. |
| `Risk_Tier` | Categorical | Categorizes transactions into Low/Medium/High/Very_High risk tiers. Useful for business rules. |

### 7. Statistical Features

| Feature Name | Type | Fraud Detection Rationale |
|--------------|------|---------------------------|
| `Amount_Deviation` | Continuous | Z-score of transaction amount deviation from population mean. Identifies unusual amounts. |
| `Amount_Anomaly_Flag` | Binary (0/1) | Flags transactions with amount deviation > 2 standard deviations. |
| `Transaction_Type_Risk` | Continuous (0-1) | Risk score based on transaction type. Some types are inherently riskier. |

## Implementation Details

### Feature Engineering Logic

1. **Threshold-based Features**: Use quantiles (75th, 95th percentiles) to define "high" or "unusual" values
2. **Risk Scores**: Normalized to 0-1 scale for interpretability
3. **Composite Scoring**: Weighted combination based on domain expertise
4. **Statistical Anomalies**: Z-score based detection of unusual patterns

### Risk Score Weights (Composite_Risk_Score)

- High Amount Flag: 20%
- Night Time Transaction: 15%
- High Velocity: 20%
- Fraud History: 25%
- New Account: 10%
- Device Risk: 10%

## Usage in Model

### Benefits for Machine Learning

1. **Improved Signal-to-Noise**: Features specifically designed for fraud detection
2. **Non-linear Relationships**: Binary flags capture threshold effects
3. **Business Interpretability**: Features align with fraud detection domain knowledge
4. **Risk Stratification**: Enables tiered risk management strategies

### Integration Steps

1. Apply after data cleaning (missing values handled)
2. Use as additional features in model training
3. Can be used for both supervised and unsupervised approaches
4. Supports real-time risk scoring in production

## Validation

### Expected Correlations

Features should show varying degrees of correlation with the target variable (`Fraudulent`):

- **High Correlation**: Fraud_History_Score, New_Account_Flag, High_Velocity_Flag
- **Medium Correlation**: Composite_Risk_Score, Night_Time_Flag, High_Amount_Flag
- **Low Correlation**: Some categorical features may show lower individual correlation

### Business Validation

- **Risk Tiers**: Should show increasing fraud rates from Low → Very_High
- **Feature Importance**: Model should prioritize fraud-relevant features
- **False Positive Reduction**: Features should help distinguish legitimate from fraudulent patterns

## Production Considerations

### Real-time Calculation

Most features can be calculated in real-time:

- **Static Features**: Account age, fraud history
- **Dynamic Features**: Transaction amount, time, frequency
- **Composite Features**: Combination of static and dynamic

### Performance Impact

- **Calculation Overhead**: Minimal (simple arithmetic and comparisons)
- **Memory Usage**: ~17 additional columns
- **Scalability**: Designed for high-volume transaction processing

## Maintenance

### Regular Updates

- **Thresholds**: Recalibrate quantiles periodically
- **Risk Weights**: Adjust based on model performance
- **New Features**: Add based on emerging fraud patterns

### Monitoring

- **Feature Drift**: Monitor distribution changes over time
- **Performance**: Track fraud detection rates by feature
- **Business Metrics**: Monitor false positive/negative rates

---

**Note**: This feature engineering implementation is designed to be modular and extensible. New features can be easily added following the same patterns and documentation standards.
