"""
Digital Fraud Detection - Feature Engineering Module
=====================================================

This module implements advanced feature engineering techniques to enhance
the fraud detection dataset with meaningful behavioral risk indicators.

Author: Senior ML Engineer
Purpose: Create fraud-specific features that improve model detection capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime

def engineer_fraud_features(df):
    """
    Engineer comprehensive fraud detection features from existing dataset columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset with fraud-related columns
        
    Returns:
    --------
    pd.DataFrame
        Enhanced dataset with new fraud detection features
    """
    
    # Create a copy to avoid modifying original data
    df_enhanced = df.copy()
    
    print("🔧 Starting Feature Engineering for Fraud Detection...")
    print(f"Original dataset shape: {df.shape}")
    
    # ==================== AMOUNT-BASED FEATURES ====================
    
    # 1. High Transaction Amount Flag
    # Rationale: Fraudsters often use unusually high amounts to maximize gains
    amount_75th = df['Transaction_Amount'].quantile(0.75)
    amount_95th = df['Transaction_Amount'].quantile(0.95)
    df_enhanced['High_Amount_Flag'] = (df['Transaction_Amount'] > amount_75th).astype(int)
    df_enhanced['Very_High_Amount_Flag'] = (df['Transaction_Amount'] > amount_95th).astype(int)
    
    # 2. Amount to Account Age Ratio
    # Rationale: New accounts with high transaction amounts are suspicious
    df_enhanced['Amount_to_Age_Ratio'] = df['Transaction_Amount'] / (df['Account_Age'] + 1)
    df_enhanced['High_Amount_Age_Ratio'] = (df_enhanced['Amount_to_Age_Ratio'] > 
                                            df_enhanced['Amount_to_Age_Ratio'].quantile(0.75)).astype(int)
    
    # 3. Log Transaction Amount
    # Rationale: Normalizes skewed distribution for better model performance
    df_enhanced['Log_Transaction_Amount'] = np.log1p(df['Transaction_Amount'])
    
    # ==================== TIME-BASED FEATURES ====================
    
    # 4. Night-Time Transaction Flag
    # Rationale: Fraudulent activities often occur during off-hours (12 AM - 6 AM)
    df_enhanced['Is_Night_Time'] = ((df['Time_of_Transaction'] >= 0) & 
                                   (df['Time_of_Transaction'] < 6)).astype(int)
    
    # 5. Weekend Transaction Flag (assuming hour-based proxy)
    # Rationale: Different transaction patterns on weekends
    df_enhanced['Is_Off_Hours'] = ((df['Time_of_Transaction'] >= 22) | 
                                   (df['Time_of_Transaction'] < 6)).astype(int)
    
    # 6. Time Category
    # Rationale: Different risk levels for different time periods
    def categorize_time(hour):
        if 0 <= hour < 6: return 'Night'
        elif 6 <= hour < 12: return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        else: return 'Evening'
    
    df_enhanced['Time_Category'] = df['Time_of_Transaction'].apply(categorize_time)
    
    # ==================== BEHAVIORAL FEATURES ====================
    
    # 7. Transaction Velocity Risk
    # Rationale: High frequency transactions in short time indicate potential fraud
    df_enhanced['Transaction_Velocity'] = df['Number_of_Transactions_Last_24H'] / 24
    df_enhanced['High_Velocity_Flag'] = (df_enhanced['Transaction_Velocity'] > 
                                        df_enhanced['Transaction_Velocity'].quantile(0.75)).astype(int)
    
    # 8. User Fraud History Score
    # Rationale: Users with previous fraud history are higher risk
    df_enhanced['Fraud_History_Score'] = df['Previous_Fraudulent_Transactions'] / 5  # Normalize to 0-1
    df_enhanced['Has_Fraud_History'] = (df['Previous_Fraudulent_Transactions'] > 0).astype(int)
    
    # 9. Account Risk Score
    # Rationale: New accounts are more susceptible to fraud
    df_enhanced['New_Account_Flag'] = (df['Account_Age'] < 30).astype(int)  # Less than 30 days
    df_enhanced['Very_New_Account_Flag'] = (df['Account_Age'] < 7).astype(int)  # Less than 7 days
    
    # 10. Transaction Frequency Risk
    # Rationale: Unusual transaction frequency patterns
    freq_mean = df['Number_of_Transactions_Last_24H'].mean()
    freq_std = df['Number_of_Transactions_Last_24H'].std()
    df_enhanced['Frequency_Anomaly'] = np.abs(df['Number_of_Transactions_Last_24H'] - freq_mean) / freq_std
    df_enhanced['High_Frequency_Flag'] = (df['Number_of_Transactions_Last_24H'] > 
                                         df['Number_of_Transactions_Last_24H'].quantile(0.75)).astype(int)
    
    # ==================== DEVICE AND LOCATION FEATURES ====================
    
    # 11. Device Risk Score
    # Rationale: Certain devices are more associated with fraud
    device_risk_mapping = {
        'Mobile': 0.3,
        'Desktop': 0.2,
        'Tablet': 0.4,
        'Unknown': 0.8
    }
    df_enhanced['Device_Risk_Score'] = df['Device_Used'].map(device_risk_mapping).fillna(0.5)
    
    # 12. Location Anomaly Flag
    # Rationale: Unusual transaction locations may indicate fraud
    location_counts = df['Location'].value_counts()
    rare_locations = location_counts[location_counts < location_counts.quantile(0.1)].index
    df_enhanced['Rare_Location_Flag'] = df['Location'].isin(rare_locations).astype(int)
    
    # ==================== PAYMENT METHOD FEATURES ====================
    
    # 13. Payment Method Risk Score
    # Rationale: Some payment methods are more vulnerable to fraud
    payment_risk_mapping = {
        'Credit Card': 0.3,
        'Debit Card': 0.2,
        'UPI': 0.4,
        'Net Banking': 0.3,
        'Wallet': 0.5,
        'Unknown': 0.8
    }
    df_enhanced['Payment_Method_Risk'] = df['Payment_Method'].map(payment_risk_mapping).fillna(0.5)
    
    # ==================== COMPOSITE RISK FEATURES ====================
    
    # 14. Composite Risk Score
    # Rationale: Combines multiple risk factors into a single score
    df_enhanced['Composite_Risk_Score'] = (
        df_enhanced['High_Amount_Flag'] * 0.2 +
        df_enhanced['Is_Night_Time'] * 0.15 +
        df_enhanced['High_Velocity_Flag'] * 0.2 +
        df_enhanced['Has_Fraud_History'] * 0.25 +
        df_enhanced['New_Account_Flag'] * 0.1 +
        df_enhanced['Device_Risk_Score'] * 0.1
    )
    
    # 15. Risk Tier Classification
    # Rationale: Categorizes transactions into risk levels for business rules
    def classify_risk(score):
        if score >= 0.7: return 'Very_High'
        elif score >= 0.4: return 'High'
        elif score >= 0.2: return 'Medium'
        else: return 'Low'
    
    df_enhanced['Risk_Tier'] = df_enhanced['Composite_Risk_Score'].apply(classify_risk)
    
    # ==================== STATISTICAL FEATURES ====================
    
    # 16. Amount Deviation from User Mean (simulated)
    # Rationale: Transactions significantly different from user's average are suspicious
    # Note: In real scenario, this would require user-level historical data
    amount_mean = df['Transaction_Amount'].mean()
    amount_std = df['Transaction_Amount'].std()
    df_enhanced['Amount_Deviation'] = (df['Transaction_Amount'] - amount_mean) / amount_std
    df_enhanced['Amount_Anomaly_Flag'] = (np.abs(df_enhanced['Amount_Deviation']) > 2).astype(int)
    
    # 17. Transaction Type Risk
    # Rationale: Some transaction types are inherently riskier
    transaction_risk_mapping = {
        'Online Purchase': 0.4,
        'ATM Withdrawal': 0.2,
        'Bank Transfer': 0.3,
        'Bill Payment': 0.2,
        'POS Payment': 0.3
    }
    df_enhanced['Transaction_Type_Risk'] = df['Transaction_Type'].map(transaction_risk_mapping).fillna(0.3)
    
    print(f"✅ Feature Engineering Complete!")
    print(f"Enhanced dataset shape: {df_enhanced.shape}")
    print(f"New features added: {df_enhanced.shape[1] - df.shape[1]}")
    
    return df_enhanced

def display_feature_summary(df_original, df_enhanced):
    """
    Display a summary of engineered features and their fraud correlation.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataset
    df_enhanced : pd.DataFrame
        Dataset with engineered features
    """
    
    print("\n" + "="*60)
    print("📊 FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    # Get only the new features
    new_features = [col for col in df_enhanced.columns if col not in df_original.columns]
    
    print(f"\n🔢 Total New Features Created: {len(new_features)}")
    print("\n📋 New Features List:")
    for i, feature in enumerate(new_features, 1):
        print(f"   {i:2d}. {feature}")
    
    # Show correlation with fraud for numerical features
    print("\n🎯 Feature-Fraud Correlations (Top 10):")
    numerical_features = df_enhanced[new_features].select_dtypes(include=[np.number]).columns
    
    if 'Fraudulent' in df_enhanced.columns:
        correlations = df_enhanced[numerical_features].corrwith(df_enhanced['Fraudulent']).abs()
        top_correlations = correlations.sort_values(ascending=False).head(10)
        
        for feature, corr in top_correlations.items():
            print(f"   {feature:<25}: {corr:.4f}")
    
    print("\n✅ Feature Engineering Successfully Completed!")
    print("="*60)

# Example usage function for the notebook
def apply_feature_engineering_to_notebook(df):
    """
    Convenience function to apply feature engineering in Jupyter notebook.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The cleaned dataframe from previous steps
        
    Returns:
    --------
    pd.DataFrame
        Enhanced dataframe with engineered features
    """
    
    print("🚀 Applying Advanced Feature Engineering...")
    df_enhanced = engineer_fraud_features(df)
    display_feature_summary(df, df_enhanced)
    
    return df_enhanced

if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module Ready for Import!")
    print("Use: from feature_engineering import apply_feature_engineering_to_notebook")
