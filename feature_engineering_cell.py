# ========================================
# ADVANCED FEATURE ENGINEERING SECTION
# ========================================
# This section enhances the dataset with fraud-specific features
# Place this cell AFTER data cleaning and BEFORE model training

# Import our feature engineering module
from feature_engineering import apply_feature_engineering_to_notebook

# Apply feature engineering to the cleaned dataset
df_engineered = apply_feature_engineering_to_notebook(df)

# Display the enhanced dataset info
print("\n📈 Enhanced Dataset Overview:")
print(f"Shape: {df_engineered.shape}")
print(f"New Features Added: {df_engineered.shape[1] - df.shape[1]}")

# Show sample of new features
new_features = [col for col in df_engineered.columns if col not in df.columns]
print(f"\n🔍 Sample of Engineered Features:")
print(df_engineered[new_features[:5]].head())

# Update the main dataframe for downstream processing
df = df_engineered
print(f"\n✅ Main dataframe updated with {len(new_features)} new features!")
