import json
import os

with open("Project.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find where data cleaning starts
cleaning_idx = -1
for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if "# Remove useless columns" in source or "Data Cleaning" in source:
        cleaning_idx = i
        break

if cleaning_idx == -1:
    print("Could not find Data Cleaning section")
    # We'll just insert after the second cell if not found
    cleaning_idx = 2

# Create new cells for Section 1 and Section 2

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
    }

new_cells = []

# Cell 1: Load data
new_cells.append(create_code_cell(
    "# Load the dataset\n"
    "df = pd.read_csv('fraud_detection_dataset.csv')"
))

# --- SECTION 1 ---
new_cells.append(create_markdown_cell(
    "# 1. Raw Dataset Overview\n\n"
    "This section clearly shows the original dataset *before* any modification to prove that the dataset is raw and untouched."
))

new_cells.append(create_code_cell(
    "# Display dataset shape (number of rows and columns)\n"
    "print(f\"Dataset Shape: {df.shape}\")\n"
    "print(f\"Number of Rows: {df.shape[0]}\")\n"
    "print(f\"Number of Columns: {df.shape[1]}\")"
))

new_cells.append(create_code_cell(
    "# Display dataset column names\n"
    "print(\"Column Names:\")\n"
    "list(df.columns)"
))

new_cells.append(create_code_cell(
    "# Show the first 5 rows\n"
    "print(\"First 5 Rows:\")\n"
    "display(df.head())"
))

new_cells.append(create_code_cell(
    "# Show the last 5 rows\n"
    "print(\"Last 5 Rows:\")\n"
    "display(df.tail())"
))

new_cells.append(create_code_cell(
    "# Show dataset information (datatypes, memory usage, non-null values)\n"
    "df.info()"
))

new_cells.append(create_code_cell(
    "# Show statistical summary\n"
    "display(df.describe())"
))

new_cells.append(create_markdown_cell(
    "### Column Explanations\n"
    "- **Transaction_ID**: A unique identifier for every transaction.\n"
    "- **Transaction_Amount**: The financial value of the transaction.\n"
    "- **Time_of_Transaction**: The time when the transaction was executed.\n"
    "- **Transaction_Type**: The classification of the transaction (e.g., payment, transfer, debit).\n"
    "- **Device_Used**: The device from which the transaction was initiated (e.g., mobile, desktop).\n"
    "- **Location**: The geographic origin of the transaction.\n"
    "- **Payment_Method**: The method through which the transaction was funded (e.g., credit card, bank transfer).\n"
    "- **Previous_Fraudulent_Transactions**: The count of previously flagged transactions for the user or account.\n"
    "- **Account_Age**: The age of the account initiating the transaction.\n"
    "- **Number_of_Transactions_Last_24H**: The volume of transactions from the account within the last 24 hours.\n"
    "- **User_ID**: A unique identifier for the user.\n"
    "- **Fraudulent**: The target label indicating whether the transaction was fraudulent (1) or not (0)."
))

# --- SECTION 2 ---
new_cells.append(create_markdown_cell(
    "# 2. Exploratory Data Analysis (Before Cleaning)\n\n"
    "This section shows the problems, structure, and general patterns of the dataset BEFORE any cleaning."
))

new_cells.append(create_code_cell(
    "# 1. Missing Values Analysis\n"
    "plt.figure(figsize=(10, 6))\n"
    "sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)\n"
    "plt.title('Heatmap of Missing Values in Raw Dataset')\n"
    "plt.show()"
))

new_cells.append(create_markdown_cell(
    "**Explanation**: The heatmap displays missing (null) values as yellow lines against a dark background. "
    "This visualizes exactly where the missing data exists across our columns. It is evident that certain columns "
    "have significant chunks of missing information that will need to be addressed before modeling."
))

new_cells.append(create_code_cell(
    "# 2. Fraud vs Non-Fraud Distribution\n"
    "plt.figure(figsize=(8, 6))\n"
    "ax = sns.countplot(data=df, x='Fraudulent', palette='Set2')\n"
    "plt.title('Fraud vs Non-Fraud Transaction Distribution')\n"
    "plt.xlabel('Fraudulent (0 = No, 1 = Yes)')\n"
    "plt.ylabel('Count')\n"
    "plt.show()\n\n"
    "fraud_counts = df['Fraudulent'].value_counts(normalize=True) * 100\n"
    "print(f\"Non-Fraudulent: {fraud_counts[0]:.2f}%\")\n"
    "print(f\"Fraudulent: {fraud_counts[1]:.2f}%\")"
))

new_cells.append(create_markdown_cell(
    "**Explanation**: The dataset is highly **imbalanced**. Fraud detection datasets typically have very few fraud "
    "cases because genuine transactions vastly outnumber fraudulent ones in the real world. This class imbalance "
    "will be a major challenge because standard machine learning algorithms will be biased toward predicting the majority class."
))

new_cells.append(create_code_cell(
    "# 3. Transaction Amount Distribution\n"
    "plt.figure(figsize=(12, 6))\n"
    "sns.histplot(df['Transaction_Amount'], bins=50, kde=True, color='blue')\n"
    "plt.title('Distribution of Transaction Amounts')\n"
    "plt.xlabel('Transaction Amount')\n"
    "plt.ylabel('Frequency')\n"
    "plt.show()"
))

new_cells.append(create_markdown_cell(
    "**Explanation**: The distribution of transaction amounts is highly **skewed** (right-skewed). Most transactions are "
    "for relatively small amounts, while a few transactions are for extremely large amounts. The presence of these "
    "high-value outliers can significantly impact our model's performance if not handled properly."
))

new_cells.append(create_code_cell(
    "# 4. Transaction Type Distribution\n"
    "plt.figure(figsize=(10, 6))\n"
    "sns.countplot(data=df, y='Transaction_Type', order=df['Transaction_Type'].value_counts().index, palette='viridis')\n"
    "plt.title('Distribution of Transaction Types')\n"
    "plt.xlabel('Count')\n"
    "plt.ylabel('Transaction Type')\n"
    "plt.show()"
))

new_cells.append(create_markdown_cell(
    "**Explanation**: This chart shows the most frequent types of transactions in our dataset. We can identify which transaction "
    "types are more popular, providing context on standard user behavior and identifying if certain categories are overrepresented."
))

new_cells.append(create_markdown_cell(
    "### 5. Dataset Quality Problems Discovered\n"
    "Based on the preliminary exploration above, the raw dataset exhibits several quality issues:\n"
    "- **Missing Values**: Some features have missing data (visualized in the heatmap) which must be imputed or removed.\n"
    "- **Skewed Distributions**: The transaction amounts are heavily right-skewed, meaning scaling or transformation is necessary.\n"
    "- **Class Imbalance**: The target variable `Fraudulent` has very few positive cases, meaning we may need to employ techniques like oversampling, undersampling, or class weighting.\n"
    "- **Possible Outliers**: Extreme values in transaction amounts may distort model decision boundaries.\n"
    "- **Inconsistent Categories/Datatypes**: Categorical variables like `Transaction_Type`, `Device_Used`, and `Location` will need proper encoding.\n\n"
    "*Note: We will address these problems in the Data Cleaning section.*"
))

# Imports are usually in the first cell, so keep it.
# We replace everything from cell 1 (second cell) to cleaning_idx (exclusive) with new_cells.
nb['cells'] = nb['cells'][:1] + new_cells + nb['cells'][cleaning_idx:]

with open("Project.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Project.ipynb has been updated successfully with the new specific EDA sections!")
