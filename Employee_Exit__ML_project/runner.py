import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


data = pd.read_csv('./submission/train.csv')
data.head()  # (59611, 24)
initial_m, initial_n = data.shape


data.isnull().sum()

data.drop_duplicates(inplace=True)
no_duplicate_m, no_duplicate_n = data.shape  # (59598, 24)

# 13 rows dropped
dropped = ( initial_m - no_duplicate_m, initial_n - no_duplicate_n )




# 1. Basic Dataset Information
def display_basic_info(data):
    print("1. BASIC DATASET INFORMATION")
    print("\nDataset Shape:", data.shape)
    print("\nDataset Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nDuplicate Rows:", data.duplicated().sum())

# 2. Statistical Summary
def display_statistical_summary(data):
    print("\n2. STATISTICAL SUMMARY")
    print("\nNumerical Columns Summary:")
    print(data.describe())
    print("\nCategorical Columns Summary:")
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col} value counts:")
        print(data[col].value_counts(normalize=True).round(3) * 100, "%")

# 3. Distribution Analysis
def plot_distributions(data):
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(15, len(numerical_cols)*4))
    
    for idx, col in enumerate(numerical_cols, 1):
        # Histogram
        plt.subplot(len(numerical_cols), 2, idx*2-1)
        sns.histplot(data=data, x=col, hue='exit_status', multiple="dodge")
        plt.title(f'Distribution of {col} by Exit Status')
        plt.xticks(rotation=45)
        
        # Box Plot
        plt.subplot(len(numerical_cols), 2, idx*2)
        sns.boxplot(data=data, y=col, x='exit_status')
        plt.title(f'Box Plot of {col} by Exit Status')
    
    plt.tight_layout()
    plt.show()

# 4. Correlation Analysis
def plot_correlation_matrix(data):
    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

# 5. Categorical Analysis
def plot_categorical_analysis(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    plt.figure(figsize=(15, len(categorical_cols)*4))
    
    for idx, col in enumerate(categorical_cols, 1):
        plt.subplot(len(categorical_cols), 2, idx*2-1)
        sns.countplot(data=data, x=col, hue='exit_status')
        plt.title(f'Count Plot of {col} by Exit Status')
        plt.xticks(rotation=45)
        
        # Calculate percentages
        plt.subplot(len(categorical_cols), 2, idx*2)
        pd.crosstab(data[col], data['exit_status'], normalize='index').plot(kind='bar')
        plt.title(f'Exit Rate by {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# 6. Age Group Analysis
def analyze_age_groups(data):
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 45, 55, 100], 
                              labels=['<25', '25-35', '35-45', '45-55', '55+'])
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='age_group', hue='exit_status')
    plt.title('Exit Status by Age Group')
    plt.show()

# 7. Income Distribution Analysis
def analyze_income_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=data, x='monthly_income', hue='exit_status')
    plt.title('Income Distribution by Exit Status')
    plt.show()

# 8. Work-Life Balance Impact
def analyze_work_life_balance(data):
    plt.figure(figsize=(10, 6))
    exit_rates = pd.crosstab(data['work_life_balance'], data['exit_status'], normalize='index')
    exit_rates['Left'].plot(kind='bar')
    plt.title('Exit Rate by Work-Life Balance')
    plt.show()

# 9. Job Satisfaction vs Performance
def analyze_job_satisfaction_performance(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.crosstab(data['job_satisfaction'], data['performance_rating'], 
                           values=data['exit_status'].map({'Left': 1, 'Stayed': 0}), 
                           aggfunc='mean'),
                annot=True, cmap='YlOrRd')
    plt.title('Exit Rate by Job Satisfaction and Performance')
    plt.show()

# 10. Tenure Analysis
def analyze_tenure(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='job_level', y='years_at_company', hue='exit_status')
    plt.title('Years at Company by Job Level and Exit Status')
    plt.show()

# 11. Remote Work Impact
def analyze_remote_work(data):
    plt.figure(figsize=(10, 6))
    remote_impact = pd.crosstab([data['remote_work'], data['work_life_balance']], 
                               data['exit_status'], normalize='index')
    remote_impact['Left'].unstack().plot(kind='bar')
    plt.title('Exit Rate by Remote Work and Work-Life Balance')
    plt.show()

# 12. Promotion vs Tenure Analysis
def analyze_promotions(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='years_at_company', y='promotions_count', 
                    hue='exit_status', size='monthly_income', sizes=(50, 400))
    plt.title('Promotions vs Tenure colored by Exit Status')
    plt.show()

# 13. Company Size Impact
def analyze_company_size(data):
    plt.figure(figsize=(12, 6))
    size_impact = pd.crosstab([data['company_size'], data['job_level']], 
                             data['exit_status'], normalize='index')
    size_impact['Left'].unstack().plot(kind='bar')
    plt.title('Exit Rate by Company Size and Job Level')
    plt.show()

# 14. Recognition vs Opportunities
def analyze_recognition_opportunities(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.crosstab(data['employee_recognition'], 
                           [data['leadership_opportunities'], data['innovation_opportunities']], 
                           values=data['exit_status'].map({'Left': 1, 'Stayed': 0}), 
                           aggfunc='mean'),
                annot=True, cmap='YlOrRd')
    plt.title('Exit Rate by Recognition and Opportunities')
    plt.show()

# 15. Multivariate Analysis
def perform_multivariate_analysis(data):
    # Select key numerical variables
    vars_of_interest = ['age', 'years_at_company', 'monthly_income', 
                       'promotions_count', 'distance_from_home']
    
    plt.figure(figsize=(15, 15))
    sns.pairplot(data=data[vars_of_interest + ['exit_status']], hue='exit_status')
    plt.suptitle('Multivariate Analysis of Key Numerical Variables', y=1.02)
    plt.show()

# Main execution function
def run_complete_eda(data):
    # Set style for all plots
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Run all analyses
    display_basic_info(data)
    display_statistical_summary(data)
    plot_distributions(data)
    plot_correlation_matrix(data)
    plot_categorical_analysis(data)
    analyze_age_groups(data)
    analyze_income_distribution(data)
    analyze_work_life_balance(data)
    analyze_job_satisfaction_performance(data)
    analyze_tenure(data)
    analyze_remote_work(data)
    analyze_promotions(data)
    analyze_company_size(data)
    analyze_recognition_opportunities(data)
    perform_multivariate_analysis(data)

# Run the complete EDA
run_complete_eda(data)