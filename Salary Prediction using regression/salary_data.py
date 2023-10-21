import pandas as pd
import numpy as np

# Define the number of data points in your dataset
num_samples = 1000

# Generate synthetic data
np.random.seed(0)  # For reproducibility
age = np.random.randint(22, 65, num_samples)
education = np.random.choice(["Bachelor's", "Master's", "Ph.D."], num_samples)
experience = np.random.randint(0, 20, num_samples)
company_type = np.random.choice(["Startup", "Corporation"], num_samples)
position = np.random.choice(["Engineer", "Data Scientist", "Manager", "Analyst"], num_samples)
past_salary_inr = np.random.randint(20000, 200000, num_samples)
salary_inr = np.random.randint(30000, 300000, num_samples)  # Target variable

# Create a DataFrame
data = pd.DataFrame({
    'Age': age,
    'Education': education,
    'Experience': experience,
    'CompanyType': company_type,
    'Position': position,
    'PastSalary_INR': past_salary_inr,
    'Salary_INR': salary_inr
})

# Save the synthetic dataset to a CSV file
data.to_csv('salary_data.csv', index=False)
