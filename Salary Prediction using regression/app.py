import warnings
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Suppress scikit-learn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the synthetic dataset
data = pd.read_csv("salary_data.csv")

# One-hot encode the "Education" column
education_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_education = education_encoder.fit_transform(data[["Education"]])
encoded_education = pd.DataFrame(encoded_education, columns=education_encoder.get_feature_names_out(["Education"]))

# One-hot encode the "CompanyType" column
company_type_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_company_type = company_type_encoder.fit_transform(data[["CompanyType"]])
encoded_company_type = pd.DataFrame(encoded_company_type, columns=company_type_encoder.get_feature_names_out(["CompanyType"]))

# One-hot encode the "Position" column
position_encoder = OneHotEncoder(sparse=False, drop='first')
encoded_position = position_encoder.fit_transform(data[["Position"]])
encoded_position = pd.DataFrame(encoded_position, columns=position_encoder.get_feature_names_out(["Position"]))

# Combine the encoded features with the rest of the data
data = pd.concat([data, encoded_education, encoded_company_type, encoded_position], axis=1)

# Define features and target variable
X = data[["Age", "Experience", "PastSalary_INR"] + list(encoded_education.columns) + list(encoded_company_type.columns) + list(encoded_position.columns)]
y = data["Salary_INR"]

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        education = request.form["education"]
        experience = int(request.form["experience"])
        company_type = request.form["company_type"]
        position = request.form["position"]
        past_salary_inr = int(request.form["past_salary_inr"])

        # Prepare user input for prediction
        encoded_education_col = "Education_" + education
        encoded_company_type_col = "CompanyType_" + company_type
        encoded_position_col = "Position_" + position

        # Check if the one-hot encoded columns exist, otherwise provide a default of 0
        user_data = [age, experience, past_salary_inr] + [
            0 if col not in X.columns else 1 for col in X.columns if col.startswith("Education_") or col.startswith("CompanyType_") or col.startswith("Position_")
        ]

        # Use the model to make a salary prediction
        predicted_salary = model.predict([user_data])[0]

        return render_template("result.html", predicted_salary=predicted_salary)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
