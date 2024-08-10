from flask import Flask, render_template, request
from joblib import load
import pandas as pd
from colleges import colleges
app = Flask(__name__)

# Load the trained model
knn_model = load('knn_model_final.joblib')

# Define the path to the dataset
DATASET_PATH = "tseamcet.csv"

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Drop unnecessary columns
df = df.drop(columns=['region', 'students_per_class', 'fee', 'seat_category', 'class_id', 'branch_code'])

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['gender', 'caste', 'branch', 'college_code'])

# Split the dataset into train and test sets
X = df_encoded.drop(columns=['college'])
y = df_encoded['college']

# Create a ranked list of colleges based on rank
ranked_colleges = df[['rank', 'college']].drop_duplicates().sort_values(by='rank').reset_index(drop=True)

# Create a set of unique colleges based on rank
colleges_set = sorted(set(ranked_colleges['college']), key=lambda x: ranked_colleges.loc[ranked_colleges['college'] == x, 'rank'].iloc[0])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    rank = int(request.form['rank'])
    gender = request.form['gender']
    caste = request.form['caste']
    branch = request.form['branch']

    # Preprocess the sample input
    sample_input_df = pd.DataFrame({'rank': [rank], 'gender': [gender], 'caste': [caste], 'branch': [branch]})
    sample_input_encoded = pd.get_dummies(sample_input_df, columns=['gender', 'caste', 'branch']).reindex(columns=X.columns, fill_value=0)

    # Make predictions
    predicted_college = knn_model.predict(sample_input_encoded)[0]

    # Find the index of the predicted college in the sorted colleges set
    sorted_colleges_list = sorted(colleges_set, key=lambda x: ranked_colleges.loc[ranked_colleges['college'] == x, 'rank'].iloc[0])
    predicted_college_code = predicted_college.split()[0]
    predicted_college_index = next((i for i, college in enumerate(colleges) if college.startswith(predicted_college_code)), -1)




    target_colleges = colleges[predicted_college_index :predicted_college_index + 10]

    return render_template('result.html', predicted_college=predicted_college, target_colleges=target_colleges)


if __name__ == '__main__':
    app.run(debug=True)
