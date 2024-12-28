
import mlflow.sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask
import threading
import subprocess
import warnings
warnings.filterwarnings('ignore')


# Models dictionary
models = {
    "LR": LogisticRegressionCV(),
    "LSVC": LinearSVC(),
    "RFC": RandomForestClassifier(),
    "ABC": AdaBoostClassifier(),
    "GBC": GradientBoostingClassifier(),
    "DTC": DecisionTreeClassifier(),
    "GNB": GaussianNB()
}

# Initialize Flask app
app = Flask(__name__)

# Function to run MLflow UI in a separate thread
def start_mlflow_ui():
    subprocess.run(["mlflow", "ui"], check=True)

@app.route("/")
def route():
    # Enable MLflow autologging for scikit-learn models
    mlflow.sklearn.autolog()

    # Load dataset and split it
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    # Start MLflow UI in a separate thread
    ui_thread = threading.Thread(target=start_mlflow_ui)
    ui_thread.start()

    # Iterate over models and train them
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train the model
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log the accuracy manually (autologging will track other details like parameters, etc.)
            mlflow.log_metric("accuracy", accuracy)

            print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
    
    # Return response after training is completed
    return "Training completed, check the MLflow UI for details."

if __name__ == "__main__":
    # Start Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)
