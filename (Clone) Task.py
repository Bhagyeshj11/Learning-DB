# Databricks notebook source
# MAGIC %md I am Using a dataset from the UCI Machine Learning Repository, presented in Modeling wine preferences by data mining from physicochemical properties [Cortez et al., 2009].

# COMMAND ----------

import mlflow

# Set Experiment
experiment_name = "/Users/bhagyesh.joshi@fractal.ai/Wine Quality Experiment"
mlflow.set_experiment(experiment_name)

# Get Experiment ID
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment:
    experiment_id = experiment.experiment_id
    print("Experiment ID:", experiment_id)
else:
    print("Experiment does not exist.")


# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

data.isna().any()

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(["quality"], axis=1)
y = data.quality

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)



# COMMAND ----------

# MAGIC %md Building the first model with mlflow.start run

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict(model_input)

with mlflow.start_run(run_name='untuned_random_forest', experiment_id=experiment_id):
    n_estimators = 10
    max_depth = 10
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=123)
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred = model.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric('accuracy', accuracy)
    
    wrappedModel = SklearnModelWrapper(model)
    signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)


# COMMAND ----------

#import joblib

#joblib.dump(model, "random_forest_model.pkl")

# COMMAND ----------

import matplotlib.pyplot as plt
import os

feature_importances = model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()

image_path = "feature_importances_plot.png"
plt.savefig(image_path)

mlflow.log_artifact(image_path)
plt.close()

# COMMAND ----------

import mlflow

# Set the name of the existing model
model_name = "wine_quality_model"

# Find the run ID of the MLflow run containing the model artifacts
run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# Define the model URI using the run ID
model_uri = f"runs:/{run_id}/random_forest_model"

# Register the model URI as a new version of the existing model
model_version = mlflow.register_model(model_uri, model_name)

# Print the model version
print(f"Registered new version of model '{model_name}' with version ID: {model_version.version}")


# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_versions = client.search_model_versions(f"name='{model_name}'")
latest_model_version = max([model_version.version for model_version in model_versions])
latest_model_version

# COMMAND ----------

# MAGIC %md Build the second model with auto logging
# MAGIC

# COMMAND ----------

import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope
import mlflow

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}

def train_model(params):
  try:
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
    mlflow.xgboost.autolog()
    with mlflow.start_run(nested=True):
      train = xgb.DMatrix(data=X_train, label=y_train)
      validation = xgb.DMatrix(data=X_val, label=y_val)
      # Pass in the validation set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
      # is no longer improving.
      booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                          evals=[(validation, "validation")], early_stopping_rounds=50)
      validation_predictions = booster.predict(validation)
      accuracy = accuracy_score(y_val, np.round(validation_predictions))  # Calculate accuracy
      mlflow.log_metric('accuracy', accuracy)

      signature = infer_signature(X_train, booster.predict(train))
      mlflow.xgboost.log_model(booster, "model", signature=signature)

      # Set the loss to -1*accuracy so fmin maximizes the accuracy
      return {'status': STATUS_OK, 'loss': -1*accuracy, 'booster': booster.attributes()}
  except Exception as e:
    # Log failure and return status fail
    mlflow.log_param("error", str(e))
    return {'status': STATUS_FAIL}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
if mlflow.active_run():
    mlflow.end_run()
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=20,
    trials=spark_trials,
  )


# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.accuracy DESC']).iloc[0]
print(f'Accuracy of Best Run: {best_run["metrics.accuracy"]}')

# COMMAND ----------

import time
new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)
time.sleep(15)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
# Archive the old model version
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
   stage="Staging"
)

# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_versions = client.search_model_versions(f"name='{model_name}'")
latest_model_version = max([model_version.version for model_version in model_versions])
latest_model_version

# COMMAND ----------

#import pandas as pd

# Combine features and target labels for training and validation sets
#reference_data = pd.concat([X_train, y_train], axis=1)
#target_data = pd.concat([X_val, y_val], axis=1)

# Save reference data and target data to CSV files
#reference_data.to_csv("reference_data.csv", index=False)
#target_data.to_csv("target_data.csv", index=False)

# COMMAND ----------

#!pip uninstall evidently --yes
#!pip install evidently==0.2.8

# COMMAND ----------

#%sh
#pip list


# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import evidently

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# Split data into train, validation, and test sets
X = data.drop(["quality"], axis=1)
y = data.quality

# Split the data into train and remaining
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)


# COMMAND ----------

# MAGIC %md ##Feature Drift
# MAGIC comparing the statistical properties of features between different datasets (training vs. test/validation), it primarily addresses the detection of feature drift. Therefore, the type of drift detection being performed is Feature Drift.

# COMMAND ----------

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from mlflow.models.signature import infer_signature
import evidently
from evidently.dashboard import Dashboard

from evidently.tabs import DataDriftTab
from sklearn.model_selection import train_test_split

# Set Experiment
experiment_name = "/Users/bhagyesh.joshi@fractal.ai/concept_drift_monitoring"
mlflow.set_experiment(experiment_name)

# Load the existing models
rfc_model_version = mlflow.pyfunc.load_model("models:/wine_quality_model/1")
xgb_model_version = mlflow.pyfunc.load_model("models:/wine_quality_model/2")

# Calculate baseline metrics for the existing models
rfc_baseline_accuracy = accuracy_score(y_val, rfc_model_version.predict(X_val))
xgb_baseline_accuracy = accuracy_score(y_val, np.round(xgb_model_version.predict(X_val)))

# Log baseline metrics with MLflow
with mlflow.start_run(run_name='Baseline Metrics'):
    mlflow.log_metric('rfc_baseline_accuracy', rfc_baseline_accuracy)
    mlflow.log_metric('xgb_baseline_accuracy', xgb_baseline_accuracy)

# Get model predictions on the validation and test sets
rfc_val_predictions = rfc_model_version.predict(X_val)
rfc_test_predictions = rfc_model_version.predict(X_test)

xgb_val_predictions = xgb_model_version.predict(X_val)
xgb_test_predictions = xgb_model_version.predict(X_test)

# Get model drift dashboard for RFC using Evidently
rfc_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
rfc_drift_dashboard.calculate(X_train, X_test)
rfc_drift_dashboard.save("rfc_drift_dashboard.html")

# Log model drift dashboard for RFC with MLflow as artifact
mlflow.log_artifact("rfc_drift_dashboard.html", "rfc_drift")

# Get model drift dashboard for XGB using Evidently
xgb_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
xgb_drift_dashboard.calculate(X_train, X_test)
xgb_drift_dashboard.save("xgb_drift_dashboard.html")

# Log model drift dashboard for XGB with MLflow as artifact
mlflow.log_artifact("xgb_drift_dashboard.html", "xgb_drift")



# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

#pip install mlflow==2.12.2

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# Split data into train, validation, and test sets
X = data.drop(["quality"], axis=1)
y = data.quality

# Split the data into train and remaining
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)


# COMMAND ----------

# MAGIC %md ##Prediction Drift

# COMMAND ----------

#print("Shape of xgb_val_predictions:", xgb_val_predictions.shape)
#print("Shape of y_val_binary:", y_val_binary.shape)
#print("Shape of xgb_test_predictions:", xgb_test_predictions.shape)
#print("Shape of y_test_binary:", y_test_binary.shape)


# COMMAND ----------

#y_val_binary = y_val_binary.flatten()
#y_test_binary = y_test_binary.flatten()


# COMMAND ----------

#import numpy as np

#print("Unique values of y_val_binary:", np.unique(y_val_binary))
#print("Unique values of y_test_binary:", np.unique(y_test_binary))


# COMMAND ----------

#print("Sample of xgb_val_predictions:", xgb_val_predictions[:10])  # Print the first 10 elements
#print("Sample of xgb_test_predictions:", xgb_test_predictions[:10])  # Print the first 10 elements


# COMMAND ----------

'''import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from evidently.dashboard import Dashboard
from evidently.tabs import ClassificationPerformanceTab

# Set Experiment
experiment_name = "/Users/bhagyesh.joshi@fractal.ai/PREDICTION_drift_monitoring"
mlflow.set_experiment(experiment_name)

# Load the existing models
rfc_model_version = mlflow.pyfunc.load_model("models:/wine_quality_model/1")
xgb_model_version = mlflow.pyfunc.load_model("models:/wine_quality_model/2")

# Get model predictions on the validation and test sets
rfc_val_predictions = rfc_model_version.predict(X_val)
rfc_test_predictions = rfc_model_version.predict(X_test)

xgb_val_predictions = xgb_model_version.predict(X_val)
xgb_test_predictions = xgb_model_version.predict(X_test)

# Convert multiclass targets to binary format
lb = LabelBinarizer()
y_val_binary = lb.fit_transform(y_val)
y_test_binary = lb.transform(y_test)

# Create DataFrames with binary targets for XGB
xgb_val_df = pd.DataFrame({'prediction': xgb_val_predictions, 'target': y_val_binary[:, 0]})
xgb_test_df = pd.DataFrame({'prediction': xgb_test_predictions, 'target': y_test_binary[:, 0]})

# Create Classification Performance dashboard for RFC using Evidently
rfc_drift_dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
rfc_drift_dashboard.calculate(xgb_val_df, xgb_test_df)
rfc_drift_dashboard.save("rfc_classification_performance_dashboard.html")

# Log Classification Performance dashboard for RFC with MLflow as artifact
mlflow.log_artifact("rfc_classification_performance_dashboard.html", "rfc_classification_performance")

# Create Classification Performance dashboard for XGB using Evidently
xgb_drift_dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
xgb_drift_dashboard.calculate(xgb_val_df, xgb_test_df)
xgb_drift_dashboard.save("xgb_classification_performance_dashboard.html")

# Log Classification Performance dashboard for XGB with MLflow as artifact
mlflow.log_artifact("xgb_classification_performance_dashboard.html", "xgb_classification_performance")'''


# COMMAND ----------

#y_val_binary = y_val_binary.flatten()
#y_test_binary = y_test_binary.flatten()


# COMMAND ----------

'''import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import evidently
from evidently.dashboard import Dashboard
from evidently.tabs import ClassificationPerformanceTab
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# Set Experiment
experiment_name = "/Users/bhagyesh.joshi@fractal.ai/concept_drift_monitoring"
mlflow.set_experiment(experiment_name)

# Load the existing models
rfc_model_version = mlflow.pyfunc.load_model("models:/wine_quality_model/1")
xgb_model_version = mlflow.pyfunc.load_model("models:/wine_quality_model/2")

# Get model predictions on the validation and test sets
rfc_val_predictions = rfc_model_version.predict(X_val)
rfc_test_predictions = rfc_model_version.predict(X_test)

xgb_val_predictions = xgb_model_version.predict(X_val)
xgb_test_predictions = xgb_model_version.predict(X_test)

# Convert continuous target values to binary (example threshold of 0.5)
y_val_binary = (y_val > 0.5).astype(int)
y_test_binary = (y_test > 0.5).astype(int)

#  Create DataFrames with binary targets for XGB
xgb_val_df = pd.DataFrame({'prediction': xgb_val_predictions, 'target': y_val_binary.astype(bool)})
xgb_test_df = pd.DataFrame({'prediction': xgb_test_predictions, 'target': y_test_binary.astype(bool)})

# Collect predictions along with ground truth labels for RFC
rfc_val_predictions_with_labels = np.column_stack((rfc_val_predictions, y_val))
rfc_test_predictions_with_labels = np.column_stack((rfc_test_predictions, y_test))

# Convert numpy arrays to pandas DataFrames for RFC
rfc_val_df = pd.DataFrame(rfc_val_predictions_with_labels, columns=['prediction', 'target'])
rfc_test_df = pd.DataFrame(rfc_test_predictions_with_labels, columns=['prediction', 'target'])

# Create Classification Performance dashboard for RFC using Evidently
rfc_drift_dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
rfc_drift_dashboard.calculate(rfc_val_df, rfc_test_df)
rfc_drift_dashboard.save("rfc_classification_performance_dashboard.html")

# Log Classification Performance dashboard for RFC with MLflow as artifact
mlflow.log_artifact("rfc_classification_performance_dashboard.html", "rfc_classification_performance")

# Create Classification Performance dashboard for XGB using Evidently
xgb_drift_dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
xgb_drift_dashboard.calculate(xgb_val_df, xgb_test_df)
xgb_drift_dashboard.save("xgb_classification_performance_dashboard.html")

# Log Classification Performance dashboard for XGB with MLflow as artifact
mlflow.log_artifact("xgb_classification_performance_dashboard.html", "xgb_classification_performance")'''

# COMMAND ----------

'''print("Shape of xgb_val_predictions:", xgb_val_predictions.shape)
print("Shape of y_val_binary:", y_val_binary.shape)
print("Shape of xgb_test_predictions:", xgb_test_predictions.shape)
print("Shape of y_test_binary:", y_test_binary.shape)
'''

# COMMAND ----------

'''y_val_binary = y_val_binary.ravel()
y_test_binary = y_test_binary.ravel()
'''

# COMMAND ----------

# MAGIC %md Sending an email using the smtplib library

# COMMAND ----------

'''import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_alert():
    # Email configuration
    sender_email = ""
    receiver_email = ""
    smtp_server = ""
    smtp_port = 587
    smtp_username = ""
    smtp_password = ""

    # Email content
    subject = "Drift Alert: Drift Metrics Exceed Threshold"
    body = "Drift metrics have exceeded the threshold. Please take action."

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Connect to SMTP server and send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(message)

# Alerting
# Define threshold for drift metrics
threshold = 0.3

# Check if any drift metric exceeds the threshold
drift_exceeds_threshold = any(drift_metrics > threshold)

if drift_exceeds_threshold:
    # Trigger alert
    print("Drift metrics exceed threshold. Sending email alert...")
    send_email_alert()
else:
    print("Drift metrics within acceptable range. No alert needed.")'''


# COMMAND ----------



# COMMAND ----------

'''from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def send_slack_alert():
    # Slack configuration
    slack_token = "TOKEN"
    channel_id = "CHANNEL ID"

    # Slack client
    client = WebClient(token=slack_token)

    # Slack message
    message = "Drift metrics have exceeded the threshold. Please take action."

    try:
        # Send message
        response = client.chat_postMessage(channel=channel_id, text=message)
        print("Slack message sent successfully:", response["ts"])
    except SlackApiError as e:
        print("Error sending Slack message:", e.response["error"])

# Alerting
# Define threshold for drift metrics
threshold = 0.3

# Check if any drift metric exceeds the threshold
drift_exceeds_threshold = any(drift_metrics > threshold)

if drift_exceeds_threshold:
    # Trigger alert
    print("Drift metrics exceed threshold. Sending Slack alert...")
    send_slack_alert()
else:
    print("Drift metrics within acceptable range. No alert needed.")'''

