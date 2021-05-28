import joblib
from termcolor import colored
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from memoized_property import memoized_property
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .data import get_data, clean_data
from .encoders import DistanceTransformer, TimeFeaturesEncoder
from .utils import compute_rmse


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[JP] [Tokyo] [bolinocroustibat] TaxiFareModel + 1.0"  # 🚨 replace with your country code, city, github_nickname and model name and version
STUDENT_NAME = "Adrien Carpentier"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        # self.pipeline = None
        self.X = X
        self.y = y
        self.set_pipeline()

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        # create the preproc pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        # create the final pipeline
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
            # ('randomforest_model', RandomForestRegressor())
        ])

    def run(self):
        '''returns a trained pipelined model'''
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


    # MLflow

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":

    # get data
    df = get_data(nrows=20_000)

    # clean data
    df = clean_data(df)

    # set X and y
    y = df.pop("fare_amount")
    X = df

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    trainer = Trainer(X=X_train, y=y_train)
    trainer.run()

    # evaluate
    rmse = trainer.evaluate(X_test=X_test, y_test=y_test)
    print(f"rmse: {rmse}")

    # Send to MLflow
    trainer.mlflow_log_param("student name", STUDENT_NAME)
    trainer.mlflow_log_param("estimator", "LinearRegression")
    trainer.mlflow_log_metric("rmse", rmse)

    # Save locally
    trainer.save_model()
