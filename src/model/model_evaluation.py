import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import mlflow
import mlflow.sklearn
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import json

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except FileNotFoundError:
        logger.error('Model file not found: %s', model_path)
        raise
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except FileNotFoundError:
        logger.error('Vectorizer file not found: %s', vectorizer_path)
        raise
    except Exception as e:
        logger.error('Error loading vectorizer: %s', e)
        raise

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X: np.ndarray, y: np.ndarray, dataset_name: str):
    """Evaluate the model and print the classification report."""
    try:
        y_pred = model.predict(X)
        report = classification_report(y, y_pred,output_dict=True)
        cm = confusion_matrix(y, y_pred)
        logger.debug(f"Classification report for {dataset_name}:\n{report}")
        
        return report, cm
    except Exception as e:
        logger.error(f'Error during model evaluation for {dataset_name}: {e}')
        raise

def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()



def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def save_model_info(run_id: str, model_path: str, file_path: str):
    """save model run_id and model path to json file."""
    try:
        # creating dict to save model info
        model_info={
            "run_id" : run_id,
            "model_path": model_path
        }
        # saving model info to jdon file
        with open(file_path,'w') as file:
            json.dump(model_info,file, indent=4)
        logger.debug(f"model info is save at: {file_path}")
    except Exception as e:
        logger.error(f"error occured during model info saving: {e}")
        raise 


def main():

    mlflow.set_tracking_uri("http://ec2-3-107-174-141.ap-southeast-2.compute.amazonaws.com:5000/")

    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            # Get root directory
            root_dir = get_root_directory()

            # load the parameter YMAL file
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            #log the parameter
            for key, value in params.items():
                mlflow.log_param(key,value)

            # Load the model and vectorizer from the root directory
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            
            # Log model parameters
            if hasattr(model, 'get_params'):
                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Log model and vectorizer
            mlflow.sklearn.log_model(model, "lgbm_model")
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            
            model_path = "lgbm_model"

            # save model info
            save_model_info(run.info.run_id, model_path, "experiment_info.json")


            # Load the training data and test data from the interim directory
            test_data = load_data(os.path.join(root_dir, 'data/processed/test_processed.csv'))

            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Evaluate on test data
            report,cm = evaluate_model(model, X_test_tfidf, y_test, "Test Data")

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


if __name__ == '__main__':
    main()