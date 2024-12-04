
import mlflow 
import random

mlflow.set_tracking_uri("http://ec2-3-107-174-141.ap-southeast-2.compute.amazonaws.com:5000/")

with mlflow.start_run():

    #log some random parameter
    mlflow.log_param("params1", random.randint(1,100))
    mlflow.log_param("params2", random.random())

    # log some random metric
    mlflow.log_metric('metric1',random.randint(5,10))
    mlflow.log_metric('metric2', random.randint(3,55))