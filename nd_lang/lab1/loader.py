import mlflow 

loaded_model = mlflow.pyfunc.load_model("my_model")

print("ok!")
