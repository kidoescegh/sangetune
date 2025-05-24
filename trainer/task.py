from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.estimator import Estimator

# Define your estimator (replace with your training image and settings)
estimator = Estimator(
    image_uri='206542257745.dkr.ecr.us-east-1.amazonaws.com/jancook:latest',
    role='arn:aws:iam::206542257745:role/service-role/AmazonSageMaker-ExecutionRole-20250524T155904',
    instance_count=1,
    instance_type='ml.m5.2xlarge',
    output_path='s3://omna/oo/'
)

# Define hyperparameter ranges to search
hyperparameter_ranges = {
    'max_depth': IntegerParameter(5, 15),
    'learning_rate': ContinuousParameter(0.01, 0.2)
}

# Define the objective metric and how to extract it from logs
objective_metric_name = 'validation:auc'
metric_definitions = [{'Name': 'validation:auc', 'Regex': 'validation-auc=([0-9\\.]+)'}]

# Create the HyperparameterTuner with random search strategy
tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_jobs=500,
    max_parallel_jobs=50,
    strategy='Random'  # Enable random search
)

# Launch the tuning job with your training data channels
tuner.fit({'train': 's3://omna/oo/', 'validation': 's3://omna/valid/'})
