""" Logger for MLFlow """

import json
import mlflow
import os

import numpy as np
import pandas as pd

from pyspark.ml import Model
from pyspark.ml.tuning import CrossValidatorModel
from mlflow.entities import Experiment
from datetime import datetime as dt
from typing import Dict, Any, Optional


def makeExperiment(name: str, **kwargs) -> Experiment:
    """ Returns a new or existing MLFlow experiment.

        Args
            name: string
                Name of the experiment.

            kwargs: Keyword arguments for the create_experiment() function (see [1]).

        Outputs
            mlflow.entities.Experiment
                The corresponding experiment ID

        References
            [1] https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment
    """
        
    try: 
        experiment   = mlflow.get_experiment_by_name(name)
        experimentID = experiment.experiment_id

    except AttributeError:

        experimentID = mlflow.create_experiment(name = name, **kwargs)
        
    return experimentID


def _metadata(model: Model) -> Dict[str, Any]:
    """ Saves training artifacts.

        Args

            model: pyspark.ml.Model
                Fitted Pyspark model

        Outputs
            dict: Dictionary with all the metadata for each stage in the pipeline
    """

    stageDict = dict()
    for stageId, stage in enumerate(model.stages):

        paramDict = dict()
        for paramInfo, paramValue in stage.extractParamMap().items():
            paramDict[paramInfo.name] = json.dumps(paramValue)

        stageDict[f"stage_{stageId}"] = {
            "model"     : str(stage),
            "parameters": paramDict
            }

    return stageDict


def _detailedStats(model: CrossValidatorModel) -> pd.DataFrame:
    """ Extracts hyperparameters and summary statistics for the error metrics
        for all models in the cross validator object.

        Args
            model: pyspark.ml.tuning.CrossValidatorModel
            Cross validation model to extract details from.

        Outputs
            pd.DataFrame containing detialed results for each hyperparameter set.

    """
    metricName = model.getEvaluator().getMetricName() # 'mae', 'rmse', etc.
    iterable   = zip(model.getEstimatorParamMaps(), model.avgMetrics, model.stdMetrics)
    dicts      = [] # List of dictionaries containing the data needed from each model

    for parameterMap, avgMetric, stdMetric in iterable:

        d = dict() # Holds data for this estimator

        # Add metric average and std achieved by this estimator
        d[metricName + "_avg"] = avgMetric
        d[metricName + "_std"] = stdMetric
        d[metricName + "_se"]  = stdMetric / np.sqrt(model.getNumFolds())

        # Add parameter names and corresponding values
        for parameter, value in parameterMap.items():
            d[parameter.name] = value

        dicts.append(d)
        
    return pd.DataFrame.from_records(dicts)


def logger(
    model             : CrossValidatorModel,
    artifactDirectory : str,
    experimentID      : str,
    tags              : Optional[Dict[str, str]] = None
    ) -> None:
    """ MLFlow logger. It logs the following:
        * Artifacts
            * Trained model.
            * All parameters for all stages of the model.
            * Metrics for all hyperparameter sets tried during cross validation.
        * Metrics for the best model.
        * Hyperparameters for the best model.
        * Tags set by the user.

        Args
            model: spark.ml.tuning.CrossValidatorModel
                Pipeline (model) to log

            artifactDirecoty: string
                Path to save the artifacts for this run

            experimentID: string
                MLFlow experiment ID to which the run belongs

            tags: dictionary
                Dictionary of tags to be set by the user
    """
        
    # Make artifact subdirectory for this run
    curTime        = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
    artifactRunDir = os.path.join(artifactDirectory, f"{curTime}")

    if not os.path.exists(artifactRunDir):
        os.makedirs(artifactRunDir)

    # Make and save Artifacts
    metadata = _metadata(model.bestModel)
    stats    = _detailedStats(model)

    modelPath = os.path.join(artifactRunDir, "model")
    metaPath  = os.path.join(artifactRunDir, "metadata.json")
    statsPath = os.path.join(artifactRunDir, "stats.csv")

    model.bestModel.save(modelPath)
    stats.to_csv(statsPath, index = False)
    with open(metaPath, 'w') as f: json.dump(metadata, f, indent = 4)
    
    # Log
    with mlflow.start_run(
        experiment_id = experimentID,
        run_name = f"run_at_{curTime}"
        ):
        
        # Any required user tags
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # Metrics and parameters for best model
        bestId     = np.argmin(model.avgMetrics) # Index of the model with the lowest error
        metricMean = model.avgMetrics[bestId]
        metricStd  = model.stdMetrics[bestId]
        params_    = model.getEstimatorParamMaps()[bestId].items()
        params     = {param.name: value for param, value in params_}
        numFolds   = model.getNumFolds()
        metricSE   = metricStd / np.sqrt(numFolds) # Standard error
        
        mlflow.log_metric("metric_average", metricMean)
        mlflow.log_metric("metric_standard_error", metricSE)
        mlflow.log_metric("metric_standard_deviation", metricStd)
        [mlflow.log_param(k, v) for k, v in params.items()]

        # Artifacts
        mlflow.log_artifacts(local_dir = artifactRunDir)

    return