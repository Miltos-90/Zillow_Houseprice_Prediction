# Zillow House Price Prediction

The following tackles the first round of [Zillow's Home Value Prediction Competition](https://www.kaggle.com/c/zillow-prize-1#description), which challenges competitors to predict the log error between Zillow's internal house price estimate (Zestimate) and the actual sale price of houses. The submissions are evaluated based on Mean Absolute Error between the predicted log error and the actual log error (see [here](https://www.kaggle.com/c/zillow-prize-1) for details). The competition was hosted from May 2017 to October 2017 on Kaggle, and the final private leaderboard was revealed after the evaluation period ended in January 2018.

The score of the model in this repo would have ranked very high on the first round, and would qualified for the private second round, which involves building a home valuation algorithm from ground up.

### Read Data

The following imports libraries and reads the data, incl. external / derived features. 

The additional features were extracted using OpenStreetMap/OSMNX, NASA's Digital Elevation Model, US Census Data, and they include:
* Number of several points of interest in a 500 meter radius around each property:
    * Cafes and restaturants, 
    * Amenities (hospitals, fire_stations, schools, police stations)
    * Roads & highways
    * Public transport options (train stations, airports)
    * Landscape (beach, hills, etc.)
    * Historic monuments (castles, churches, monasteries, etc.)
* The name of the nearest city.
* The distance from the district's coordinates to the nearest city.
* The population of the nearest city.
* The nearest 'big city' (a big city is categorized as > 250,000 residents in the given year).
* The distance to the nearest 'big city'.
* Elevation of each property using NASA's Digital Elevation Model
* Population density of the neighborhood each property is located in from the US Census Data using canpy

See the *feature_extraction.ipynb* notebook for details.


```python
import numpy as np
import mlflow
import json

from pathlib import Path
from xgboost.spark import SparkXGBRegressor as XGBoost

from pyspark.ml.feature import (
    VectorAssembler, FeatureHasher, OneHotEncoder, StringIndexer, Imputer 
    )
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from src.reader import read
import src.modelling as M

# Spark configuration
with open('./constants/config.json') as f   : config = json.load(f)
with open('./constants/modelling.json') as f: const = json.load(f)

conf = SparkConf()
conf.setMaster(config["spark_URL"])
conf.setAppName(config["app_name"])
[conf.set(key, value) for key, value in config["spark"].items()]
spark = SparkSession.builder.config(conf = conf).getOrCreate()

# MLflow configuration
mlflow.set_tracking_uri(uri = config["mlflow_URL"])

experimentID = M.makeExperiment(
    name = config["app_name"], 
    artifact_location = Path.cwd().joinpath(config["artifact_MLflow_dir"]).as_uri()
    )

# Read data + extracted features and perform preprocessing
df = read(spark, config["data"], config["features_file"])
df = M.preprocess(df)

dfTrain, dfTest = df.randomSplit(
    weights = [const["train_ratio"], 1 - const["train_ratio"]], 
    seed    = const["seed"]
    )
```

### Pyspark pipeline

Setup the pipeline to perform the necessary preprocessing and hyperparameter tuning of XGBoost


```python
gImputer  = M.GroupImputer(
    inputCol   = M.const.GROUP_IMPUTER_IN, 
    outputCol  = M.const.GROUP_IMPUTER_OUT, 
    groupCol   = "garagecarcnt", 
    stochastic = bool(const["impute_stochastic"])
    )

imputer = Imputer(
    inputCols  = M.const.IMPUTER_COLS, 
    outputCols = M.const.IMPUTER_COLS, 
    strategy   = const["impute_strategy"]
    )

indexer = StringIndexer(
    inputCols     = M.const.INDEX_IN, 
    outputCols    = M.const.INDEX_OUT, 
    handleInvalid = "keep"
    )

encoder = OneHotEncoder(
    inputCols     = M.const.ONEHOT_IN, 
    outputCols    = M.const.ONEHOT_OUT, 
    handleInvalid = "keep"
    )

hasher = FeatureHasher(
    inputCols = M.const.HASH_IN, 
    outputCol = M.const.HASH_OUT
    )

assembler = VectorAssembler(
    inputCols     = M.const.ASSEMBLER_IN, 
    outputCol     = M.const.ASSEMBLER_OUT, 
    handleInvalid = "keep"
    )

regressor = XGBoost(
    label_col      = M.const.TARGET_COL, 
    features_col   = M.const.ASSEMBLER_OUT, 
    prediction_col = M.const.PREDICTION_COL, 
    **M.const.XGB_CONSTANTS
    )

pipeline = Pipeline(stages = [gImputer, imputer, indexer, encoder, hasher, assembler, regressor]) 

evaluator = RegressionEvaluator(
    predictionCol = M.const.PREDICTION_COL, 
    labelCol      = M.const.TARGET_COL, 
    metricName    = const["metric"]
    )

h = const["hyperparameters"]
paramGrid = (
    M.RandomGridBuilder(numIterations = const["num_search_iterations"])
    .addGrid(hasher.numFeatures,         lambda: np.random.choice(h["hasher_num_features"]))
    .addGrid(regressor.max_depth,        lambda: np.random.choice(h["xgb_max_depth"]))
    .addGrid(regressor.n_estimators,     lambda: np.random.choice(h["xgb_n_estimators"]))
    .addGrid(regressor.reg_lambda,       lambda: np.random.choice(h["xgb_reg_lambda"]))
    .addGrid(regressor.learning_rate,    lambda: np.random.choice(h["xgb_learning_rate"]))
    .addGrid(regressor.gamma,            lambda: np.random.choice(h["xgb_gamma"]))
    .addGrid(regressor.colsample_bytree, lambda: np.random.choice(h["xgb_colsample_bytree"]))
    .addGrid(regressor.max_leaves,       lambda: np.random.choice(h["xgb_max_leaves"]))
    .addGrid(regressor.subsample,        lambda: np.random.choice(h["xgb_subsample"]))
    .build()
)

crossval = CrossValidator(
    estimatorParamMaps = paramGrid,
    estimator = pipeline,
    evaluator = evaluator,
    numFolds  = const["num_folds"])

model = crossval.fit(dfTrain)

M.logger(model, config["artifact_local_dir"], experimentID, tags = None)

results = model.transform(dfTest).select(["logerror", "prediction"]).toPandas()
```

Compute the error on the test set


```python
actual    = results["logerror"]
predicted = results["prediction"]
mae = np.abs(actual - predicted).mean()

print(f"Test set MAE: {mae:.5f}")
```

    Test set MAE: 0.07484
