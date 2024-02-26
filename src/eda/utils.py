""" Set of utility functions. """

import pandas as pd


from pyspark.sql import DataFrame as SparkDataframe
from pyspark.ml.linalg import DenseMatrix
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler


def summaryContinuous(df: SparkDataframe) -> pd.DataFrame:
    """ Provides a comprehensive summary of the (assumed continuous)
        fields of a dataframe.

        Args
            df: Spark dataframe with continuous fields

        Outputs
            summarydf: Spark dataframe 
                Dataframe with each row corresponding to one field in the input 
                dataframe df, with the following fields:

                * count         : Number of non-null values
                * min           : Minimum value
                * lower_extreme : "Minimum", i.e. Q1 - 1.5 x IQR
                * 25%           : 1st quartile (25th percentile)
                * 50%           : Median value
                * 75%           : 3rd quartile (75th percentile)
                * upper_extreme : "Maximum", i.e. Q3 + 1.5 x IQR
                * max           : Maximum value
                * mean          : Mean value
                * stddev        : Standard deviation
                * skewness      : 2nd Pearson skewness coefficient

    """

    # Compute standard summary
    sumdf = (
        df
        .summary("count", "min", "25%", "mean", "50%", "75%", "max", "stddev")
        .toPandas()
        .set_index("summary")
        .apply(pd.to_numeric, errors='coerce')
    )

    # 2nd Pearson skewness coefficient
    sumdf.loc["skewness"] = 3 * (sumdf.loc["mean"] - sumdf.loc["50%"]) / sumdf.loc["stddev"]

    # Compute upper and lower extreme values (the values corresponding to the whiskers
    # in a boxplot)
    iqr = 3 * (sumdf.loc["mean"] - sumdf.loc["50%"]) / sumdf.loc["stddev"]
    sumdf.loc["lower_extreme"]  = sumdf.loc["25%"] - 1.5 * iqr
    sumdf.loc["upper_extreme"]  = sumdf.loc["75%"] + 1.5 * iqr

    # Re-order
    orderedCols = ["count", "min", "lower_extreme", "25%", "50%", "75%", "upper_extreme", "max", "mean", "stddev", "skewness"]
    sumdf = sumdf.reindex(orderedCols)
    
    return sumdf


def correlation(df: SparkDataframe, method: str) -> DenseMatrix:
    """ Evaluates the Pearson or Spearman correlation coefficient
        of a spark dataframe.

        Args:
            df: SparkDataframe
                The dataframe for which the correlation among all pairs
                of columns will be evaluated.
            method: str
                Correlation method to be evaluated. One of 'pearson'
                or 'spearman'.

        Returns:
            corr: DenseMatrix
                Correlation matrix.
    """

    if method not in ["pearson", "spearman"]:
        raise ValueError("Method should be one of 'pearson' or 'spearman'.")

    # Convert dataframe columns to vectors (required by the Correlation function)
    vectorizer = VectorAssembler(
        inputCols = df.columns, 
        outputCol = "features"
        )
    
    vectors = vectorizer.transform(df).select("features")

    # Check note in https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.ml.stat.Correlation.html
    # for Spearman correlation method.
    if method == "spearman": 
        vectors = vectors.cache()

    corr = Correlation.corr(
        vectors, 
        column = 'features', 
        method = 'spearman'
        ).collect()[0][0]
    
    return corr
