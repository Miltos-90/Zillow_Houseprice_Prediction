""" Function used for model training / validation. """
import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark.sql import DataFrame as SparkFrame

from . import constants as C


def preprocess(df: SparkFrame) -> SparkFrame:
    """ Performs the necessary preprocessing on the input dataframe.

        Args
            df: SparkFrame
                Dataframe to be transformed
        
        Outputs
            pyspark.sql.DataFrame: Transformed spark dataframe
    """
        
    # Compute the minimum values for the columns that will be log- or sqrt-transformed
    minCols = C.LOG_COLS + C.SQRT_COLS
    minDict = (
        df
        .select(*minCols)
        .agg( *[ F.min( F.col(col) ).alias(col) for col in minCols] )
        .first()
        .asDict()
    )

    # Integer columns that will be converted to floats.
    intCols = [col for (col, dtype) in df.dtypes if dtype == "int"]

    # Run preprocessing
    df = (
        df
        .fillna(C.NA_DICT)
        .withColumns(_transformations(minDict, intCols))
        .drop(*C.DROP_COLS)
    )

    return df


def _transformations(
    minima     : dict,
    intCols    : list,
    logCols    : list = C.LOG_COLS, 
    sqrtCols   : list = C.SQRT_COLS, 
    powerCols  : list = C.POWER_COLS, 
    replaceCols: list = C.VALUE_REPLACEMENT_COLS, 
    binaryCols : list = C.INDICATOR_COLS,
    ):

    """
        Generates a dictionary of Spark SQL expressions for data transformation based on specified columns.

        Args
            
            minima: dict
                A dictionary containing the minimum values for all fields.

            intCols: list
                List of column names of integer type. They will be converted to floats.

            logCols: list, optional
                List of column names for which logarithmic transformation is applied.

            sqrtCols: list, optional
                List of column names for which square root transformation is applied.

            powerCols: list, optional
                List of column names for which cubic root transformation is applied.

            replaceCols: list, optional
                List of columns with replacement rules (specified as namedtuples).
                
            binaryCols: list, optional
                List of columns to be converted to binary indicators.

        Outputs
            dict: A dictionary containing Spark SQL expressions for each specified transformation.

        Notes:
        - Logarithmic transformation is applied using the formula log10(col / average + 1e-3).
        - Square root transformation is applied using the square root function.
        - Cubic root transformation is applied using the formula col ** (1.0 / 3.0).
        - Value replacement is performed based on the rules specified in the replaceCols list (see constants.py).
        - Binary indicators are created based on a threshold for each specified column (see constants.py).

    """

    expressions = dict()

    # Log-transforms
    for col in logCols:

        if minima[col] <= 0.0:
            expressions[col] = F.log1p( F.col(col) + abs(minima[col]) ).alias(col)
        else:
            expressions[col] = F.log1p(F.col(col)).alias(col)

    # Square root transforms
    for col in sqrtCols:

        if minima[col] <= 0.0:
            expressions[col] = F.sqrt( F.col(col) + abs(minima[col]) ).alias(col)
        else:
            expressions[col] = F.sqrt( F.col(col) ).alias(col)

    # Power transforms
    for col in powerCols:
        expressions[col] = ( F.col(col) ** (1.0 / 3.0) ).alias(col)

    # Value replacement transforms
    for (col, valuesFrom, valueTo) in replaceCols:
        expressions[col] = (
            F.when(~F.col(col).isin(valuesFrom), valueTo)
            .otherwise( F.col(col) )
            .alias(col)
        )
    
    # Binary indicators
    for (col, threshold) in binaryCols:
        expressions[col] = (
            F.when(F.col(col) > threshold, 1)
            .otherwise(0)
            .alias("_".join([col, "bin"]))
        )

    # Tax delinquency flag conversion to numeric
    expressions["taxdelinquencyflag"] = (
        F.when(F.col("taxdelinquencyflag") == "Y", 1)
            .otherwise(0)
            .alias("taxdelinquencyflag")
    )

    # Convert rows with zero garage size and more than one cars to 
    # null garage size (so that they will be imputed in the pipeline).
    wrong = (F.col("garagetotalsqft") == 0) & ( F.col("garagecarcnt") > 0)

    expressions["garagetotalsqft"] = (
        F.when(wrong, None)
        .otherwise(F.col("garagetotalsqft"))
        .alias("garagetotalsqft")
    )

    # Extract year, month, day of the week from transactiondate
    expressions["transaction_year"]  = F.year("transactiondate").alias("transaction_year")
    expressions["transaction_month"] = F.month("transactiondate").alias("transaction_month")
    expressions["transaction_day"]   = F.dayofmonth("transactiondate").alias("transaction_day")


    # Convert all integer columns to float (otherwise pyspark's imptuer complains)
    for c in intCols:
        expressions[c] = F.col(c).cast(T.FloatType()).alias(c)

    return expressions


def _columnNames(columns: list) -> dict:
    """ Genenerates the column names used in the various transformers within the pipeline.

        Args
            columns: list
                List of columns in the input dataframe

        Outputs
            dict: Dictionary containing the input and output columns for each
                transformer in the pipeline.
    """
        
    # Appends a string to each element of a list
    addSuffix  = lambda string_, list_: [value + string_ for value in list_]   

    # Make the output column names from the various objects
    indexerOut = addSuffix("_indexed", C.INDEX_IN)
    encoderOut = addSuffix("_encoded", C.ONEHOT_IN)

    # The vectorAseembler's input columns are the features generated within the pipeline
    # (i.e. all the outputs from all transformers that are not subsequently fed into
    # other transformers), plus the features in the initial dataframe that are not used
    # by any transformers for further processing:
    outputFeats = set(columns + indexerOut + encoderOut + [C.HASH_OUT])
    inputFeats  = set(C.INDEX_IN + C.ONEHOT_IN + C.HASH_IN + [C.TARGET_COL])
    assemblerIn = list(outputFeats.difference(inputFeats))

    # Assemble all in a dictionary
    d = {
        "indexer"   : {"in": C.INDEX_IN,  "out": indexerOut},
        "encoder"   : {"in": C.ONEHOT_IN, "out": encoderOut},
        "hasher"    : {"in": C.INDEX_IN,  "out": C.HASH_OUT},
        "assembler" : {"in": assemblerIn, "out": C.ASSEMBLER_OUT}
    }

    return d