""" (Stochastic) Group-wise imputation estimator and transformer: 
    Groups the dataset by a column, and imputes using the 
    mean value within the group. Stochastic imputation with the
    standard deviation (mean + ranom.standard.normal * std)
    is also supported.
"""

import pyspark.sql.functions as F
import pyspark.sql.types as T

from typing import Any, Dict, List, Optional

from pyspark import keyword_only

from pyspark.sql import SparkSession, Column
from pyspark.sql import DataFrame as SparkFrame
from pyspark.sql.types import DataType

from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import (
    HasInputCol, HasOutputCol, HasInputCols, HasOutputCols, Param, Params, TypeConverters
)

spark = SparkSession.builder.getOrCreate()

# Internal variables
SEED = 42 # Used for stochastic imputation in the _transform() method.

# Column names for the lookup table that holds the summary statistics.
# Note that an additional column is added with the name set in the groupCol 
# argument inside the _lookupdf() function of the GroupImputerModel class
MEAN_COLUMN_NAME = "__GroupImputerMean"
STD_COLUMN_NAME  = "__GroupImputerStd"

# Column name for the additional temporary column that will be created 
# when joining the input dataset with the lookup table. This column will
# (temporarily) contain the imputed values that will be conditionally used.
TEMP_COLUMN_NAME = "__GroupImputerTemp"

# Accepted datatypes for the input column (i.e. all numeric types)
# See here for a list: https://spark.apache.org/docs/latest/sql-ref-datatypes.html
NUMERIC_TYPES = [
    T.ByteType(),
    T.ShortType(),
    T.IntegerType(),
    T.LongType(),
    T.FloatType(),
    T.DoubleType(),
    T.DecimalType()
]

# Typehint for the lookup table.
LookupDict = Dict[int, Dict[str, Any]]

""" Transformer parameters """
class GroupImputerParams(
    HasInputCol, HasOutputCol, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable
):
    
    lookupTable: Param[LookupDict] = Param(
        Params._dummy(),
        "lookupTable",
        "Lookup table containing necessary intra-group statistics",
        TypeConverters.identity,
    )

    groupCol: Param[str] = Param(
        Params._dummy(),
        "groupCol",
        "Group column name.",
        TypeConverters.toString
    )

    stochastic: Param[bool] = Param(
        Params._dummy(),
        "stochastic",
        "Flag indicating if the values should be imputed randomly"
        "taking into account the intra-group standard deviation."
        "Defaults to False.",
        TypeConverters.toBoolean
    )

    
    def __init__(self, *args):
        super().__init__(*args)
        
        self._setDefault(
            lookupTable = { 0: {'group': None, 'mean': None, 'std': None} }, 
            groupCol    = None,
            stochastic  = False
            )

    """ Getters """
    def getLookupTable(self) -> LookupDict:
        return self.getOrDefault(self.lookupTable)
    
    def getGroupCol(self) -> str:
        return self.getOrDefault(self.groupCol)

    def getStochastic(self) -> bool:
        return self.getOrDefault(self.stochastic)


class GroupImputerModel(Model, GroupImputerParams):

    @keyword_only
    def __init__(
        self,
        inputCol    : str            = None, 
        outputCol   : str            = None,
        inputCols   : List[str]      = None,
        outputCols  : List[str]      = None,
        groupCol    : str            = None,
        stochastic  : Optional[bool] = False,
        lookupTable : LookupDict     = None,
    ):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

        return
    
    """ Setters """
    @keyword_only
    def setParams(
        self, 
        inputCol    : str            = None, 
        outputCol   : str            = None,
        inputCols   : List[str]      = None,
        outputCols  : List[str]      = None,
        groupCol    : str            = None,
        stochastic  : Optional[bool] = False,
        lookupTable : LookupDict     = None,
    ) -> "GroupImputerModel":
        
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    
    def setInputCol(self, newInputCol: str) -> "GroupImputerModel":
        return self._set(inputCol = newInputCol)

    def setOutputCol(self, newOutputCol: str) -> "GroupImputerModel":
        return self._set(outputCol = newOutputCol)

    def setInputCols(self, newInputCols: List[str]) -> "GroupImputerModel":
        return self._set(inputCols = newInputCols)

    def setOutputCols(self, newOutputCols: List[str]) -> "GroupImputerModel":
        return self._set(outputCols = newOutputCols)
    
    def setGroupCol(self, newGroupCol: str) -> "GroupImputerModel":
        return self._set(groupCol = newGroupCol)

    def setLookupTable(self, newLookup: LookupDict) -> "GroupImputerModel":
        return self._set(lookupTable = newLookup)

    def setStochastic(self, newStochasticFlag: bool) -> "GroupImputerModel":
        return self._set(stochastic = newStochasticFlag)
    
    """ Name getters for the various column names """
    @staticmethod
    def _meanName(name: str) -> str: return name + MEAN_COLUMN_NAME

    @staticmethod
    def _stdName(name: str) -> str: return name + STD_COLUMN_NAME
    
    @staticmethod
    def _newValueExpressions(
        stochastic: bool, inputColNames: List[str], outputColNames: List[str]
        ) -> Dict[str, Column]:
        """
        Generate expressions for imputed values based on input columns.

        Args
            stochastic: bool
                If True, generate stochastic imputed values using random noise and standard deviation.
            inputColNames: list
                List of input column names for imputation.
            outputColNames: list
                List of output column names corresponding to imputed values.

        Outputs
            dict: A dictionary mapping output column names to expressions for imputed values.
        """
        
        expressions = dict()
        randn = F.randn(seed = SEED)

        for cIn, cOut in zip(inputColNames, outputColNames):

            imputedValue = F.col(GroupImputerModel._meanName(cIn))

            if stochastic:
                imputedValue += randn * F.col(GroupImputerModel._stdName(cIn))

            expressions[cOut] = imputedValue.alias(cOut)
        
        return expressions

    @staticmethod
    def _replacementExpressions(
        inputColNames: List[str], outputColNames: List[str], datatypes: List[DataType]
        ) -> Dict[str, Column]:
        """
        Generates expressions for replacing null values in input columns with default values,
        and optionally casting the result to specified data types.

        Args
            inputColNames: list
                List of input column names with potential null values.
        
            outputColNames: list
                List of output column names for storing replacement expressions.
        
            datatypes: list
                List of target data types for casting, corresponding to output columns.

        Outputs
            dict: A dictionary mapping output column names to replacement expressions.

        """

        expressions = dict()

        for cIn, cOut, type_ in zip(inputColNames, outputColNames, datatypes):

            expressions[cOut] = (
                F.when( F.col(cIn).isNull(), F.col(cOut) )
                .otherwise(F.col(cIn))
                .cast(type_)
            )

        return expressions

    def _transform(self, dataset: SparkFrame) -> SparkFrame:
        """ Transform function. Imputes the given dataset

            Args
                dataset: pyspark.sql.DataFrame
                    Dataframe to be transformed.

            Outputs
                pyspark.sql.DataFrame: Transformed dataframe.
        """

        if self.getInputCol() is None:
            inputColNames  = self.getInputCols()
            outputColNames = self.getOutputCols()
        else:
            inputColNames  = (self.getInputCol(),)
            outputColNames = (self.getOutputCol(),)
            
        groupColName = self.getGroupCol()
        stochastic   = self.getStochastic()

        # Make the expressions needed for the imputation
        datatypes    = [dataset.schema[c].dataType for c in inputColNames]
        newColumns   = self._newValueExpressions(stochastic, inputColNames, outputColNames)
        replacements = self._replacementExpressions(inputColNames, outputColNames, datatypes)

        # Make the lookup table. If the dict contains None values they will be converted to 'NaN'
        # when generating to dataframe. The following re-converts the 'NaN' values to None (NULL in spark)
        lookup    = spark.createDataFrame(self.getLookupTable().values())
        nanToNull = lambda col: F.when(F.isnan(F.col(col)), None).otherwise(F.col(col)).alias(col)
        lookup    = lookup.select(*[nanToNull(col) for col in lookup.columns])

        # When joining the lookup table, some intermediate columns will be appended to the dataframe.
        # Tese need to be removed at the end. The following lists contains their names.
        tempCols = list( set(lookup.columns).difference([groupColName]) )

        # Perform imputation
        datasetImputed = (
            dataset
            .join(lookup, on = groupColName, how = "left")
            .withColumns(newColumns)
            .withColumns(replacements)
            .drop(*tempCols)
            )

        return datasetImputed


class GroupImputer(Estimator, GroupImputerParams):

    @keyword_only
    def __init__(
        self, 
        inputCol    : str            = None, 
        outputCol   : str            = None,
        inputCols   : List[str]      = None,
        outputCols  : List[str]      = None,
        groupCol    : str            = None,
        stochastic  : Optional[bool] = False,
        ):

        super().__init__()
        kwargs = self._input_kwargs

        self._setDefault(
            inputCol   = None, 
            outputCol  = None,
            inputCols  = None,
            outputCols = None,
            groupCol   = None,
            stochastic = False
        )

        self.setParams(**kwargs)

        return

    """ Setters """
    @keyword_only
    def setParams(
        self, 
        inputCol    : str            = None, 
        outputCol   : str            = None,
        inputCols   : List[str]      = None,
        outputCols  : List[str]      = None,
        groupCol    : str            = None,
        stochastic  : Optional[bool] = False,
        ) -> "GroupImputer":

        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def setGroupCol(self, newGroupCol: str) -> "GroupImputer":
        return self._set(groupCol = newGroupCol)

    def setInputCol(self, newInputCol: str) -> "GroupImputer":
        return self._set(inputCol = newInputCol)

    def setOutputCol(self, newOutputCol: str) -> "GroupImputer":
        return self._set(outputCol = newOutputCol)
    
    def setInputCols(self, newInputCols: List[str]) -> "GroupImputer":
        return self._set(inputCols = newInputCols)

    def setOutputCols(self, newOutputCols: List[str]) -> "GroupImputer":
        return self._set(outputCols = newOutputCols)
    
    def setStochastic(self, newStochasticFlag: bool) -> "GroupImputer":
        return self._set(stochastic = newStochasticFlag)

    @staticmethod
    def _makeLookup(inputColNames: list, groupColName: str, dataset: SparkFrame) -> LookupDict:
        """ 
        Creates the lookup table that will be used to impute the missing
        values of the input column. It contains summary statistics 
        (mean, std) of the input column per group.

        Args
            dataset: pyspark.sql.DataFrame
                Dataframe to be transformed.

        Outputs
            dict: Lookup table as a python dictionary.
        
        NOTE
        The schema of the lookup table prior to flattening is as follows:

        groupCol | mean(column_1) | std(column_1) | mean(column_2) | ... | std(column_N)
        
        where the names of the columns are as follows:
        mean(column_i): column_i + MEAN_COLUMN_NAME
        std(column_i) : column_i + STD_COLUMN_NAME
        """

        # Expressions to evaluate the mean and standard deviation
        expressions = []
        for col in inputColNames:
            
            meanName = col + MEAN_COLUMN_NAME
            stdName  = col + STD_COLUMN_NAME
            
            meanExpr = F.mean(col).alias(meanName)
            stdExpr  = F.std(col)

            # In case a single element exists for a group in the groupby operation
            # the standard deviation will be null. To avoid errors further downstream,
            # these nulls will be converted to zeros, and imputation using only the mean
            # will be performed.  
            stdExprNull = (
                F.when(stdExpr.isNull() & meanExpr.isNotNull(), 0)
                .otherwise(stdExpr)
                .alias(stdName)
            )

            expressions.append(meanExpr)
            expressions.append(stdExprNull)

        # Make lookup table (python dictionary to allow for JSON serialization)
        lookup = (
            dataset
            .groupby(groupColName)
            .agg(*expressions)
            .toPandas()
            .T
            .to_dict('dict')
        )

        return lookup
    
    def _exactlyOneSet(self, variableA: str, variableB: str) -> None:
        """ Checks if exactly one variable has been set from the pair 
            of variables given. Raises value error if not.

            Args
                variableA: string
                    Name of the first variable
                
                variableB: string
                    Name of the second variable   
        """

        bothSet = variableA and variableB
        noneSet = (not variableA) and (not variableB)

        if bothSet or noneSet:
            raise ValueError(
                f"Exactly one of `{variableA}` and `{variableB}` must be set."
            )
        
        return

    def _checkParams(self) -> None:
        """ Checks if all parameters have been set correctly.
            Raises Value error if any of the parameters has not been set.
        """

        # Get all inputs
        inputCol   = self.getInputCol()
        outputCol  = self.getOutputCol()
        inputCols  = self.getInputCols()
        outputCols = self.getOutputCols()
        groupCol   = self.getGroupCol()

        # Test 1: Either inputCol or inputCols can be set (but not both and not none).
        self._exactlyOneSet(inputCol, inputCols)

        # Test 2: Either outputCol or outputCols can be set (but not both and not none).
        self._exactlyOneSet(outputCol, outputCols)
        
        # Test 3: If `inputCols` is set, then `outputCols` must also be set.
        if inputCols and not outputCols:
            raise ValueError("If `inputCols` is set, then `outputCols` must also be set.")
        
        # Test 4: If `inputCols` is set, then `outputCols` must also be set.
        if outputCols and not inputCols:
            raise ValueError("If `outputCols` is set, then `inputCols` must also be set.")

        # Test 5: If `inputCol` is set, then `outputCol` must also be set.
        if inputCol and not outputCol:
            raise ValueError("If `inputCol` is set, then `outputCol` must also be set.")
        
        # Test 6: If `outputCol` is set, then `inputCol` must also be set.
        if inputCol and not outputCol:
            raise ValueError("If `outputCol` is set, then `inputCol` must also be set.")

        # Test 7: If `inputCols` is set, then `outputCols` must be a list of the same length
        if inputCols and outputCols:
            if not len(inputCols) == len(outputCols):
                raise ValueError("The length of `inputCols` does not match the length of `outputCols`")
        
        # Test 8: `groupCol` must always be set
        if not groupCol: 
            raise ValueError(f"GroupImputerModel: `groupCol` has not been set.")
        
        return

    @staticmethod
    def _checkInput(inputColumns: list, groupCol: str, dataset: SparkFrame) -> None:
        """ Checks the input with the parameters set. Raises value error
            if any checks fail.

            Args
                inputColumns: list
                    List of names with the set input columns
                
                groupCol: string
                    Name fo the grouping column
                
                dataset: spark.sql.DataFrame
                    Dataset to be used for the checks
            
        """

        # Test 1: All input columns should exist in the dataset
        inDataset = lambda col: col in dataset.columns
        if not all([inDataset(c) for c in inputColumns]):
            raise ValueError("Not all `inputCols` were found in the dataset.")
    
        # Test 2: Group column should exist in the dataset
        if not inDataset(groupCol):
            raise ValueError("`groupCol` was not found in the dataset.")
        
        # Test 3: All input columns should be numeric
        datatypes = [dataset.schema[c].dataType for c in inputColumns]
        if not all([type_ in NUMERIC_TYPES for type_ in datatypes]):
            raise TypeError("All input columns should be of numeric type.")

        return

    def _fit(self, dataset: SparkFrame) -> GroupImputerModel:
        """ Generates the lookup table and returns the transfromer. 
        
            Args
                dataset: spark.sql.DataFrame
                    Dataset to use for parameter estimation
            
            Outputs
                _GroupImputerModel: Trained transformer.
        """

        # Run input and aprameter checks
        groupCol = self.getGroupCol()

        if self.isSet("inputCol"): inputCols = (self.getInputCol(), )
        else                     : inputCols = self.getInputCols()

        self._checkParams()
        self._checkInput(inputCols, groupCol, dataset)

        # Prepare lookup table and train model
        lookup  = self._makeLookup(inputCols, groupCol, dataset)

        model   = GroupImputerModel(
            inputCol    = self.getInputCol(),
            outputCol   = self.getOutputCol(),
            inputCols   = self.getInputCols(),
            outputCols  = self.getOutputCols(),
            groupCol    = self.getGroupCol(),
            stochastic  = self.getStochastic(),
            lookupTable = lookup
        )

        return model


if __name__ == "__main__":

    from pyspark.ml import Pipeline

    df = spark.createDataFrame(
        schema = ["groups", "y", "y2"],
        data   = [
            ('1.0', 3.0, 4.0),
            ('1.0', 4.0, 5.0),
            ('1.0', 5.0, 6.0),
            ('1.0', None, None),
            ('2.0', 1.0, 2.0),
            ('2.0', None, None),
            ('3.0', None, None)
        ]
    )

    imputer = GroupImputer(
        inputCols   = ["y", "y2"], 
        outputCols  = ["y_imputed", "y2_imputed"], 
        groupCol   = "groups",
        stochastic = True
        )

    pipeline = Pipeline(stages=[imputer])
    model    = pipeline.fit(df)
    results  = model.transform(df)
    results.show()
