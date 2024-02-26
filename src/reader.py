""" Set of utility functions. """

from functools import reduce
from pyspark.sql import DataFrame as SparkDataframe
from pyspark.sql import SparkSession


def read(session: SparkSession, config: dict, featureFile: str = None) -> SparkDataframe:
    """
    Reads multiple files based on the provided configuration and returns a 
    merged Spark DataFrame.

    Args
        session: pyspark.sql.SparkSession
            The Spark session used for reading data.
    
        config: dict
            A dictionary containing configuration parameters, including:
            - "files" (dict): A nested dictionary specifying filetypes and their corresponding filenames.
            - "schema" (dict): A dictionary mapping filetypes to their respective column schemas.
            - "format" (str): The file format to use for reading (e.g., "parquet", "csv").
            - "header" (bool): Whether the files have headers.
            - "delimiter" (str): The delimiter used in the files.

        featureFile: string
            Path to a feature file containing a dataframe with additinal features to be 
            merged with the original dataframe.
            NOTE: It is assumed that a colun "parcelid" exists on the file, to be used
                  for merging with the initial dataframe.

    Outputs
        df: pyspark.sql.DataFrame
            Merged Spark DataFrame containing the data from the input files.
    """

    dfList = []
    for _, files in config["files"].items():

        dfListInner = []
        for filetype, filename in files.items():

            # Get schema for this file
            schema = ", ".join(config["schema"][filetype])
            
            # Parse file + append to the inner list
            dfInner = (
                session
                .read
                .format(config["format"])
                .schema(schema)
                .options(
                    header    = config["header"],
                    delimiter = config["delimiter"]
                )
                .load(filename)
            )
            
            dfListInner.append(dfInner)
        
        # Join the dataframes + append to the (outer) list
        df = reduce(
                lambda dfLeft, dfRight: dfLeft.join(dfRight, on = "parcelid", how = "inner"), 
                dfListInner
            )

        dfList.append(df)

    # Merge all dataframes
    df = reduce(lambda dfLeft, dfRight: dfLeft.union(dfRight), dfList)

    # Add external features
    if featureFile is not None:
        
        # Parse file with additional features
        schema = ", ".join(config["schema"]["external_features"])
        extFeats = (
            session
            .read
            .format(config["format"])
            .schema(schema)
            .options(
                header    = config["header"],
                delimiter = config["delimiter"]
            )
            .load(featureFile)
        )

        # Merge
        df = df.join(extFeats, on = "parcelid", how = "left")

    return df
