""" Functions used in the feature extraction process from the 
    latitude and longitude columns. Used in the feature_extraction.ipynb
"""

import pandas as pd
import geopandas as gpd
import rasterio
import cenpy

from typing import Union, List
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries


def getElevation(series: GeoSeries, demFile: str) -> pd.DataFrame:
    """ 
        Extracts elevation values from a Digital Elevation Model (DEM) raster file
        at specified house locations.

        Args
        series: GeoSeries
                Series object consisting of geopandas point objects (the "geometry" column).

        demFile: string 
            The file path to the DEM raster file.

        Outputs
            pd.DataFrame:
            A DataFrame containing the extracted elevation values
            corresponding to the input house locations. The DataFrame has 
             a single column 'elevation', containing the extracted elevation values,
             and the same index as the input locations geoseries.
            
        NOTE
            The output may contain a different number of rows if the input 
            geoseries contains nans: Rows containing nans (EMPTY POINTs) are
            dropped prior to extraction
    """

    # Read file
    dem = rasterio.open(demFile)

    # Create table with XY coordinates of house locations
    xy = pd.DataFrame(
        {"X": series.x, "Y": series.y}
        ).dropna(how = "any")

    # Save results as a DataFrame
    elevation = pd.DataFrame(
        data    =  dem.sample(xy.to_records(index = False)),
        columns = ["elevation"],
        index   = xy.index,
    )

    return elevation


def nearPOIs(properties: GeoSeries, pois: GeoDataFrame, tags: dict, epsg: int = None) -> GeoDataFrame:
    """ Evaluates the closest points-of-interest (POIs) per property.

        Args
           properties: GeoSeries
                Series object consisting of geopandas point objects corresponding to the properties
                for which the points-of-interest (POIs) will be counted.

            pois: GeoDataFrame
                Geodatafrmae object with indicator (boolean) fields for each POI type, and a single
                active geometry column 


            tags: dict
                Dictionary of tags for the features that have been requested by openstreetmaps.
                See [1] for details.

            epsg: int
                EPSG code specifying output projection. If None, no projection is performed.
            
        Outputs
            geopandas.geodataframe.GeoSeries:
                Geoseries containing the nearest OSM feature per property, for each feature type.

        References
        [1] https://wiki.openstreetmap.org/wiki/Map_features
    """

    # Convert to dataframe to enable the spatial join operation.
    # It contains a (single) geometry dtype column.
    properties = gpd.GeoDataFrame(properties).set_geometry(properties)

    types, poiData = [], []

    # Loop over all POI types requested
    for _, poiTypes in tags.items():
        
        for poiType in poiTypes:

            try:
                # It is not guaranteed that all poiTypes (cafe, bar, morway, etc)
                # have been returned by osmnx. If the current type is not found
                # in the dataframe
                poisSubset = pois[[poiType, "geometry"]].dropna(how = "any")

            except KeyError:
                # Poitype not found. Move to the next type directly
                continue
            
            else:
                # Get the smallest distance of this specific poi type to each
                # property
                nearPOIs = getNearest(
                    from_ = poisSubset, 
                    to_   = properties, 
                    ties  = "distance", 
                    epsg  = epsg,
                    ascending = True
                    )
                
                poiData.append(nearPOIs[["distance"]])
                types.append(poiType)

    # Merge all
    nearestPOIs = pd.concat(poiData, axis = 1)
    nearestPOIs.columns = types

    return nearestPOIs


def countPOIs(properties: GeoSeries, pois: GeoDataFrame, radius: int, epsg: int = None) -> GeoDataFrame:
    """ Evaluates the number of points-of-interest (POIs) per property that lie within 
        a given radius away from the property.

        Args
           properties: GeoSeries
                Series object consisting of geopandas point objects corresponding to the properties
                for which the points-of-interest (POIs) will be counted.

            pois: GeoDataFrame
                Geodatafrmae object with indicator (boolean) fields for each POI type, and a single
                active geometry column 


            radius: int
                Radius around each property to search for POIs

            epsg: int
                EPSG code specifying output projection. If None, no projection is performed.
            
        Outputs
            geopandas.geodataframe.GeoSeries:
                Geoseries containing OSM feature counts per property.

    """

    # Get the active geometry column from the POIs dataframe...
    geomCol = list(pois.select_dtypes('geometry'))[0] 

    # ... and collect all the other columns in a list
    poiCols = [c for c in pois.columns if c != geomCol]

    # Get the name of the properties series (it is not retained after the 
    # buffering operation below)
    propertiesName = properties.name

    # Project if needed
    if epsg is not None:
        properties = properties.to_crs(epsg)
        pois        = pois.to_crs(epsg)

    # Make a radius [meters] around each property
    properties = properties.buffer(radius)
    properties.name = propertiesName # Set name again

    # Convert to dataframe to enable the spatial join operation
    properties = gpd.GeoDataFrame(properties).set_geometry(properties)

    # Join by appending attributes from the right table to left one.
    # This contains a row for every pair of POI and house that are linked.
    # If no POIs are associated with a property, the corresponding row is filled
    # with nans.
    joined = gpd.sjoin(pois, properties, predicate = "within", how = "right")

    # Compute counts:
    # Fill nans with boolean 'False' to indicate the absence of POIs for a property,
    # and sum the occurence of 'True' bools per POI type after grouping per property
    # (=the index of the joined columns)
    counts = joined[poiCols].fillna(False).groupby(joined.index).sum()

    return counts


def getCensus(
    year     : int, 
    msa      : Union[List[str], str], 
    variable : Union[List[str], str]
    ):
    """
    Retrieves census data for a specific year, metropolitan statistical area (MSA),
    and variable(s).

    Args
        year: int
            The year of the ACS dataset.
        msa: list or string 
            Either a list of MSAs or a single MSA as a string.
        variable: list or string 
            Either a list of variables or a single variable as a string.
        density: bool, optional
            If True, convert variables to densities (per unit of area). Default is True.

    Outputs
        pd.DataFrame: DataFrame containing the retrieved census data.

    Raises
        TypeError: If the provided 'msa' parameter is not of type list or str.

    NOTE: The cenpy package used here requires internet connectivity.
    """

    # initialize a connection to the 2017 1-year ACS dataset
    con = cenpy.products.ACS(year)
    # To see available variables simply run: print(con.variables)

    # Run the query to get the data
    if isinstance(msa, str):
        census = con.from_msa(msa, variable)

    elif isinstance(msa, list):
        censusList = [con.from_msa(m, variable) for m in msa]
        census     = pd.concat(censusList, axis = 0)

    else:
        raise TypeError("msa should be of type list or str")

    return census


def map(data, to):
    """
    Spatially joins a GeoDataFrame to a GeoSeries using a common coordinate system.

    Args
        data (geopandas.GeoDataFrame)
            The GeoDataFrame to be spatially joined.
        
        to (geopandas.GeoSeries)
            The GeoSeries providing the geometry for the spatial join.

    Outputs
        geopandas.GeoDataFrame: A new GeoDataFrame resulting from the spatial join.

    """

    # Convert the series to a dataframe 
    # This allows to use the spatial join operation further below.
    projector = gpd.GeoDataFrame(
        geometry = to.values, 
        index    = to.index
        )

    # Project to the same reference coordinate system
    dataProjected  = data.to_crs(projector.crs)

    # Join
    joined = gpd.sjoin(projector, dataProjected, how = "left")

    # (Implicitly) remove the index of the left dataframe joined above
    # that now appears as an additional field.
    joined = joined[dataProjected.columns]

    return joined


def getNearest(
    from_: GeoDataFrame, to_: GeoDataFrame, ties: str, ascending: bool, epsg: int = None
    ) -> GeoDataFrame:
    """
    Perform a spatial join between two GeoDataFrames and break ties based on a specified column.

    Args
        from_: GeoDataFrame
            GeoDataFrame containing source geometries.
    
        to_: GeoDataFrame
            GeoDataFrame containing target geometries.
    
        ties: str 
            The column name used to break ties in case of multiple nearest geometries.
        
        ascending: bool
            If True, sort ties in ascending order; if False, sort in descending order.
    
        epsg: int, optional 
            EPSG code for coordinate system transformation. Default is None.

    Outputs
        GeoDataFrame: 
            A new GeoDataFrame containing the nearest geometry from 'from_'
            with the corresponding distance and broken ties.

    """
       
    # Check if EPSG code is provided for coordinate system transformation
    if epsg is not None:
        from_ = from_.to_crs(epsg)
        to_   = to_.to_crs(epsg)

    # Perform the join
    nearest = gpd.sjoin_nearest(to_, from_, distance_col = "distance", how = "left")
    
    # Break ties: 
    # The spatial nearest join will retain multiple results per property
    # if more than one have the same distance. In this case, another column
    # will be used to order the resulting GeoDataframe, and retain either the
    # first or the last result
    nearest = (
        nearest
        .reset_index()
        .sort_values(ties, ascending = ascending)
        .drop_duplicates(subset = nearest.index.name)
        .set_index(nearest.index.name)
    )

    return nearest



""" 
The following features are computed according to [1]. 

They can be computed from a spark dataframe
containing 'latitude' and 'longitude' fields, as follows

```python

df = (
    df
    .withColumn(
        "coordinatesFrom",
        F.struct( 
            F.col('latitudeFrom'), 
            F.col('longitudeFrom')
            )
        )
    .withColumn(
        "coordinatesTo", 
        F.struct( 
            F.col('latitudeTo'), 
            F.col('longitudeTo')
            )
        )
    .select(["coordinatesFrom", "coordinatesTo"])
    )

(
    ddf
    .select(
        haversine( F.col("coordinatesFrom"), F.col("coordinatesTo") ).alias("haversine_distance"),
        manhattan( F.col("coordinatesFrom"), F.col("coordinatesTo") ).alias("manhattan_distance"),
        geohash( F.col("coordinatesFrom"), F.lit(5) ).alias("geohash"),
        bearingDegree( F.col("coordinatesFrom"), F.col("coordinatesTo") ).alias("bearing_degree")
    )
).show(4)

```

References
    [1] https://bmanikan.medium.com/feature-engineering-all-i-learned-about-geo-spatial-features-649871d16796

    
"""

import pandas as pd
import pygeohash as pgh

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.functions import pandas_udf
from pyspark.sql import Column

RADIUS = 6371  # Earth's radius [km]


@pandas_udf(StringType())
def geohash(coordinates: pd.Series, precision: pd.Series) -> pd.Series:
    """ Extracts geohash from latitude and longitude.

        Args
            coordinates: pandas Series
                Struct column with two fields, "latitude" and "longitude".
            
            precision: pandas Series
                Integer column containing the precision to be used when 
                extracting the geohash.

        Outputs
            geohash: pandas Series
                String column with the extracted geohash.
    """

    # Convert lat/lon to floats. The zillow dataset contains
    # lat/lon as integers that need to be multiplied by 1e-6
    d = {
        'lat': coordinates["latitude"], 
        'lon': coordinates["longitude"], 
        'precision': precision
    }

    # Function to compute the geohash over a row of a dataframe
    # containing lat/lon fields
    f = lambda row: pgh.encode(row['lat'], row['lon'], precision = row['precision'])

    # Compute geohash
    geohash = pd.DataFrame(d).apply(f, axis = 1)

    return geohash


def haversine(coordsFrom: Column, coordsTo: Column) -> Column:
    """ Evaluates the Haversine (great-circle) distance between two pairs of 
        latitude /  longitude coordinates.

        Args
            coordsFrom: pyspark.sql.Column
                Struct column with two fields, "latitude" and "longitude"

            coordsFrom: pyspark.sql.Column
                Struct column with two fields, "latitude" and "longitude"

        Outputs
            dist: pyspark.sql.Column
                Long column containing the Haversine distance [kilometres].
    """

    latFrom = F.radians(coordsFrom["latitude"])
    lngFrom = F.radians(coordsFrom["longitude"])
    latTo   = F.radians(coordsTo["latitude"])
    lngTo   = F.radians(coordsTo["longitude"])

    latDiff = latTo - latFrom
    lngDiff = lngTo - lngFrom

    dist    = 2 * RADIUS * F.asin(
        F.sqrt(
            F.sin(latDiff / 2) ** 2 + 
            F.cos(latFrom) * F.cos(latTo) * F.sin(lngDiff / 2) ** 2
            )
        )
    
    return dist


def manhattan(coordsFrom: Column, coordsTo: Column) -> Column:
    """ Evaluates the Manhattan (square block) distance between two pairs of 
        latitude / longitude coordinates.

        Args
            coordsFrom: pyspark.sql.Column
                Struct column with two fields, "latitude" and "longitude"

            coordsFrom: pyspark.sql.Column
                Struct column with two fields, "latitude" and "longitude"

        Outputs
            dist: pyspark.sql.Column
                Long column containing the Manhattan distance [kilometres].
    """

    coordsA = F.struct(coordsFrom["latitude"], coordsTo["longitude"]  )
    coordsB = F.struct(coordsTo["latitude"]  , coordsFrom["longitude"])
    dist    = haversine(coordsFrom, coordsA) + haversine(coordsFrom, coordsB)

    return dist


def bearingDegree(coordsFrom: Column, coordsTo: Column) -> Column:
    """ Evaluates the bearing (primary compass direction) between two pairs of
        latitude / longitude coordinates.

        Args
            coordsFrom: pyspark.sql.Column
                Struct column with two fields, "latitude" and "longitude"

            coordsFrom: pyspark.sql.Column
                Struct column with two fields, "latitude" and "longitude"

        Outputs
            dist: pyspark.sql.Column
                Long column containing the bearing angle [deg].
    """

    latFrom = F.radians(coordsFrom["latitude"])
    latTo   = F.radians(coordsTo["latitude"])
    diffLng = F.radians(coordsTo["longitude"] - coordsFrom["longitude"])

    y       = F.sin(diffLng) * F.cos(latTo)
    x       = F.cos(latFrom) * F.sin(latTo) - F.sin(latFrom) * F.cos(latTo) * F.cos(diffLng)
    
    bearing = F.degrees(F.atan2(y, x))

    return bearing
