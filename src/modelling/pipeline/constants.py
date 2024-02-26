""" Defines constants used in the modelling pipeline. """

# Appends a string to each element of a list. Used to generate the output column names
# fro m the various transformers iused in the pipeline.
addSuffix  = lambda string_, list_: [value + string_ for value in list_]   

""" Preprocessing constants"""
# Dictionary indicating how to fill null values per column.
# Columns not appearing here will be imputed within the pipeline
NA_DICT = {
    "poolsizesum"           : 0,
    "yardbuildingsqft17"    : 0,
    "garagetotalsqft"       : 0,
    "garagecarcnt"          : 0,
    "lotsizesquarefeet"     : 0,
    "numberofstories"       : 0,
    "poolcnt"               : 0,
    "pooltypeid7"           : 0,
    "fips"                  : 0,
    "decktypeid"            : 0,
    "fireplacecnt"          : 0,
    "heatingorsystemtypeid" : 0,
    "propertylandusetypeid" : 0,
    "buildingqualitytypeid" : 0,
    "regionidcounty"        : 0,
    "regionidneighborhood"  : 0,
    "regionidcity"          : 0,
    "propertyzoningdesc"    : 0,
    "threequarterbathnbr"   : 0,
    "unitcnt"               : 0,
    "airconditioningtypeid" : 0,
    "assessmentyear"        : 2015,
    "taxdelinquencyflag"    : "N",
    "propertycountylandusecode": "UNK",
    "propertyzoningdesc"    : "OTHER",
    "censustractandblock"   : "UNK",
    "name_close_city"       : "UNK",
    "name_close_big_city"   : "UNK"
}

""" Pipeline constants """
# Column that will be predicted
TARGET_COL = "logerror"
PREDICTION_COL = "prediction"

# Columns that will be transformed using square root (x^1/2)
SQRT_COLS  = ["bar", "restaurant", "school", "aerodrome", "monument"]

# Columns that will be transformed using power transformation (x^1/3)
POWER_COLS = ["fast_food", "parking", "station"]

# Columns that will be log-transformed
LOG_COLS   = [
    "calculatedfinishedsquarefeet", "yardbuildingsqft17", "taxvaluedollarcnt", "lotsizesquarefeet", "elevation", "amenity", 
    "B00001_001E", "B19313_001E", "distance_close_city", "distance_close_big_city", "population_close_city", "population_close_big_city"
]

# Columns whose values will be replaced (grouped) in a new value
# Each tuple contains: column name, list of values to be RETAINED, value to replace everything else
VALUE_REPLACEMENT_COLS = [ 
    ("decktypeid", [0], 1),
    ("fireplacecnt", [0], 1),
    ("heatingorsystemtypeid", [2, 7, 0], 3),
    ("airconditioningtypeid", [0, 1, 13], 0),
    ("propertylandusetypeid", [261, 266, 246, 269], 0),
    ("propertyzoningdesc", ["LAR1", "LAR3", "LARS", "LBR1N", "LARD1.5", "LAR2", "SCUR2", "LARD2"], "OTHER"),
    ("propertycountylandusecode", ["0100", "122", "010C", "0101", "34", "1111", "1", "010E", "010D", "0200"], "OTHER")
]

# Columns for which new indicator columns will be generated, indicating if their value lies above
# the given threshold. Each tuple contains the column name and the threshold value
INDICATOR_COLS = [ 
    ("distance_close_city", -2),
    ("distance_close_big_city", -3),
    ("highway", 0),
    ("aeroway", 0),
    ("railway", 0),
    ("historic", 0),
    ("natural", 0)
]

# Columns that will be dropped
DROP_COLS = [
    "finishedsquarefeet13", "finishedsquarefeet6",  "finishedsquarefeet15",     "finishedfloor1squarefeet", 
    "yardbuildingsqft26",   "basementsqft",         "landtaxvaluedollarcnt",    "structuretaxvaluedollarcnt",  
    "buildingclasstypeid",  "fullbathcnt",          "architecturalstyletypeid", "storytypeid", 
    "finishedsquarefeet50", "finishedsquarefeet12", "taxamount",                "parcelid", 
    "pooltypeid10",         "pooltypeid2",          "taxdelinquencyyear",       "typeconstructiontypeid",
    "regionidzip",          "fireplaceflag",        "hashottuborspa",           "transactiondate",   
]


# Column for group-wise stochastic imputation
GROUP_IMPUTER_IN = "garagetotalsqft"
GROUP_IMPUTER_OUT = "garagetotalsqft_imputed"

# Columns for mode imputation
IMPUTER_COLS = [
    'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
    'calculatedfinishedsquarefeet', 'latitude', 'aerodrome',
    'longitude', 'rawcensustractandblock', 'roomcnt', 'yearbuilt',
    'taxvaluedollarcnt', 'elevation', 'B00001_001E', 'B19313_001E',
    'population_close_city', 'population_close_big_city', 'bar',
    'restaurant', 'hospital', 'school', 'parking', 'fast_food', 
    'station', 'monument', 'beach']

# String columns, to be converted to numerical values.  
# The indexer will output the same column names with the given suffix appended.
INDEX_IN = [
    'propertycountylandusecode', 'propertyzoningdesc', 'censustractandblock', 'name_close_city', 'name_close_big_city'
]

INDEX_OUT = addSuffix('_indexed', INDEX_IN)

# Columns that will be one-hot encoded, using the OneHotEncoder transformer of the spark ML lib.
# The encoder will output the same column names with the given suffix appended.
ONEHOT_IN = [
    "airconditioningtypeid", "fips", "heatingorsystemtypeid",      "propertycountylandusecode_indexed", 
    "name_close_big_city_indexed",   "propertyzoningdesc_indexed", "regionidcounty",
]

ONEHOT_OUT = addSuffix('_onehot', ONEHOT_IN)

# Columns that will be hashed.
HASH_OUT = "x_hash"   # The hasher will output a single column with this name
HASH_IN  = [
    "censustractandblock_indexed", "rawcensustractandblock", "regionidneighborhood", "regionidcity", "name_close_city_indexed"
]

ASSEMBLER_OUT = "x_assembler" # The vector asembler will output a single column with this name
ASSEMBLER_IN  = [
    'distance_close_big_city', 'propertylandusetypeid', 'propertyzoningdesc_indexed_onehot',
    'calculatedbathnbr', 'threequarterbathnbr', 'hospital', 'buildingqualitytypeid', 'highway', 
    'station', 'regionidcounty_onehot', 'unitcnt', 'bedroomcnt', 'propertycountylandusecode_indexed_onehot',
    'heatingorsystemtypeid_onehot', 'school', 'latitude', 'aeroway', 'beach', 'amenity', 'taxvaluedollarcnt', 
    'calculatedfinishedsquarefeet', 'numberofstories', 'taxdelinquencyflag', 'population_close_city', 'garagetotalsqft_imputed',
    'B00001_001E', 'transaction_month', 'transaction_day', 'distance_close_city', 'assessmentyear', 'poolsizesum',
    'fireplacecnt', 'railway', 'parking', 'B19313_001E', 'bathroomcnt',  'name_close_big_city_indexed_onehot',
    'restaurant', 'x_hash', 'pooltypeid7', 'bar', 'lotsizesquarefeet', 'population_close_big_city',
    'elevation', 'natural', 'monument', 'transaction_year', 'decktypeid', 'roomcnt', 'airconditioningtypeid_onehot',
    'yearbuilt', 'fast_food', 'poolcnt', 'aerodrome', 'yardbuildingsqft17', 'longitude', 'historic', 'fips_onehot'
]

# NOTE: The following lists were generated automatically by the columnNames() function in preprocessor.py:
# INDEXER_OUT, ENCODER_OUT, ASSEMBLER_IN

""" Training constants """
XGB_CONSTANTS = { 
    "tree_method" : "gpu_hist",
    "grow_policy" : "depthwise",
    "num_workers" : 1,
    "use_gpu"     : True,
    "random_state": 42
}