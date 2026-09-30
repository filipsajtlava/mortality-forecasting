DATA_DIRECTORY_NAME = "hmd_data"
STALE_FALLBACK_COUNTRY_AVAILABLE = "supported_countries_fallback.json"
CACHED_COUNTRY_AVAILABLE = "countries_available.json"
CACHED_TTL_COUNTRY_CODE_DAYS = 360
CACHED_TTL_COUNTRY_DATA_DAYS = 360

FILE_SELECTION_COUNTRY_DATA = {
    "M": "Mx_1x1.txt",
    "E": "Exposures_1x1.txt",
    "D": "Deaths_1x1.txt"
}
REQUIRED_DATA_FOR_MODELS = ["E", "D"]

VALUE_COLUMNS = ["Female", "Male", "Total"]
MAXIMUM_LC_ITERATIONS = 300

LAMBDA_INITIALIZATION = 1.0

AGE_DIM = "Age"
YEAR_DIM = "Year"
SIMULATION_DIM = "Simulation"
BOUND_DIM = "Bound"

PLOTTING_LABELS = {
    AGE_DIM: "x",
    YEAR_DIM: "t"
}

INFO_INDENT = " " * 5