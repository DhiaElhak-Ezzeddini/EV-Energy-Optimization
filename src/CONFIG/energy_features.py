"""
Energy Features Configuration
==============================
Global configuration for energy consumption features used in EV energy optimization.
This module defines all the feature variables required for calculating total energy consumed.
"""

# ============================================================================
# VEHICLE FEATURES
# ============================================================================

VEHICLE_ID = 'Vehicle_ID'
"""str: Unique identifier for each vehicle"""

SPEED_KMH = 'Speed_kmh'
"""str: Vehicle speed in kilometers per hour"""

ACCELERATION_MS2 = 'Acceleration_ms2'
"""str: Vehicle acceleration in meters per second squared"""

VEHICLE_WEIGHT_KG = 'Vehicle_Weight_kg'
"""str: Total weight of the vehicle in kilograms"""

DISTANCE_TRAVELLED_KM = 'Distance_Travelled_km'
"""str: Distance travelled by the vehicle in kilometers"""


# ============================================================================
# BATTERY FEATURES
# ============================================================================

BATTERY_STATE_PERCENT = 'Battery_State_%'
"""str: State of charge of the battery as a percentage"""

BATTERY_VOLTAGE_V = 'Battery_Voltage_V'
"""str: Battery voltage in volts"""

BATTERY_TEMPERATURE_C = 'Battery_Temperature_C'
"""str: Battery temperature in degrees Celsius"""


# ============================================================================
# DRIVING CONDITION FEATURES
# ============================================================================

DRIVING_MODE = 'Driving_Mode'
"""str: Current driving mode (e.g., Eco, Normal, Sport)"""

ROAD_TYPE = 'Road_Type'
"""str: Type of road (e.g., Highway, Urban, Rural)"""

TRAFFIC_CONDITION = 'Traffic_Condition'
"""str: Current traffic conditions (e.g., Light, Moderate, Heavy)"""

SLOPE_PERCENT = 'Slope_%'
"""str: Road slope/grade as a percentage"""


# ============================================================================
# ENVIRONMENTAL FEATURES
# ============================================================================

WEATHER_CONDITION = 'Weather_Condition'
"""str: Current weather conditions (e.g., Clear, Rain, Snow)"""

TEMPERATURE_C = 'Temperature_C'
"""str: Ambient temperature in degrees Celsius"""

HUMIDITY_PERCENT = 'Humidity_%'
"""str: Relative humidity as a percentage"""

WIND_SPEED_MS = 'Wind_Speed_ms'
"""str: Wind speed in meters per second"""


# ============================================================================
# VEHICLE COMPONENT FEATURES
# ============================================================================

TIRE_PRESSURE_PSI = 'Tire_Pressure_psi'
"""str: Tire pressure in pounds per square inch"""


# ============================================================================
# TARGET VARIABLE
# ============================================================================

ENERGY_CONSUMPTION_KWH = 'Energy_Consumption_kWh'
"""str: Total energy consumption in kilowatt-hours (target variable)"""


# ============================================================================
# FEATURE GROUPS
# ============================================================================

VEHICLE_FEATURES = [
    VEHICLE_ID,
    SPEED_KMH,
    ACCELERATION_MS2,
    VEHICLE_WEIGHT_KG,
    DISTANCE_TRAVELLED_KM
]
"""list: All vehicle-related features"""

BATTERY_FEATURES = [
    BATTERY_STATE_PERCENT,
    BATTERY_VOLTAGE_V,
    BATTERY_TEMPERATURE_C
]
"""list: All battery-related features"""

DRIVING_FEATURES = [
    DRIVING_MODE,
    ROAD_TYPE,
    TRAFFIC_CONDITION,
    SLOPE_PERCENT
]
"""list: All driving condition features"""

ENVIRONMENTAL_FEATURES = [
    WEATHER_CONDITION,
    TEMPERATURE_C,
    HUMIDITY_PERCENT,
    WIND_SPEED_MS
]
"""list: All environmental features"""

COMPONENT_FEATURES = [
    TIRE_PRESSURE_PSI
]
"""list: All vehicle component features"""


# ============================================================================
# COMPLETE FEATURE SETS
# ============================================================================

ALL_INPUT_FEATURES = (
    VEHICLE_FEATURES +
    BATTERY_FEATURES +
    DRIVING_FEATURES +
    ENVIRONMENTAL_FEATURES +
    COMPONENT_FEATURES
)
"""list: All input features for energy consumption prediction"""

ALL_FEATURES = ALL_INPUT_FEATURES + [ENERGY_CONSUMPTION_KWH]
"""list: All features including the target variable"""


# ============================================================================
# FEATURE CATEGORIES BY TYPE
# ============================================================================

NUMERICAL_FEATURES = [
    SPEED_KMH,
    ACCELERATION_MS2,
    BATTERY_STATE_PERCENT,
    BATTERY_VOLTAGE_V,
    BATTERY_TEMPERATURE_C,
    SLOPE_PERCENT,
    TEMPERATURE_C,
    HUMIDITY_PERCENT,
    WIND_SPEED_MS,
    TIRE_PRESSURE_PSI,
    VEHICLE_WEIGHT_KG,
    DISTANCE_TRAVELLED_KM,
    ENERGY_CONSUMPTION_KWH
]
"""list: All numerical features"""

CATEGORICAL_FEATURES = [
    DRIVING_MODE,
    ROAD_TYPE,
    TRAFFIC_CONDITION,
    WEATHER_CONDITION
]
"""list: All categorical features"""

IDENTIFIER_FEATURES = [
    VEHICLE_ID
]
"""list: Identifier features (not used for modeling)"""


# ============================================================================
# FEATURE UNITS
# ============================================================================

FEATURE_UNITS = {
    VEHICLE_ID: 'identifier',
    SPEED_KMH: 'km/h',
    ACCELERATION_MS2: 'm/s²',
    BATTERY_STATE_PERCENT: '%',
    BATTERY_VOLTAGE_V: 'V',
    BATTERY_TEMPERATURE_C: '°C',
    DRIVING_MODE: 'categorical',
    ROAD_TYPE: 'categorical',
    TRAFFIC_CONDITION: 'categorical',
    SLOPE_PERCENT: '%',
    WEATHER_CONDITION: 'categorical',
    TEMPERATURE_C: '°C',
    HUMIDITY_PERCENT: '%',
    WIND_SPEED_MS: 'm/s',
    TIRE_PRESSURE_PSI: 'psi',
    VEHICLE_WEIGHT_KG: 'kg',
    DISTANCE_TRAVELLED_KM: 'km',
    ENERGY_CONSUMPTION_KWH: 'kWh'
}
"""dict: Units for each feature"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_by_category(category: str) -> list:
    """
    Get features by category.
    
    Args:
        category: Category name ('vehicle', 'battery', 'driving', 
                  'environmental', 'component', 'numerical', 'categorical')
    
    Returns:
        list: List of features in the specified category
    """
    category_map = {
        'vehicle': VEHICLE_FEATURES,
        'battery': BATTERY_FEATURES,
        'driving': DRIVING_FEATURES,
        'environmental': ENVIRONMENTAL_FEATURES,
        'component': COMPONENT_FEATURES,
        'numerical': NUMERICAL_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'identifier': IDENTIFIER_FEATURES
    }
    return category_map.get(category.lower(), [])


def get_feature_unit(feature: str) -> str:
    """
    Get the unit for a specific feature.
    
    Args:
        feature: Feature name
    
    Returns:
        str: Unit of the feature
    """
    return FEATURE_UNITS.get(feature, 'unknown')


def validate_features(feature_list: list) -> bool:
    """
    Validate if all features in the list are recognized.
    
    Args:
        feature_list: List of feature names to validate
    
    Returns:
        bool: True if all features are valid, False otherwise
    """
    return all(feature in ALL_FEATURES for feature in feature_list)


