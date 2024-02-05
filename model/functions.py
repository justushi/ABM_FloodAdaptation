# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

Functions that are used in the model_file.py and agent.py for the running of the Flood Adaptation Model.
Functions get called by the Model and Agent class.
"""
import random
import numpy as np
import math
from shapely import contains_xy
from shapely import prepare
import geopandas as gpd

def set_initial_values(input_data, parameter, seed):
    """
    Function to set the values based on the distribution shown in the input data for each parameter.
    The input data contains which percentage of households has a certain initial value.

    Parameters
    ----------
    input_data: the dataframe containing the distribution of paramters
    parameter: parameter name that is to be set
    seed: agent's seed

    Returns
    -------
    parameter_set: the value that is set for a certain agent for the specified parameter
    """
    parameter_set = 0
    parameter_data = input_data.loc[(input_data.parameter == parameter)] # get the distribution of values for the specified parameter
    parameter_data = parameter_data.reset_index()
    random.seed(seed)
    random_parameter = random.randint(0,100)
    for i in range(len(parameter_data)):
        if i == 0:
            if random_parameter < parameter_data['value_for_input'][i]:
                parameter_set = parameter_data['value'][i]
                break
        else:
            if (random_parameter >= parameter_data['value_for_input'][i-1]) and (random_parameter <= parameter_data['value_for_input'][i]):
                parameter_set = parameter_data['value'][i]
                break
            else:
                continue
    return parameter_set


def get_flood_map_data(flood_map):
    """
    Getting the flood map characteristics.

    Parameters
    ----------
    flood_map: flood map in tif format

    Returns
    -------
    band, bound_l, bound_r, bound_t, bound_b: characteristics of the tif-file
    """
    band = flood_map.read(1)
    bound_l = flood_map.bounds.left
    bound_r = flood_map.bounds.right
    bound_t = flood_map.bounds.top
    bound_b = flood_map.bounds.bottom
    return band, bound_l, bound_r, bound_t, bound_b

shapefile_path = r'../input_data/model_domain/houston_model/houston_model.shp'
floodplain_path = r'../input_data/floodplain/floodplain_area.shp'

# Model area setup
map_domain_gdf = gpd.GeoDataFrame.from_file(shapefile_path)
map_domain_gdf = map_domain_gdf.to_crs(epsg=26915)
map_domain_geoseries = map_domain_gdf['geometry']
map_minx, map_miny, map_maxx, map_maxy = map_domain_geoseries.total_bounds
map_domain_polygon = map_domain_geoseries[0]  # The geoseries contains only one polygon
prepare(map_domain_polygon)

# Floodplain setup
floodplain_gdf = gpd.GeoDataFrame.from_file(floodplain_path)
floodplain_gdf = floodplain_gdf.to_crs(epsg=26915)
floodplain_geoseries = floodplain_gdf['geometry']
floodplain_multipolygon = floodplain_geoseries[0]  # The geoseries contains only one multipolygon
prepare(floodplain_multipolygon)

def generate_random_location_within_map_domain():
    """
    Generate random location coordinates within the map domain polygon.

    Returns
    -------
    x, y: lists of location coordinates, longitude and latitude
    """
    while True:
        # generate random location coordinates within square area of map domain
        x = random.uniform(map_minx, map_maxx)
        y = random.uniform(map_miny, map_maxy)
        # check if the point is within the polygon, if so, return the coordinates
        if contains_xy(map_domain_polygon, x, y):
            return x, y

def get_flood_depth(corresponding_map, location, band):
    """
    To get the flood depth of a specific location within the model domain.
    Households are placed randomly on the map, so the distribution does not follow reality.

    Parameters
    ----------
    corresponding_map: flood map used
    location: household location (a Shapely Point) on the map
    band: band from the flood map

    Returns
    -------
    depth: flood depth at the given location
    """
    row, col = corresponding_map.index(location.x, location.y)
    depth = band[row -1, col -1]
    return depth


def get_position_flood(bound_l, bound_r, bound_t, bound_b, img, seed):
    """
    To generater the position on flood map for a household.
    Households are placed randomly on the map, so the distribution does not follow reality.

    Parameters
    ----------
    bound_l, bound_r, bound_t, bound_b, img: characteristics of the flood map data (.tif file)
    seed: seed to generate the location on the map

    Returns
    -------
    x, y: location on the map
    row, col: location within the tif-file
    """
    random.seed(seed)
    x = random.randint(round(bound_l, 0), round(bound_r, 0))
    y = random.randint(round(bound_b, 0), round(bound_t, 0))
    row, col = img.index(x, y)
    return x, y, row, col


# ! Function was adapted to include adaptation measure
def calculate_basic_flood_damage(flood_depth, is_adapted, mitigation_effectiveness):
    """
    Calculate the flood damage based on the flood depth of a household.

    The calculation is based on the logarithmic regression model proposed by de Moer and Huizinga (2017).
    If the flood depth is greater than or equal to 6m, the damage factor is 1.
    If the flood depth is less than 0.025m, the damage factor is 0.
    For other flood depths, the damage factor is calculated using the logarithmic regression equation.

    Parameters
    ----------
    flood_depth : float
        The flood depth as given by the location within the model domain.
    is_adapted : bool
        A flag indicating whether the agent is adapted to flooding.
    mitigation_effectiveness : float
        The effectiveness of the mitigation measures implemented by the agent.

    Returns
    -------
    flood_damage : float
        The damage factor between 0 and 1.
    """

    # * Reducing flood damage if agent is adapted
    if is_adapted:
       flood_depth = flood_depth - mitigation_effectiveness

    if flood_depth >= 6:
        flood_damage = 1
    elif flood_depth < 0.025:
        flood_damage = 0
    else:
        # see flood_damage.xlsx for function generation
        flood_damage = 0.1746 * math.log(flood_depth) + 0.6483

    return flood_damage



# ! Probability of failure function
def calculate_probability_of_failure(flood_map_choice):
    """
    Calculates the probability of failure based on the chosen flood map.

    Parameters
    ----------
    flood_map_choice : str
        The chosen flood map ("harvey", "100yr", or "500yr").

    Returns
    -------
    probability_of_failure : float
        The calculated probability of failure.
    """

    probability_of_failure = model.flood_probability

    # flood_map_choice = model.flood_map_choice

    # ALTERNATIVE CODE TO USE THE FLOOD MAP CHOICE
    # if flood_map_choice == "harvey":
    #     probability_of_failure = 1 / 50
    # elif flood_map_choice == "100yr":
    #     probability_of_failure = 1 / 100
    # elif flood_map_choice == "500yr":
    #     probability_of_failure = 1 / 500
    # else:
    #     raise ValueError("Invalid flood map choice")

    return probability_of_failure


# ! Probability of dying function
def calculate_probability_of_dying(flood_depth_estimated):
    """
    Calculate the probability of death based on Boyd et al. (2005) mortality function.

    This function estimates the probability of death based on the water depth using Boyd et al.'s mortality function.
    The function takes the estimated flood depth as input and returns the probability of death.

    Reference:
    - Boyd, M. J., Penning-Rowsell, E. C., & Tapsell, S. M. (2005). Further specification of the dose-relationship for flood fatality estimation.
        Natural Hazards, 36(1-2), 147-161. doi:10.1007/s11069-008-9227-5

    Parameters:
    - flood_depth_estimated (float): The estimated water depth in meters.

    Returns:
    - death_probability (float): The probability of death as a decimal value between 0 and 1.
    """

    # Calculate the water depth
    depth = flood_depth_estimated

    # Constants from Boyd's mortality function
    a = 0.34
    b = 20.37
    c = 6.18

    # Calculate the mortality function
    probability_of_dying = a / (1 + math.exp((b - c * depth)))

    return probability_of_dying


# ! The individual risk function
def calculate_individual_risk(probability_of_failure, probability_of_dying, low_risk_threshold, medium_risk_threshold):
    """
    Calculates the individual risk for each agent.

    Parameters
    ----------
    probability_of_failure : float
        The likelihood of flood or failure.
    probability_of_dying : float
        The probability of dying, which depends on the severity of the flood.
    low_risk_threshold : float
        The threshold below which the individual risk is considered low.
    medium_risk_threshold : float
        The threshold below which the individual risk is considered medium.

    Returns
    -------
    risk_group : str
        The risk group to which the individual belongs. Can be "low", "medium", or "high".
    """

    # Calculate the individual risk
    individual_risk = probability_of_failure * probability_of_dying

    # Determine the risk group
    if individual_risk <= low_risk_threshold:
        risk_group = "low" # acceptably low risk
    elif individual_risk < medium_risk_threshold:
        risk_group = "medium" # medium risk
    else:
        risk_group = "high" # unaccaptable high risk

    return risk_group


# ! Function to calculate height of subsidy based on individual risk, income, and risk group
def calculate_subsidy_height(risk_group, income_group, low_risk_subsidy_height, medium_risk_subsidy_height, high_risk_subsidy_height, low_income_subsidy_factor, medium_income_subsidy_factor, high_income_subsidy_factor):
    """
    Calculates the height of the subsidy based on the individual risk, income, and risk group.

    Parameters
    ----------
    risk_group : str
        The risk group to which the individual belongs. Can be "low", "medium", or "high".
    income_group : str
        The income group to which the individual belongs. Can be "low", "medium", or "high".
    low_risk_subsidy_height : float
        The subsidy height for the low risk group.
    medium_risk_subsidy_height : float
        The subsidy height for the medium risk group.
    high_risk_subsidy_height : float
        The subsidy height for the high risk group.
    low_income_subsidy_factor : float
        The subsidy deduction factor for the low income group.
    medium_income_subsidy_factor : float
        The subsidy deduction factor for the medium income group.
    high_income_subsidy_factor : float
        The subsidy deduction factor for the high income group.

    Returns
    -------
    subsidy_height : float
        The calculated height of the subsidy that the agent is eligible for based on their individual risk, income, and risk group.
    """

    # Calculate actual height of subsidy the agent is eligible for
    if income_group == "low":
        if risk_group == "low":
            subsidy_height = low_risk_subsidy_height * low_income_subsidy_factor
        elif risk_group == "medium":
            subsidy_height = medium_risk_subsidy_height * low_income_subsidy_factor
        else:
            subsidy_height = high_risk_subsidy_height * low_income_subsidy_factor
    elif income_group == "medium":
        if risk_group == "low":
            subsidy_height = low_risk_subsidy_height * medium_income_subsidy_factor
        elif risk_group == "medium":
            subsidy_height = medium_risk_subsidy_height * medium_income_subsidy_factor
        else:
            subsidy_height = high_risk_subsidy_height * medium_income_subsidy_factor
    elif income_group == "high":
        if risk_group == "low":
            subsidy_height = low_risk_subsidy_height * high_income_subsidy_factor
        elif risk_group == "medium":
            subsidy_height = medium_risk_subsidy_height * high_income_subsidy_factor
        else:
            subsidy_height = high_risk_subsidy_height * high_income_subsidy_factor
    else:
        raise ValueError("Invalid income or risk group")

    return subsidy_height

