# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon

# Policy related functions
from functions import calculate_probability_of_dying, calculate_individual_risk, calculate_subsidy_height

# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    #################################### AGENT INITIALISATION ######################################

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Initial adaptation status set to False
        self.is_adapted = False

        # Initiate network related parameters
        self.friends_amount = None
        self.friends_network = None
        self.friends_adapted = False

        # INCOME and SAVINGS____________________________________

        # ! Income parameters
        # Min and max income values
        income_min = model.income_min
        income_max = model.income_max
        low_income_threshold = model.low_income_threshold
        medium_income_threshold = model.medium_income_threshold

        # ! Income distribution
        # Uniform distribution
        self.income_distribution = model.income_distribution
        if self.income_distribution == 'uniform':
            self.income = int(random.uniform(income_min, income_max))
        # Normal distribution
        elif self.income_distribution == 'normal':
            self.income = int(random.normalvariate(income_min, income_max))
        # Normal distribution with peak around low_income_threshold
        elif self.income_distribution == 'normal_low_peak':
            mean_low_peak = low_income_threshold
            self.income = max(income_min, min(int(random.normalvariate(mean_low_peak, income_max)), income_max))
        # Normal distribution with peak around medium_income_threshold
        elif self.income_distribution == 'normal_medium_peak':
            mean_medium_peak = medium_income_threshold
            self.income = max(income_min, min(int(random.normalvariate(mean_medium_peak, income_max)), income_max))
        # If not distribution is specified, use random distribution
        else:
            self.income = int(random.randint(income_min, income_max))

        # ! Saving parameters
        # Initial savings
        self.savings = 0

        # Share of savings taken each quarter from income dependent on income group
        if self.income < low_income_threshold:
            self.savings_share = model.savings_share_low_income
            self.income_group = 'low'
        elif self.income < medium_income_threshold:
            self.savings_share = model.savings_share_medium_income
            self.income_group = 'medium'
        else:
            self.savings_share = model.savings_share_high_income
            self.income_group = 'high'


        # OTHER PARAMETERS________________________________________

        # getting flood map values and get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates.
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0

        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        # ! Changed to take into account if a agent has already adapted by the time the simulation starts
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated, is_adapted=self.is_adapted, mitigation_effectiveness=model.mitigation_effectiveness)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        # To reduce the actual flood damage based on the adaptation measure
        # calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        # ! Checks if agent has adapted to flooding
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual, is_adapted=self.is_adapted, mitigation_effectiveness=model.mitigation_effectiveness)


        # PMT parameters_______________________________________

        # ! THREAT APPRAISAL
        # * Perceived flood probability
        self.flood_probability_percept = model.flood_probability_percept
            # Sets initial perceived flood probability based on the initial flood probability parameter
            # (0 = completely safe, 1 = more frequent than once a year)
            # Influenced by: time passed since flood / flood frequency; evtl. social network

        # * Perceived flood damage
        self.flood_damage_percept = self.flood_damage_estimated
            # Perceived flood damage (0 = not at all severe, 1 = extremely severe)
            # Influenced by: elevation at location / expected flood depth

        # ! COPING APPRAISAL
        # * Perceived effectiveness of protective measures
        self.coping_efficacy = None # 0.5
            # Perceived effectiveness of protective measures (0 = not at all effective, 1 = extremely effective)
            # Influenced by: availability of information

        # * Perceived ability to implement protective measures
        self.self_efficacy = None # 0.5
            # Perceived ability to implement protective measures (0 = not at all able, 1 = extremely able)
            # Influenced by: availability of subsidies, evtl. availability of information, evtl. social network (neighbours have adapted)

        self.income_contribution = None
        self.savings_contribution = None
        self.social_network_contribution = None

        # ! UPDATE INITIAL Threat and Coping appraisal
        # * Update threat appraisal
        # self.threat_appraisal = self.update_threat_appraisal()

        # * Update coping appraisal
        # self.coping_appraisal = self.update_coping_appraisal()


        # POLICY PARAMETERS______________________________________

        # ! Individual risk
        # Individual risk is the probability of failure multiplied with the probability of dying

        # Probability of the flood happening
        self.probability_of_failure = model.flood_probability # ! calculate_probability_of_failure(flood_map_choice=model.flood_map_choice)

        # Probability of dying
        self.probability_of_dying = calculate_probability_of_dying(flood_depth_estimated=self.flood_depth_estimated)
            # Would need to know agent adaptation status if agents could be adapted at the beginning of the simulation
            # --> self.probability_of_dying = calculate_probability_of_dying(flood_depth_estimated=self.flood_depth_estimated, is_adapted=self.is_adapted, mitigation_effectiveness=model.mitigation_effectiveness)

        # Individual risk
        self.individual_risk = calculate_individual_risk(probability_of_failure=self.probability_of_failure, probability_of_dying=self.probability_of_dying, low_risk_threshold=model.low_risk_threshold, medium_risk_threshold=model.medium_risk_threshold)

        # ! Subsidy height
        if model.policy: #  == 'subsidy':
            # Subsidy height is the amount of money the government is willing to pay for adaptation measures
            self.actual_subsidy_height = calculate_subsidy_height(
                risk_group=self.individual_risk, income_group=self.income_group,
                low_risk_subsidy_height=model.low_risk_subsidy_height, medium_risk_subsidy_height=model.medium_risk_subsidy_height,
                high_risk_subsidy_height=model.high_risk_subsidy_height, low_income_subsidy_factor=model.low_income_subsidy_factor,
                medium_income_subsidy_factor=model.medium_income_subsidy_factor, high_income_subsidy_factor=model.high_income_subsidy_factor)
        # elif model.policy == 'no_subsidy':
        #     self.actual_subsidy_height = 0
        else:
            self.actual_subsidy_height = 0
            #raise ValueError('Policy parameter is not specified correctly. Please choose either "subsidy" or "no_policy".')


#################################### AGENT FUNCTIONS ######################################

    # Function to count friends who can be influencial.
    def count_friends(self, radius):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return len(friends)

    # PMT FUNCTIONS________________________________________________

    # ! Threat appraisal
    def update_threat_appraisal(self):
        """Function to update threat appraisal."""

        # Weights for flood probability and flood damage
        flood_probability_percept_weight = self.model.flood_probability_percept_weight
        flood_damage_percept_weight = self.model.flood_damage_percept_weight

        # Calculate flood probability perception based on time since last flood
        self.flood_probability_percept = self.model.flood_probability_percept

        # Flood damage
        self.flood_damage_percept = self.flood_damage_estimated

        # Threat appraisal
        threat_appraisal = flood_probability_percept_weight * self.flood_probability_percept + flood_damage_percept_weight * self.flood_damage_percept

        return threat_appraisal

    # ! Coping appraisal
    def update_coping_appraisal(self):
        """Function to update coping appraisal."""

        # * Coping Efficacy
        # Weights for coping efficacy
        coping_efficacy_weight = self.model.coping_efficacy_weight

        # Ensure the divisor (self.flood_depth_estimated) is not zero
        if self.flood_depth_estimated != 0:
            # Calculate Coping efficacy with a cap at 1
            reduction_potential = max(1 - (self.flood_depth_estimated - self.model.mitigation_effectiveness) / self.flood_depth_estimated, 0)
            # Example:
                # 1 - (3 - 2.4) / 3 = 0.8 (80% reduction)
                # 1 - (1 - 2.4) / 1 = 2.4 (240% reduction) --> Values need to be capped at 1
                # 1 - (16 - 2.4) / 16 = 0.15 (15% reduction)
                # 1 - (0 - 2.4) / 0 = 1 (100% reduction) --> Actually here no reduction is needed since there is no flood depth so value should be 0
        else:
            # Handle the case where self.flood_depth_estimated is zero
            reduction_potential = 0

        # Calculate Coping efficacy with a cap at 1
        self.coping_efficacy = max(0, min(1, reduction_potential))

        # * Self Efficacy
        # Weights for self efficacy
        self_efficacy_weight = self.model.self_efficacy_weight
        savings_weight = self.model.savings_weight
        income_weight = self.model.income_weight
        social_network_weight = self.model.social_network_weight

        # Calculate the contribution of each factor
        self.savings_contribution = min(1, self.savings / self.model.cost_of_measure)  # Capped at 1
        self.income_contribution = self.income / self.model.income_max
        self.social_network_contribution = self.friends_network / self.model.number_of_households

        # Calculate Self efficacy
        self.self_efficacy = savings_weight * self.savings_contribution + income_weight * self.income_contribution + social_network_weight * self.social_network_contribution

        # * Coping Appraisal
        # Calculate final coping appraisal
        coping_appraisal = coping_efficacy_weight * self.coping_efficacy + self_efficacy_weight * self.self_efficacy

        return coping_appraisal

    # ADAPTATION FUNCTIONS________________________________________________

    # ! Function to decide whether to adapt or not
    def adapt_to_flooding(self):
        """Function to adapt to flooding."""

        # Cost of adaptation measure
        cost_of_measure = self.model.cost_of_measure

        # Set threat and coping thresholds
        threat_threshold = self.model.threat_threshold  # Adjust this threshold as needed
        coping_threshold = self.model.coping_threshold  # Adjust this threshold as needed

        # Check if agent can afford adaptation measure and if threat and coping appraisals are above thresholds
        if not self.is_adapted and self.savings + self.actual_subsidy_height >= cost_of_measure:
            if self.threat_appraisal > threat_threshold and self.coping_appraisal > coping_threshold:
                # Adapt to flooding if agent is not already adapted, can afford adaptation measure,
                # and threat and coping appraisals are above thresholds
                self.is_adapted = True

                # Deduct subsidy amount from the total subsidies spent
                self.model.total_subsidies_used += min(self.actual_subsidy_height, cost_of_measure)

                # Reduce savings by cost of adaptation measure
                self.savings = self.savings - cost_of_measure + self.actual_subsidy_height


#################################### AGENT STEP ######################################

    def step(self):

        # ! Update friends count
        # Count friends within a radius of 1
        #self.friends_amount = self.count_friends(radius=1)
        self.friends_network = self.count_friends(radius=3)

        # ! Increase Savings
        savings_increase = self.income * self.savings_share
        self.savings += savings_increase

        # ! Update threat and coping appraisal
        self.threat_appraisal = self.update_threat_appraisal()
        self.coping_appraisal = self.update_coping_appraisal()

        # ! Adapt to Flooding
        if not self.is_adapted:
            self.adapt_to_flooding() # adapts agent to flooding if conditions are met
