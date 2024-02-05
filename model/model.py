# Importing necessary libraries
import random
import math
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt

# Import the agent class(es) from agents.py
from agents import Households

# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    #################################### MODEL INITIALISATION ########################################

    def __init__(self,
                 seed = None,
                 number_of_households = 25, # number of household agents
                 # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                 flood_map_choice='harvey',
                 # ### network related parameters ###
                 # The social network structure that is used.
                 # Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                 network = 'watts_strogatz',
                 # likeliness of edge being created between two nodes
                 probability_of_network_connection = 0.4,
                 # number of edges for BA network
                 number_of_edges = 3,
                 # number of nearest neighbours for WS social network
                 number_of_nearest_neighbours = 5,

                 # ! flood parameters
                 flood_probability = 0.02, # actual flood probability
                 time_since_last_flood = 10, # time in quarters since last flood at initialization

                 # ! income parameters
                 income_min = 4000,
                 income_max = 80000,
                 income_distribution = 'uniform',
                 low_income_threshold = 25000,
                 medium_income_threshold = 60000,

                 # ! savings parameters
                 savings_share_low_income=0.001,
                 savings_share_medium_income=0.04,
                 savings_share_high_income=0.2,

                 # ! Mitigation measure parameters
                 cost_of_measure = 150000,
                 mitigation_effectiveness = 2.4,

                 # ! PMT parameters
                 flood_probability_percept = 0, # default flood probability perception, will be updated based on time since last flood
                 flood_probability_percept_weight = 0.5,
                 flood_damage_percept_weight = 0.5,

                 coping_efficacy_weight = 0.3,
                 self_efficacy_weight = 0.7,
                 savings_weight = 0.5,
                 income_weight = 0.2,
                 social_network_weight = 0.3,
                 # friends_adaptaion_weight = 0,

                 threat_threshold = 0.6,
                 coping_threshold = 0.3,

                 #! Policy parameters
                 policy = False, # "no_policy", # Can be "no_policy" or "subsidy"

                 low_risk_subsidy_height = 0,
                 medium_risk_subsidy_height = 75000,
                 high_risk_subsidy_height = 150000,

                 low_income_subsidy_factor = 1,
                 medium_income_subsidy_factor = 0.5,
                 high_income_subsidy_factor = 0,

                 # ! Individual risk parameters
                 low_risk_threshold = 0.000000000004845, # 10e-5,
                 medium_risk_threshold = 0.0000000001, # 10e-4,

                ):

        super().__init__(seed = seed)

        # defining the variables and setting the values
        self.number_of_households = number_of_households  # Total number of household agents
        self.seed = seed

        # Set the seed for random number generation
        random.seed(self.seed)

        # network
        self.network = network # Type of network to be created
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours


        # ! flood parameters
        self.flood_probability = flood_probability
        self.time_since_last_flood = time_since_last_flood

        # ! Income parameters
        self.income_min = income_min
        self.income_max = income_max
        self.income_distribution = income_distribution
        self.low_income_threshold = low_income_threshold
        self.medium_income_threshold = medium_income_threshold

        # ! Savings parameters
        self.savings_share_low_income = savings_share_low_income
        self.savings_share_medium_income = savings_share_medium_income
        self.savings_share_high_income = savings_share_high_income

        # ! Mitigation parameters
        self.cost_of_measure = cost_of_measure
        self.mitigation_effectiveness = mitigation_effectiveness

        # ! PMT parameters
        # * Threat Appraisal
        self.flood_probability_percept = flood_probability_percept
        self.flood_probability_percept_weight = flood_probability_percept_weight
        self.flood_damage_percept_weight = flood_damage_percept_weight

        # * Coping Appraisal
        self.coping_efficacy_weight = coping_efficacy_weight
        self.self_efficacy_weight = self_efficacy_weight
        self.savings_weight = savings_weight
        self.income_weight = income_weight
        self.social_network_weight = social_network_weight
        # self.friends_adaptation_weight = friends_adaptation_weight

        # * Thresholds for Adaptation
        self.threat_threshold = threat_threshold
        self.coping_threshold = coping_threshold

        # ! Policy parameters
        self.policy = policy
        self.low_risk_subsidy_height = low_risk_subsidy_height
        self.medium_risk_subsidy_height = medium_risk_subsidy_height
        self.high_risk_subsidy_height = high_risk_subsidy_height
        self.low_income_subsidy_factor = low_income_subsidy_factor
        self.medium_income_subsidy_factor = medium_income_subsidy_factor
        self.high_income_subsidy_factor = high_income_subsidy_factor
        self.total_subsidies_used = 0
        self.total_damage = 0
        self.total_damage_per_group = {'low': 0, 'medium': 0, 'high': 0}

        # ! Individual risk parameters
        self.low_risk_threshold = low_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold


        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)


        # Data collection setup to collect data
        model_metrics = {
                        "TotalAdaptedHouseholds": self.total_adapted_households,
                        "FloodProbabilityPercept": "flood_probability_percept",
                        "TimeSinceLastFlood": "time_since_last_flood",
                        "TotalSubsidiesUsed": "total_subsidies_used",
                        "TotalDamage": "total_damage",
                        "TotalDamageLowIncome": lambda m: m.total_damage_per_group['low'],
                        "TotalDamageMediumIncome": lambda m: m.total_damage_per_group['medium'],
                        "TotalDamageHighIncome": lambda m: m.total_damage_per_group['high'],
                        }

        agent_metrics = {
                        "FloodDepthEstimated": "flood_depth_estimated",
                        "FloodDamageEstimated" : "flood_damage_estimated",
                        "FloodDepthActual": "flood_depth_actual",
                        "FloodDamageActual" : "flood_damage_actual",
                        "IsAdapted": "is_adapted",
                        "location":"location",

                        #* Social network metrics
                        "FriendsCount": lambda a: a.count_friends(radius=1),
                        "FriendsNetwork": "friends_network",
                        "FriendsAdapted": "friends_adapted",

                        #* Income and savings metrics
                        "Income": "income",
                        "IncomeGroup": "income_group",
                        "Savings": "savings",

                        #* PMT metrics
                        "ThreatAppraisal": "threat_appraisal",
                        "FloodProbabilityPercept": "flood_probability_percept",
                        "FloodDamagePercept": "flood_damage_percept",

                        "CopingAppraisal": "coping_appraisal",
                        "CopingEfficacy": "coping_efficacy",
                        "SelfEfficacy": "self_efficacy",
                        "SavingsContribution": "savings_contribution",
                        "IncomeContribution": "income_contribution",
                        "SocialNetworkContribution": "social_network_contribution",

                        #* Policy metrics
                        "ProbabilityOfFailure": "probability_of_failure",
                        "ProbabilityOfDying": "probability_of_dying",
                        "IndividualRisk": "individual_risk",
                        "SubsidyHeight": "actual_subsidy_height",
                        }

        #set up the data collector
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

#################################### MODEL FUNCTIONS ########################################

    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                        k=self.number_of_nearest_neighbours,
                                        p=self.probability_of_network_connection,
                                        seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                            f"Currently implemented network types are: "
                            f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")


    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'../input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'../input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'../input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        """Return the total number of households that have adapted."""
        #BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        adapted_count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.is_adapted])
        return adapted_count

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses
        for agent in self.schedule.agents:
            color = 'blue' if agent.is_adapted else 'red'
            ax.scatter(agent.location.x, agent.location.y, color=color, s=10, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0,1), ha='center', fontsize=9)

        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue: adapted")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    # ! Function to calculate flood probability perception
    def calculate_flood_probability_percept(self):
        """
        Calculate the flood probability perception based on the time since last flood.

        Maximum flood probability perception is 1, and is when a flood just occured
        Over time, the flood probability perception decreases towards 0

        """

        # * Alternative 1
        # flood_probability_percept = 1 / max(self.time_since_last_flood, 1)

        # * Alternative 2
        # decay_factor = 0.001
        # flood_probability_percept = 1 / (max(self.time_since_last_flood, 1) + decay_factor * self.time_since_last_flood)

        # * Alternative 3 - Exponential decay
        # decay_rate = 0.02
        # flood_probability_percept = math.exp(-decay_rate * self.time_since_last_flood)

        # * Alternative 4 - Logistic function decay (s-curve)
        steepness=0.1
        midpoint=50
        flood_probability_percept = 1 / (1 + math.exp(steepness * (self.time_since_last_flood - midpoint)))

        return flood_probability_percept

#################################### MODEL STEP ########################################

    def step(self):
        """
        introducing a shock:
        at time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth. In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and
        assume local flooding instead of global flooding). The actual flood depth can be
        estimated differently
        """

        # * Update time since last flood
        self.time_since_last_flood += 1  # Increment time since last flood at each step in quarters
        self.flood_probability_percept = self.calculate_flood_probability_percept()

        # * Introduce flood at timeset 5
        if self.schedule.steps == 5:
            for agent in self.schedule.agents:
                # Calculate the actual flood depth as a random number between 0.5 and 1.2 times the estimated flood depth
                agent.flood_depth_actual = random.uniform(0.5, 1.2) * agent.flood_depth_estimated
                # calculate the actual flood damage given the actual flood depth
                agent.flood_damage_actual = calculate_basic_flood_damage(agent.flood_depth_actual, agent.is_adapted, self.mitigation_effectiveness)

                # Accumulate total damage
                self.total_damage += agent.flood_damage_actual

                # Accumulate total damage per income group
                self.total_damage_per_group[agent.income_group] += agent.flood_damage_actual


            # * Reset time since last flood
            self.time_since_last_flood = 0

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
