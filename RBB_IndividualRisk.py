from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import math

################################## FUNCTIONS ##################################

#   PROBABILITY OF DYING FUNCTION
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

#   INDIVIDUAL RISK FUNCTION
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
    individual_risk = probability_of_failure * probability_of_dying

    if individual_risk <= low_risk_threshold:
        risk_group = "low"  # acceptably low risk
    elif individual_risk < medium_risk_threshold:
        risk_group = "medium"  # medium risk
    else:
        risk_group = "high"  # unacceptably high risk

    return risk_group

################################## AGENT CLASS ##################################

class Households(Agent):
    """
    An agent representing a household in the model.

    Attributes:
    - probability_of_failure (float): Probability of flooding for the household.
    - flood_depth_estimated (float): Randomly assigned estimated flood depth for demonstration purposes.
    - probability_of_dying (float): Probability of dying based on the flood depth.
    - individual_risk (str): Risk category for the household (low, medium, high).
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Individual risk parameters
        self.probability_of_failure = model.flood_probability  # Probability of flooding
        self.flood_depth_estimated = random.uniform(0, 10)  # Random flood depth estimate for demonstration purposes

        # Probability of dying
        self.probability_of_dying = calculate_probability_of_dying(flood_depth_estimated=self.flood_depth_estimated)

        # Individual risk
        self.individual_risk = calculate_individual_risk(
            probability_of_failure=self.probability_of_failure,
            probability_of_dying=self.probability_of_dying,
            low_risk_threshold=model.low_risk_threshold,
            medium_risk_threshold=model.medium_risk_threshold
        )

################################## MODEL CLASS ##################################

class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents, and collects data.

    Attributes:
    - number_of_households (int): Number of household agents in the model.
    - grid (MultiGrid): Mesa MultiGrid representing the spatial environment.
    - schedule (RandomActivation): Mesa RandomActivation for agent activation.
    - low_risk_threshold (float): Threshold below which the individual risk is considered low.
    - medium_risk_threshold (float): Threshold below which the individual risk is considered medium.
    - flood_probability (float): Probability of flooding for the model.
    - datacollector (DataCollector): Mesa DataCollector for collecting model and agent data.
    """

    def __init__(self,
                 seed=None,
                 number_of_households=25,
                 width=10,
                 height=10,
                 flood_probability=0.01,
                 low_risk_threshold=10e-5,
                 medium_risk_threshold=10e-4,
                 ):

        super().__init__(seed=seed)

        # Model parameters
        self.number_of_households = number_of_households
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # Individual risk parameters
        self.low_risk_threshold = low_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.flood_probability = flood_probability

        # Create agents
        for i in range(self.number_of_households):
            a = Households(i, self)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            self.schedule.add(a)

        # Data collection setup to collect data
        model_metrics = {}

        agent_metrics = {
            "EstimatedFloodDepth": "flood_depth_estimated",
            "ProbabilityOfFailure": "probability_of_failure",
            "ProbabilityOfDying": "probability_of_dying",
            "IndividualRisk": "individual_risk",
        }

        # Set up the data collector
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    # Model step
    def step(self):
        """
        Advance the model by one step, collect data, and activate agents.
        """
        # Collect data
        self.datacollector.collect(self)
        # Activate agents
        self.schedule.step()

################################## Run the model##################################

# Set the number of time steps to run the model
time_steps = 10

# Create a model instance
model = AdaptationModel(
    seed=None,
    number_of_households=10,
    width=10,
    height=10,
    flood_probability=0.01,  # 100-year flood probability
    low_risk_threshold=10e-5,
    medium_risk_threshold=10e-4,
)

# Run the model for the specified number of time steps
for step in range(time_steps):
    model.step()

# Get the agent data
agent_data = model.datacollector.get_agent_vars_dataframe()
print(agent_data.head())
