import pytest
import numpy as np

import pdb
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from pufferlib.ocean.crossing.crossing import RiverCrossing

@pytest.fixture
def env():
    env = RiverCrossing(passengers=1, max_passengers= 1, max_boats=1, boats=1, max_ep_steps=20)
    env.reset()
    return env

def get_entity_rows():
    return 1, 2  # passenger row, agent row

def get_boat_row():
    return 3  # header + 1 passenger + 1 agent = boat starts at row 3

def test_reset_state(env):
    obs, _ = env.reset()
    obs = obs[0]
    passenger_row, agent_row = get_entity_rows()
    boat_row = get_boat_row()

    assert obs[passenger_row][0] == 1  # passenger on left
    assert obs[agent_row][0] == 1      # agent on left
    assert obs[boat_row][0] == 1       # boat on left
    assert obs[passenger_row][2] == 0  # passenger not on right
    assert obs[agent_row][2] == 0      # agent not on right

def test_valid_trip(env):
    """Simulates a full successful river crossing"""
    p_row, a_row = get_entity_rows()
    boat_idx = 0
    passenger_idx = 0
    agent_idx = 1

    env.step([boat_idx, agent_idx, 1])  # LOAD agent
    env.step([boat_idx, passenger_idx, 1])  # LOAD passenger
    env.step([boat_idx, agent_idx, 2])  # MOVE
    env.step([boat_idx, passenger_idx, 0])  # UNLOAD passenger
    obs, reward, done, _, _ = env.step([boat_idx, agent_idx, 0])  # UNLOAD agent

    assert done[0] == True
    assert reward[0] == 1.0

def test_invalid_double_load(env):
    """Tries to load too many entities into the boat"""
    passenger_idx = 0
    agent_idx = 1
    boat_idx = 0

    env.step([boat_idx, agent_idx, 1])  # LOAD agent
    env.step([boat_idx, passenger_idx, 1])  # LOAD passenger (boat now full)
    obs, reward, done, _, _ = env.step([boat_idx, agent_idx, 1])  # Try to load again

    assert done[0] == True
    assert reward[0] == -1.0

def test_invalid_move_without_agent(env):
    """Tries to move boat with no agent inside"""
    boat_idx = 0
    passenger_idx = 0

    obs, reward, done, _, _ = env.step([boat_idx, passenger_idx, 2])  # MOVE with no agent

    assert done[0] == True
    assert reward[0] == -1.0

def test_passenger_alone_is_allowed(env):
    """Passenger alone on opposite bank is allowed — should not terminate or penalize"""

    boat_idx = 0
    passenger_idx = 0
    agent_idx = 1
    # LOAD agent and passenger
    env.step([boat_idx, agent_idx, 1])      # LOAD agent
    env.step([boat_idx, passenger_idx, 1])  # LOAD passenger

    # MOVE across and unload only passenger
    env.step([boat_idx, agent_idx, 2])      # MOVE
    env.step([boat_idx, passenger_idx, 0])  # UNLOAD passenger

    # Agent returns
    env.step([boat_idx, agent_idx, 2])      # MOVE back
    obs, reward, done, _, _ = env.step([boat_idx, agent_idx, 0])  # UNLOAD agent

    # Nothing bad should happen — episode should not be done
    assert done[0] == False
    assert reward[0] == 0.0


def test_passenger_with_wrong_agent_causes_failure():
    """Passenger ends up with the wrong agent while correct agent is absent — should fail"""

    # 2 passengers => 2 agents
    env = RiverCrossing(passengers=2, max_passengers=2, max_boats=1, boats=1, max_ep_steps=20)
    env.reset()

    boat_idx = 0
    wrong_agent_idx = 3 #This is A2
    passenger_idx = 0  # Passenger 1 is assigned to agent 1

    # Load wrong agent (agent 0) and passenger 1
    obs, reward, done, _, _ = env.step([boat_idx, wrong_agent_idx, 1])  # LOAD agent 2
    #Should immediately fail as leaves pasenger 2 alone with A1
    assert done[0] == True
    assert reward[0] == -1.0

    env.step([boat_idx, passenger_idx, 1]) # LOAD passenger 1
    obs, reward, done, _, _ = env.step([boat_idx, wrong_agent_idx, 1])  # LOAD agent 2   
    #should immediately fail
    assert done[0] == True
    assert reward[0] == -1.0

    