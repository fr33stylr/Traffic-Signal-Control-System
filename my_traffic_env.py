import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import os
import sys

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        # 1. DEFINE THE "EYES" (Observation Space)
        # The AI will "see" the number of cars in 4 specific lanes.
        # We define a box of 4 numbers, ranging from 0 to 100 cars.
        self.observation_space = spaces.Box(low=0, high=100, shape=(8,), dtype=np.float32)

        # 2. DEFINE THE "HANDS" (Action Space)
        # The AI has 2 buttons:
        # 0 = Keep the current light phase
        # 1 = Switch to the next phase
        self.action_space = spaces.Discrete(2)

        # 3. YOUR SPECIFIC CONFIGURATION
        self.sumo_cmd = ["sumo", "-c", "mysim.sumocfg", "--start", "--quit-on-end", "--no-step-log", "true", "--waiting-time-memory", "1000"]
        self.tls_id = "J7"  # <--- CONFIRM THIS ID IN NETEDIT!
        
        # *** CRITICAL: PUT YOUR INCOMING LANE IDs HERE ***
        # These are the lanes the AI watches to decide if it should switch.
        # Usually: [North_Incoming, East_Incoming, South_Incoming, West_Incoming]
        # Example: "E1_0" means Edge E1, Lane 0.
        self.lanes = ["E1_0", "-E0_0", "-E1_0", "E3_0","E1_1", "-E0_1", "-E1_1", "E3_1"] 

    def reset(self, seed=None, options=None):
        """Called at the start of every training episode."""
        try:
            traci.close()
        except:
            pass
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        
        # Return the first "sight" of the world (0 cars initially)
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        """The AI takes an action, and we tell it what happened."""
        
        # 1. APPLY ACTION
        if action == 1:
            # If AI says "Switch", we cycle to the next phase
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            traci.trafficlight.setPhase(self.tls_id, (current_phase + 1) % 4)
        
        # 2. FAST FORWARD
        # We don't make decisions every millisecond. We act, then wait 5 seconds.
        for _ in range(15):
            traci.simulationStep()

        # 3. GET NEW STATE (What does the AI see now?)
        # We ask SUMO: "How many cars are halted in these lanes?"
        observations = []
        for lane in self.lanes:
            # specifically counting cars moving slower than 0.1 m/s (stopped)
            vehicle_count = traci.lane.getLastStepHaltingNumber(lane)
            observations.append(vehicle_count)
        
        state = np.array(observations, dtype=np.float32)

        # 4. CALCULATE REWARD (The Grade)
        # Goal: Reduce waiting time.
        # If queue is long, Reward is Negative (Punishment).
        # If queue is short, Reward is close to Zero (Good).
        total_stopped_cars = sum(observations)
        reward = -total_stopped_cars 

        if action == 1:
            reward -= 10  # Small penalty for switching to encourage thoughtful decisions

        # 5. CHECK IF DONE
        # Stop after 1000 steps so we can restart and practice again.
        terminated = traci.simulation.getTime() > 3600
        truncated = False
        
        #print(f"AI Sees: {state} | Reward: {reward}")

        return state, reward, terminated, truncated, {}