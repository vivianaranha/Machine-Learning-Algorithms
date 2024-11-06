import numpy as np
import random

# Define the environment (4x4 grid)
num_states = 16  # 4x4 grid
num_actions = 4  # Up, Right, Down, Left
q_table = np.zeros((num_states, num_actions))

# Define the parameters
alpha = 0.1         # Learning rate
gamma = 0.9         # Discount factor
epsilon = 0.2       # Exploration rate
num_episodes = 1000

# Define a simple reward structure
rewards = np.zeros(num_states)
rewards[15] = 1  # Goal state with a reward

# Function to determine the next state based on the action
def get_next_state(state, action):
    if action == 0 and state >= 4:            # Up
        return state - 4
    elif action == 1 and (state + 1) % 4 != 0: # Right
        return state + 1
    elif action == 2 and state < 12:          # Down
        return state + 4
    elif action == 3 and state % 4 != 0:      # Left
        return state - 1
    else:
        return state  # If action goes out of bounds, remain in the same state

# Q-Learning algorithm
for episode in range(num_episodes):
    state = random.randint(0, num_states - 1)  # Start from a random state
    while state != 15:  # Loop until reaching the goal state
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)  # Random action (exploration)
        else:
            action = np.argmax(q_table[state])           # Best known action (exploitation)

        next_state = get_next_state(state, action)       # Get the resulting state
        reward = rewards[next_state]                     # Get the reward for the new state
        old_value = q_table[state, action]               # Current Q-value
        next_max = np.max(q_table[next_state])           # Max Q-value for next state

        # Q-Learning update rule
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state  # Move to the next state

# Display the learned Q-table
print("Learned Q-Table:")
print(q_table)










