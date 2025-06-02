import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib.patches import Rectangle, Arrow
from matplotlib.path import Path

class RoomEnvironment:
    def __init__(self):
        # Define the possible states (rooms 0-5)
        self.states = [0, 1, 2, 3, 4, 5]
        
        # Define the connections between rooms
        self.connections = {
            0: [4],
            1: [3, 5],
            2: [3],
            3: [1, 2, 4],
            4: [0, 3, 5],
            5: [1, 4]
        }
        
        # Define rewards
        self.rewards = {}
        for state in self.states:
            for next_state in self.connections[state]:
                if next_state == 5:
                    self.rewards[(state, next_state)] = 100  # Reward for reaching outside
                elif state == 5 and next_state == 5:
                    self.rewards[(state, next_state)] = 100  # Reward for staying outside
                else:
                    self.rewards[(state, next_state)] = 0  # No reward for other transitions

    def get_available_actions(self, state):
        return self.connections[state]

    def get_reward(self, state, action):
        return self.rewards.get((state, action), 0)

    def transition(self, state, action):
        if action in self.connections[state]:
            return action  # In this simple environment, the action directly determines the next state
        else:
            return state  # Invalid action, stay in the same state

def q_learning(env, episodes=1000, alpha=0.1, gamma=0.8, epsilon=0.1):
    # Initialize Q-table with zeros
    q_table = {}
    for state in env.states:
        q_table[state] = {}
        for action in env.connections[state]:
            q_table[state][action] = 0

    # Store Q-table history for visualization
    q_history = []

    # Run Q-learning algorithm
    for episode in range(episodes):
        # Choose a random starting state (excluding the goal state 5)
        if episode == 0:
            current_state = 1  # Start from room 1 for the first episode for demonstration
        else:
            current_state = random.choice([0, 1, 2, 3, 4])

        done = False

        while not done:
            # Choose action using epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                # Explore: choose a random action
                action = random.choice(env.get_available_actions(current_state))
            else:
                # Exploit: choose the best action
                best_actions = [a for a in env.get_available_actions(current_state)
                              if q_table[current_state][a] == max(q_table[current_state].values())]
                action = random.choice(best_actions)

            # Take action and observe reward and next state
            next_state = env.transition(current_state, action)
            reward = env.get_reward(current_state, next_state)

            # Update Q-value
            best_next_action_value = max(q_table[next_state].values()) if q_table[next_state] else 0
            q_table[current_state][action] += alpha * (reward + gamma * best_next_action_value - q_table[current_state][action])

            # Move to the next state
            current_state = next_state

            # Check if episode is done
            if current_state == 5:  # Reached outside
                done = True

        # Save a copy of the Q-table at certain episodes
        if episode in [0, 10, 50, 100, 500, 999]:
            q_history.append(q_table.copy())

    return q_table, q_history

# Extract optimal policy from Q-table
def get_optimal_policy(q_table):
    policy = {}
    for state in q_table:
        if q_table[state]:  # Check if the state has any actions
            best_action = max(q_table[state], key=q_table[state].get)
            policy[state] = best_action
    return policy

# Visualize the environment and optimal path
def visualize_environment(policy):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw rooms
    room_positions = {
        0: (1, 1),
        1: (1, 3),
        2: (3, 1),
        3: (3, 3),
        4: (5, 2),
        5: (7, 2)  # Outside
    }

    room_size = 1

    for room, pos in room_positions.items():
        color = 'lightgreen' if room == 5 else 'lightblue'
        rect = Rectangle((pos[0]-room_size/2, pos[1]-room_size/2), room_size, room_size,
                        facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], f'Room {room}', ha='center', va='center')

    # Draw connections (gray lines)
    for room, connections in env.connections.items():
        for connected_room in connections:
            start_pos = room_positions[room]
            end_pos = room_positions[connected_room]
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'gray', linestyle='--', alpha=0.5)

    # Draw optimal path (red arrows)
    current_room = 1  # Start from room 1
    visited = set()

    while current_room != 5 and current_room not in visited:
        visited.add(current_room)
        if current_room in policy:
            next_room = policy[current_room]
            start_pos = room_positions[current_room]
            end_pos = room_positions[next_room]

            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]

            arrow = Arrow(start_pos[0], start_pos[1], dx*0.8, dy*0.8,
                         width=0.2, color='red')
            ax.add_patch(arrow)

            current_room = next_room
        else:
            break

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.set_title('Room Environment with Optimal Path')
    plt.show()

# Visualize Q-values evolution
def visualize_q_table_evolution(q_history, episodes):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (q_table, episode) in enumerate(zip(q_history, episodes)):
        # Convert Q-table to a 2D matrix for visualization
        q_matrix = np.zeros((6, 6))
        for state in q_table:
            for action in q_table[state]:
                q_matrix[state, action] = q_table[state][action]

        # Plot heatmap
        sns.heatmap(q_matrix, annot=True, cmap='viridis', ax=axes[i], fmt='.1f')
        axes[i].set_title(f'Q-values after {episode+1} episodes')
        axes[i].set_xlabel('Action (Next Room)')
        axes[i].set_ylabel('Current Room')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create environment
    env = RoomEnvironment()

    # Run Q-learning algorithm
    q_table, q_history = q_learning(env, episodes=1000)

    # Extract optimal policy
    policy = get_optimal_policy(q_table)

    # Print the final Q-table
    print("Final Q-table:")
    for state in q_table:
        for action in q_table[state]:
            print(f"Q({state}, {action}) = {q_table[state][action]:.2f}")

    # Print optimal policy
    print("\nOptimal Policy:")
    for state in policy:
        print(f"From Room {state}, go to Room {policy[state]}")

    # Visualize environment and optimal path
    visualize_environment(policy)

    # Visualize Q-table evolution
    visualize_q_table_evolution(q_history, [0, 10, 50, 100, 500, 999])