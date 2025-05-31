import json
import matplotlib.pyplot as plt
import numpy as np
from constants import ACTION_MAP

def plot_rl_training_data(json_file_path="logs/episode_stats.json"):
    """
    Reads RL training data from a JSON file and plots various metrics over episodes.

    Args:
        json_file_path (str): The path to your JSON file containing training logs.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Check file format.")
        return

    if not data:
        print("No data found in the JSON file.")
        return

    # Extract data for plotting
    episodes = [entry['episode'] for entry in data]
    avg_rewards = [entry['avg_reward'] for entry in data]
    max_rewards = [entry['max_reward'] for entry in data]
    avg_wall_hits = [entry['avg_wall_hits'] for entry in data]
    total_wall_hits = [entry['total_wall_hits'] for entry in data]

    # --- Plotting Average and Max Rewards ---
    plt.figure(figsize=(12, 6))
    #plt.plot(episodes, avg_rewards, label='Average Reward per Car in Episode', marker='o', linestyle='-', markersize=4)
    plt.plot(episodes, max_rewards, label='Max Reward (Best Car) in Episode', marker='x', linestyle='--', markersize=4)
    plt.title('Reward Trends Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reward_trends.png')
    plt.show()

    # --- Plotting Average and Total Wall Hits ---
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, avg_wall_hits, label='Average Wall Hits per Car in Episode', marker='o', linestyle='-', markersize=4, color='orange')
    plt.plot(episodes, total_wall_hits, label='Total Wall Hits in Episode', marker='x', linestyle='--', markersize=4, color='red')
    plt.title('Wall Hits Trends Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Wall Hits')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('wall_hits_trends.png')
    plt.show()

    # --- Optional: Analyze Action Distribution (from a single episode or aggregated) ---
    # This example takes the first episode's action distribution.

    # Aggregate action counts over all episodes
    action_totals = {}
    episodes_with_actions = 0

    for entry in data:
        if 'episode_actions' in entry:
            episodes_with_actions += 1
            for action, count in entry['episode_actions'].items():
                action_totals[action] = action_totals.get(action, 0) + count

    if action_totals and ACTION_MAP:
        # Map action indices to action names using ACTION_MAP
        actions_labels = [str(ACTION_MAP.get(int(a), str(a))) for a in action_totals.keys()]
        actions_counts = [action_totals[a] for a in action_totals.keys()]

        actions_labels = [label[1:-1] for label in actions_labels]  # Clean up labels to remove brackets
        plt.figure(figsize=(10, 5))
        plt.bar(actions_labels, actions_counts, color='skyblue')
        plt.title(f'Aggregated Action Distribution over {episodes_with_actions} Episodes')
        plt.xticks(fontsize=8)  # Set font size for action labels
        plt.xlabel('Action')
        plt.ylabel('Total Count')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig('action_distribution_aggregated.png')
        plt.show()
    elif not ACTION_MAP:
        print("ACTION_MAP is missing or empty. Cannot map action indices to names.")
    else:
        print("No 'episode_actions' data found to plot aggregated action distribution.")


# --- How to use the function ---
if __name__ == "__main__":
    # Make sure your JSON file is in the same directory as this script,
    # or provide the full path to your JSON file.
    plot_rl_training_data() # <--- REMEMBER TO RENAME THIS TO YOUR ACTUAL JSON FILE NAME