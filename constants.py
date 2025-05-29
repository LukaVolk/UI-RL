CAR_INSTANCES = 15
REINFORCEMENT_LEARNING = True
SHOW_CHECKPOINTS = True
SHOW_WALLS = False
SHOW_BOUNDRIES = False
SHOW_SENSORS = False
CHECKPOINT_WIDTH = 50
TOP_K = 5 # Number of top cars to learn from

#Reinforcement Learning Constants
EPISODE_LENGTH = 30
EPISODE_NUMBERS = 1500
ACTION_INTERVAL = 0.1 # Time interval between actions in seconds
OBESERVATION_SIZE = 9+8 # Number of sensors + speed + angle + distance to next checkpoint

# SCORING
# Checkpoint rewards
CHECKPOINT_REWARD = 1000 # DONE
WRONG_CHECKPOINT_PENALTY = -5000 # DONE

# Terminal rewards
FINISH_LINE_REWARD = 10000 # DONE
WALL_PENALTY = -10 # Increased penalty for hitting walls

# Continuous rewards
TIME_PENALTY = -0.1 # Reduced time penalty
SPEED_REWARD = 2.0 # Increased speed reward to encourage forward movement
SPEED_PENALTY = -5.0 # Increased penalty for low speed
BACKWARD_PENALTY = -10.0 # New: penalty for moving backward
PROGRESS_REWARD = 1.0 # New: reward for making progress toward checkpoint
MIN_SPEED_THRESHOLD = 5
MIN_FORWARD_SPEED = 1.0 # New: minimum forward speed threshold

# Action space - removed backward actions to prevent backward preference
ACTION_MAP = {
    0: ['brake'],
    1: ['forward'],  # Forward only
    2: ['forward', 'left'],  # Forward + Left
    3: ['forward', 'right'],  # Forward + Right
    4: ['left'],  # Left only (for corrections)
    5: ['right'],  # Right only (for corrections)
    6: ['right', 'brake'],  # Forward + Brake
    7: ['left', 'brake'],  # Forward + Brake
    # Removed backward and handbrake actions to fix backward preference
}
NUM_ACTIONS = len(ACTION_MAP)