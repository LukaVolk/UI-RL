CAR_INSTANCES = 2
REINFORCEMENT_LEARNING = True
SHOW_CHECKPOINTS = True
SHOW_WALLS = False
SHOW_BOUNDRIES = False
SHOW_SENSORS = False
CHECKPOINT_WIDTH = 50
TOP_K = 1 # Number of top cars to learn from

#Reinforcement Learning Constants
EPISODE_LENGTH = 10
EPISODE_NUMBERS = 20
ACTION_INTERVAL = 0.1 # Time interval between actions in seconds
OBESERVATION_SIZE = 9+5 # Number of sensors + speed + angle + distance to next checkpoint

# SCORING
# Checkpoint rewards
CHECKPOINT_REWARD = 1000 # DONE
WRONG_CHECKPOINT_PENALTY = -250 # DONE

# Terminal rewards
FINISH_LINE_REWARD = 10000 # DONE
WALL_PENALTY = -1

# Continuous rewards
TIME_PENALTY = -1
SPEED_REWARD = 0.1
SPEED_PENALTY = -0.5
MIN_SPEED_THRESHOLD = 5

ACTION_MAP = {
    0: [],
    1: ['forward'],
    2: ['forward', 'left'],
    3: ['forward', 'right'],
    4: ['back'],
    5: ['handbrake'],
    6: ['left'],
    7: ['right']
}