CAR_INSTANCES = 10
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
    0: ['forward'],
    1: ['forward', 'left'],
    2: ['forward', 'right'],
    #3: ['back'],
    3: [],
    4: ['left'],
    5: ['right']
}