CAR_INSTANCES = 1
REINFORCEMENT_LEARNING = False
SHOW_CHECKPOINTS = True
SHOW_WALLS = True
SHOW_BOUNDRIES = True
SHOW_SENSORS = False
CHECKPOINT_WIDTH = 50

# SCORING
# Checkpoint rewards
CHECKPOINT_REWARD = 50 # DONE
WRONG_CHECKPOINT_PENALTY = -25 # DONE

# Terminal rewards
FINISH_LINE_REWARD = 200 # DONE
WALL_PENALTY = -1

# Continuous rewards
TIME_PENALTY = -1
SPEED_REWARD = 0.1
SPEED_PENALTY = -0.5
MIN_SPEED_THRESHOLD = 5

ACTION_MAP = {
    1: ['forward'],
    2: ['forward', 'left'],
    3: ['forward', 'right'],
    4: ['back'],
    5: ['handbrake']
}