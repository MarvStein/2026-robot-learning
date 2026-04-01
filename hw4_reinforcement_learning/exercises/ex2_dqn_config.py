"""
Hyperparameters for Exercise 2 (DQN).

You are encouraged to tune:
- lr
- epsilon
- target_update
- hidden_dim

Please keep the remaining parameters unchanged unless explicitly stated.
"""

DQN_PARAMETERS = {
    # Tune the following hyperparameters
    # Replace the default values with your own choices.
    # NOTE: These default parameters already achieve perfect performance
    # with mean return of 500 with std of 0.00 in evaluation.
    # I couldn't find other hyperparemeters with the same performance,
    # which is why i left these in place.
    "lr": 1e-3,             # default: 1e-3
    "epsilon": 0.03,        # default: 0.03
    "target_update": 10,    # default: 10
    "hidden_dim": 128,      # default: 128
    
    # Fixed parameters
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 0,
}