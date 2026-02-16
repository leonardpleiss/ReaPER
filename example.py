from custom_policies.custom_ddqn import CustomDDQN
from custom_buffers.reaper import ReaPER
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold

# === Configuration === #
SEED = 0
TOTAL_TIMESTEPS = 50_000
ENVIRONMENT_NAME = "CartPole-v1"
REPLAY_BUFFER_CLASS = ReaPER
MODEL_NAME = CustomDDQN
REPLAY_BUFFER_KWARGS = {
    "alpha": .4,
    "omega": .2,
}

# === Initialize model and environment === #
env = make_vec_env(ENVIRONMENT_NAME, seed=SEED)
eval_env = make_vec_env(ENVIRONMENT_NAME, seed=SEED)

model = CustomDDQN(
    env=env,
    verbose=0,
    seed=SEED,
    policy="MlpPolicy",
    replay_buffer_class=REPLAY_BUFFER_CLASS,
    replay_buffer_kwargs=REPLAY_BUFFER_KWARGS,

    # learning_rate = 2.3e-3,
    # learning_starts = 1000,
    # batch_size = 64,
    # buffer_size = 100000,
    # target_update_interval = 10,
    # train_freq = 256,
    # gradient_steps = 128,
    # exploration_fraction = 0.16,
    # exploration_final_eps = 0.04,
)

# === Evaluation Callback === #
eval_callback = EvalCallback(
    eval_env=eval_env,
    callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=475, verbose = 1),
    eval_freq=500,
    n_eval_episodes=5,
    verbose=1
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
