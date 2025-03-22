# ---------------------training with MLP policy or CnnPolicy----------------
# ---------train.py file--------------------
gym.register_envs(ale_py)
# **Set the Atri environment
env_id = 'ALE/Breakout-v5'
env = gym.make(env_id, render_mode='rgb_array')

# **Seperate the evaluation environment

env_eval = gym.make(env_id, render_mode='rgb_array')

# ** Experiment with policies
policy_type = 'MlpPolicy'


# *Define the DQN agent model
model = DQN(
    policy=policy_type,
    env=env,
    learning_rate=1e-5,
    buffer_size=10000,
    learning_starts=50000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    verbose=1,
    train_freq=5,
    device=device,
    tensorboard_log='./dqn_agent_tensorboard_log/'
)

# ** Evaluate the callback

eval_callback = EvalCallback(
    env_eval,
    best_model_save_path='./logs/best_model/',
    log_path='./logs/results/',
    eval_freq=50000,
    n_eval_episodes=7,
    deterministic=True,
    render=False
)

# Train the model
timesteps = 500000
model.learn(total_timesteps=timesteps,
            progress_bar=False, callback=eval_callback)

# Delete the replay buffer from memory before saving
model.replay_buffer = None

# Save the model without replay buffer
model.save('dqn_agent_breakout_mlp_final')

# *evaluate the trained model
mean_reward, std_reward = evaluate_policy(
    model,
    env_eval,
    n_eval_episodes=15,
    deterministic=True
)
print(f'final reward: {mean_reward}+/-{std_reward}')
