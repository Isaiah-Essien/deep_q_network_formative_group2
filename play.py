'''Paly the Atari game on PC using the trained model '''

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import ale_py

# Load and preprocess environment


def main():
    gym.register_envs(ale_py)

    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = MaxAndSkipEnv(env, skip=4)  # Reduce frame rate
    env = DummyVecEnv([lambda: env])  # Convert to vectorized env
    env = VecFrameStack(env, n_stack=4)  # Stack frames for better performance

    # Load the trained model
    try:
        model = DQN.load('dqn_agent_breakout_final', exclude=['replay_buffer'])

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'dqn_agent_breakout_final.zip' exists and is a valid model file.")
        return  # Exit script if model loading fails

    # Play multiple episodes
    episodes = 5
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Use the model to predict the best action (greedy policy)
            action, _ = model.predict(obs, deterministic=True)

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(
            f"Episode {episode + 1} - Total Reward: {total_reward} - Steps: {steps}")

    env.close()


main()


'''Play the game using with recorded videos on Google Colab Comment this part out if you are running on your computer'''

# --------------------------Play.py script----------

# **Load the environment

env_id = 'ALE/Breakout-v5'
# env=gym.make(env_id, render_mode='human')

env = gym.make(env_id, render_mode='rgb_array')
# Wrap environment for recording videos (saved in "./videos/" folder)
env = RecordVideo(env, video_folder='./videos/',
                  episode_trigger=lambda x: True)


# **Load the trained model

model = DQN.load(model_path)

# Play 2 episodes
episodes = 5


for episode in range(episodes):
  obs, info = env.reset()
  done = False
  total_reward = 0

  while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

  print(f'Episodes {episode+1}\n Reward: {total_reward}')

env.close()
