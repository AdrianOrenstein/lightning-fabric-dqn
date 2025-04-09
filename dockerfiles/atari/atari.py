import gymnasium as gym

def check_ale_with_gymnasium():
    try:
        # Initialize the Atari environment (e.g., Breakout)
        env = gym.make("ALE/Breakout-v5")

        # Reset the environment to get the initial observation
        observation, info = env.reset()
        print("Environment initialized and reset.")

        # Take a few random actions
        for _ in range(1000):
            action = env.action_space.sample()  # Take a random action
            observation, reward, done, truncated, info = env.step(action)
            print(f"Action: {action=}, Reward: {reward=}")

            # If the episode ends, reset the environment
            if done or truncated:
                print("Episode finished. Resetting environment.")
                observation, info = env.reset()

        # Close the environment when done
        env.close()
        print("ALE with gymnasium is working correctly!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_ale_with_gymnasium()