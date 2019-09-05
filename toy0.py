import gym
import time

# env = gym.make('MountainCar-v0')
env = gym.make("CartPole-v0")
env.reset()
step = 0
for _ in range(100):
    env.render()
    action = env.action_space.sample()
    # print("action:", action)
    observation, reward, done, info = env.step(action)  # take a random action
    # print("s,r,d,i", observation, reward, done, info)
    step += 1
    if done:
        print("Reset....", step)
        env.reset()
        step = 0
    # time.sleep(0.03)
env.close()
