import gym
from toy_mc_q import Q_net, choose_action
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0")
env.reset()
Q = Q_net().to(device)
Q.load_state_dict(torch.load("./models/q_net_v1_44862_0.pth"))
s = np.zeros((4))
step = 0
for i in range(1000):
    env.render()
    step += 1
    if i == 0:
        action = env.action_space.sample()
    else:
        action = choose_action(Q, s)
    # print("action:", action)
    observation, reward, done, info = env.step(action)
    s = observation
    if done:
        print("Reset....", step)
        step = 0
        env.reset()
env.close()
