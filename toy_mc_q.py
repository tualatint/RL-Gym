import gym
import numpy as np
import torch
from torch.utils import data


env = gym.make("CartPole-v0")
env.reset()

#### calculate the cumulative reward ##
def cal_exp_f_r(l, gamma):
    t_r = 0.0
    r = 1.0
    for i in range(l):
        if i == 0:
            t_r += 1.0
        else:
            r = r * gamma
            t_r += r
    return t_r - 1.0


####  define the network ####
class Q_net(torch.nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.fc1 = torch.nn.Linear(5, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# ### training Q net ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
criterion = torch.nn.MSELoss()
gamma = 0.99
lr = 1e-4
Q = Q_net().to(device)
optimizer = torch.optim.Adam(Q.parameters(), lr=lr, weight_decay=1e-5)


class Sampler(data.Dataset):
    def __init__(self, record_list):
        self.rl = record_list
        self.l = len(record_list)

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        try:
            r = self.rl[index]
            length = len(r)
            seed = np.random.randint(0, length)
            future_r = cal_exp_f_r(length - seed, gamma)
            return r[seed], future_r
        except RuntimeError:
            print("r", r[seed])

    def clear(self):
        del self.rl


def choose_action(Q, s):
    x_0 = np.concatenate((np.array(s), np.array(0).reshape(1)), axis=None)
    x_1 = np.concatenate((np.array(s), np.array(1).reshape(1)), axis=None)
    q_0 = Q.forward(torch.Tensor(x_0).to(device))
    q_1 = Q.forward(torch.Tensor(x_1).to(device))
    if q_0 > q_1:
        return 0
    else:
        return 1


######### one shot Q learning ########

params = {"batch_size": 640, "shuffle": True, "num_workers": 4}
max_epoch = 3000
if __name__ == "__main__":
    for loop in range(1):
        ####### collect data #########

        print("##########  Loop :", loop)
        s = np.zeros((4, 1))
        step = 0
        record = []
        record_list = []
        episode = 0
        t_step = np.int64(0)
        total_r = 0
        act_1 = np.int64(0)
        last_observation = np.zeros((4))
        max_step = np.int64(1e6)
        for i in range(max_step):
            # env.render()
            if loop == 0:
                action = env.action_space.sample()  # random action
            else:
                if step == 0:
                    action = env.action_space.sample()
                    last_observation = np.zeros((4))
                else:
                    if np.random.rand() < 1.0 / (loop + 3):
                        action = env.action_space.sample()
                    else:
                        action = choose_action(Q, s)
            step += 1
            act_1 += action
            observation, reward, done, info = env.step(action)
            s = observation
            total_r += reward
            record.append(
                [last_observation, observation, action, reward, total_r, not done]
            )
            last_observation = observation.copy()
            if done:
                t_step += step
                episode += 1
                step = 0
                record_list.append(record.copy())
                record.clear()
                env.reset()
                total_r = 0
        score = t_step / episode
        print("average steps:{:.4f}".format(t_step / episode))
        print("average action:{:.4f}".format(act_1 / max_step))
        env.close()
        S = Sampler(record_list)
        training_generator = data.DataLoader(S, **params)

        ####### train Q network #########
        try:
            for epoch in range(max_epoch):
                total_loss = 0.0
                iter = 0
                for xr, y in training_generator:
                    iter += 1
                    y = y.view(-1, 1).to(device, dtype=torch.float)
                    x = np.concatenate(
                        (np.array(xr[0]), np.array(xr[2]).reshape(-1, 1)), axis=1
                    )
                    out = Q.forward(torch.Tensor(x).to(device, dtype=torch.float)).view(
                        -1, 1
                    )
                    loss = criterion(y, out)
                    total_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print("Epoch {}: {:.4f}".format(epoch, total_loss / iter))
        except (KeyboardInterrupt, SystemExit):
            model_file_name = (
                "q_net_v1_" + str(len(record_list)) + "_" + str(loop) + ".pth"
            )
            torch.save(Q.state_dict(), "./models/" + model_file_name)
            print("Model file: " + model_file_name + " saved.")

    model_file_name = "q_net_v1_" + str(len(record_list)) + "_" + str(loop) + ".pth"
    torch.save(Q.state_dict(), "./models/" + model_file_name)
    print("Model file: " + model_file_name + " saved.")
