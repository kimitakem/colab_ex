import numpy as np
import matplotlib.pyplot as plt


class ELAgent:
    def __init__(self, epsilon: float):
        """
        Epsilon-Greedyで動くエージェント
        :param epsilon: ランダム探索を行う確率
        """
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s: int, actions: list):
        """
        方策の実装
        :param s: 状態
        :param actions: 行動
        :return: Q[s]はlen(a)の配列で、epsilon出ないときはQ[s]が最大となるindexを行動として返す
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        """
        報酬のログを初期化する
        """
        self.reward_log = []

    def log(self, reward: float):
        """
        報酬をログに記録する
        :param reward: 報酬の値
        :return: None
        """
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        """

        :param interval:
        :param episode:
        :return:
        """
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {}, average reward is {} (+/-{}).".format(
                episode, mean, std))

        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i: (i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()
