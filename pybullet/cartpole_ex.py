import gym
from gym import spaces
import pybullet as p
import pybullet_data
import math
import numpy as np
from os import path


class MyCartPole(gym.Env):
    def __init__(self):
        p.connect(p.DIRECT)  #
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 12°
        self.x_threshold = 0.4
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.force_mag = 10  # force magnitude

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        p.resetSimulation()
        self.cartpole = p.loadURDF(path.join(pybullet_data.getDataPath(), "cartpole.urdf", [0, 0, 0]))



        pass

    def step(self, action):
        # 力の設定：actionが1の時は正の方向の力、そうでないときは負の力
        force = self.force_mag if action==1 else -self.force_mag
        p.setJointMotorControl2(self.cart)

        pass



    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self):
        pass










# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv

from baselines import deepq





def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = CartPoleBulletEnv(renders=True)
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
