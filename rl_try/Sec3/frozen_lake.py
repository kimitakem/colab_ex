import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})


def show_q_value(q: dict) -> None:
    """
    Q関数の値を画面上に表示する
    :param q: Q関数、dictであるとする
    :return: なし
    """

    env = gym.make("FrozenLake-v0")
    nrow = env.unwrapped.nrow  # 4
    ncol = env.unwrapped.ncol  # 4
    state_size = 3
    q_nrow = nrow * state_size  # 4 * 3
    q_ncol = ncol * state_size  # 4 * 3
    reward_map = np.zeros((q_nrow, q_ncol))

    for r in range(nrow):
        for c in range(ncol):
            s = r * nrow + c  # 状態の番号（マスの番号に対応）

            state_exist = False
            if isinstance(q, dict) and s in q:
                state_exist = True

            if state_exist:
                _r = 1 + (nrow - 1 - r) * state_size
                _c = c * state_size + 1
                reward_map[_r][_c - 1] = q[s][0]  # 左への移動：インデックス0
                reward_map[_r - 1][_c] = q[s][1]  # 下への移動：インデックス1
                reward_map[_r][_c + 1] = q[s][2]  # 右への移動：インデックス2
                reward_map[_r + 1][_c] = q[s][3]  # 上への移動：インデックス3

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(reward_map, cmap=cm.RdYlGn, interpolation="none",
               vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()


if __name__ == "__main__":
    show_q_value({0: [1, 2, 3, 4],
                  1: [5, 2, 3, 4],
                  4: [2, 2, 3, 1]})

