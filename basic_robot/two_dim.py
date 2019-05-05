import sympy as sym
from sympy import Derivative, Eq, symbols, sin, cos, pi
from sympy.physics.vector import Point, Vector, dynamicsymbols, time_derivative
from matplotlib import animation as ani
from matplotlib import pyplot as plt
from sympy.physics.mechanics import Lagrangian, LagrangesMethod
from sympy.physics.mechanics import ReferenceFrame, Point, Particle
import numpy as np


class TwoDimArm(object):
    def __init__(self):
        self.m_1, self.m_2 = symbols("m1, m2")  # リンクの質量
        self.l_1, self.l_2 = symbols("l1, l2")  # リンクの長さ

        self.im_1, self.im_2 = 1., 2.
        self.il_1, self.il_2 = 3., 4.

        self.q_1, self.q_2 = dynamicsymbols("q1 q2")  # リンクの回転角

    def link_kinematics(self, iq_1, iq_2):
        # リンク1の先端位置
        x_1 = self.l_1 * cos(self.q_1)
        y_1 = self.l_1 * sin(self.q_1)

        # リンク2の先端位置
        x_2 = self.l_1 * cos(self.q_1) + self.l_2 * cos(self.q_1 + self.q_2)
        y_2 = self.l_1 * sin(self.q_1) + self.l_2 * sin(self.q_1 + self.q_2)

        instance_list = {self.l_1: self.il_1, self.l_2: self.il_2,
                         self.q_1: iq_1, self.q_2: iq_2
                         }

        ix_1 = x_1.subs(instance_list).evalf()
        iy_1 = y_1.subs(instance_list).evalf()

        ix_2 = x_2.subs(instance_list).evalf()
        iy_2 = y_2.subs(instance_list).evalf()

        return ix_1, iy_1, ix_2, iy_2

    def inverse_kinema(self, ix_2, iy_2):
        pass


td = TwoDimArm()
print(td.link_kinematics(0.2, 0.2))

import pdb
pdb.set_trace()


l_g1, l_g2 = symbols("lg1 lg2")  # 関節軸からリンクの重心位置までの距離


# リンク2の先端位置
x_2 = l_1 * cos(q_1) + l_2 * cos(q_1 + q_2)
y_2 = l_1 * sin(q_1) + l_2 * sin(q_1 + q_2)

# 具体的な位置の算出
q_1i = 0.5
q_2i = 0.3
l_1i = 1.
l_2i = 2.

#### 逆運動学 ####


p = np.array([0, 0])
p_d = np.array([1, 2])

