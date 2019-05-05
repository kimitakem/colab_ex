# ボールの運動方程式を記述する
import sympy as sym
from sympy import Derivative, Eq, symbols, sin, cos, pi
from sympy.physics.vector import Point, Vector, dynamicsymbols, time_derivative
from matplotlib import animation as ani
from matplotlib import pyplot as plt
from sympy.physics.mechanics import Lagrangian, LagrangesMethod
from sympy.physics.mechanics import ReferenceFrame, Point, Particle

q = dynamicsymbols("q")     # 角度
length = symbols("length")  # 振子の長さ

t = symbols("t")  # 時間
theta = pi * sin(t)  # 角度の時間変異

x = length * cos(q)  # x座標
y = length * sin(q)  # y座標

# v_x = symbols('v_x')  # x方向の速度
# v_y = symbols('v_y')  # y方向の速度
v_x = x.diff(t)
v_y = y.diff(t)

N = ReferenceFrame('N')  # 参照座標系
pt = Point("P")
pt.set_vel(N, v_x * N.x + v_y * N.y)

m = symbols("m")  # 質量
pcl = Particle("mass", pt, m)

f_x, f_y = dynamicsymbols("f_x f_y")


LL = Lagrangian(N, pcl)


LM = LagrangesMethod(LL, [q])
eom = LM.form_lagranges_equations().simplify()
rhs = LM.rhs()

import pdb
pdb.set_trace()



def get_path(t_i):
    """
    :param t_i: 時間
    :return: 時間t_iでのx, y座標
    """
    x_t = x.subs({q: theta, t: t_i, length: 1}).evalf()
    y_t = y.subs({q: theta, t: t_i, length: 1}).evalf()
    return x_t, y_t


def calc_dq(t_i):
    dq = Derivative(q, t)
    dq_t = dq.subs({q: theta, t: t_i, length: 1}).evalf()
    return dq_t


fig = plt.figure()
plt.axes().set_aspect('equal')
plt.axes().set_xlim(-1, 1)
plt.axes().set_ylim(-1, 1)

ims = []
dq_log = []
t_log = []
for n in range(100):
    dt = 0.1
    t_n = dt * n
    x_t, y_t = get_path(t_n)
    img = plt.plot([0, x_t], [0, y_t], color='red')
    ims.append(img)
    t_log.append(t_n)
    dq_log.append(calc_dq(t_n))

import pdb
pdb.set_trace()

a = ani.ArtistAnimation(fig, ims, interval=100)
plt.show()

plt.clf()
plt.scatter(t_log, dq_log)
plt.show()

e_x = Eq(m * Derivative(x, t, 2), 0)
