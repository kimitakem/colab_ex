from sympy import symbols, Matrix, sin, cos, diff
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, Particle
from sympy.physics.mechanics import Lagrangian, LagrangesMethod
from sympy.physics.mechanics import mprint

t = symbols('t')  # 時間
m1, m2, l, g = symbols('m1 m2 l g')  # パラメータ
p, theta = dynamicsymbols('p theta')  # 一般化座標
F = dynamicsymbols('F')  # 外力

q = Matrix([p, theta])  # 座標ベクトル
qd = q.diff(t)  # 座標の時間微分

N = ReferenceFrame('N')  # 参照座標系

# 質点1（台車）の質点、位置
P1 = Point('P1')
x1 = p  # 質点1の位置
y1 = 0
vx1 = diff(x1, t)  # x1.diff(t)?でもOK?
vy1 = diff(y1, t)
P1.set_vel(N, vx1 * N.x + vy1 * N.y)

Pa1 = Particle('Pa1', P1, m1)
Pa1.potential_energy = m1 * g * y1

# 質点2（振子）の質点、位置
P2 = Point('P2')
x2 = p - l * sin(theta)
y2 = cos(theta)
vx2 = diff(x2, t)
vy2 = diff(y2, t)
P2.set_vel(N, vx2 * N.x + vy2 * N.y)

Pa2 = Particle('Pa2', P2, m2)
Pa2.potential_energy = m1 * g * y2

# 外力
fl = [(P1, F*N.x), (P2, 0*N.x)]

# 運動方程式
LL = Lagrangian(N, Pa1, Pa2)
LM = LagrangesMethod(LL, q, forcelist=fl, frame=N)

eom = LM.form_lagranges_equations().simplify()
f = LM.rhs().simplify()

# 線形化
linearlizer = LM.to_linearizer(q_ind=q, qd_ind=qd)
op_point = {p: 0, theta: 0, p.diff(t): 0, theta.diff(t): 0}
A, B = linearlizer.linearize(A_and_B=True, op_point=op_point)

mprint(A)
mprint(B)