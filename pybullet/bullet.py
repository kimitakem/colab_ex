# pybulletで作成した環境を制御する
import pybullet as p
import pybullet_data
from time import sleep


def showAllJointInfo():
    joint_info_list = []
    joint_state_list = []
    for i in range(p.getNumJoints(cartId)):
        joint_info_list.append(p.getJointInfo(cartId, i))
        joint_state_list.append(p.getJointState(cartId, i))
    joint_info_index = [["jointIndex", "インデックス"],
                        ["jointName", "名称（URDFファイルで指定されているもの）"],
                        ["jointType", "ジョイントの形態（位置や速度の次元はここからわかる）"],
                        ["qIndex", "the first position index in the positional state variables for this body"],
                        ["uIndex", "the first velocity index in the velocity state variables for this body"],
                        ["flags", "予約されている"],
                        ["jointDamping", "ダンピング係数（URDFファイルで指定されいるもの）"],
                        ["jointFriction", "摩擦係数（URDFファイルで指定されいるもの）"],
                        ["jointLowerLimit", "位置の下限（SLIDERやREVOLUTEのジョイントに対する）"],
                        ["jointUpperLimit", "位置の上限（SLIDERやREVOLUTEのジョイントに対する）"],
                        ["jointMaxForce", "力の最大値（URDFで指定されているもの）/ 注：この値は自動的には使われない。'setJointMotorControl2'で指定可能"],
                        ["jointMaxVelocity", "速度の最大値（URDFで指定されているもの）/ 注：現時点では制御コマンドに使われていない"],
                        ["linkName", "リンクの名称（URDFで指定されているもの）"],
                        ["jointAxis", "ローカル座標系での回転軸。JOINT_FIXEDでは無視される。"],
                        ["parentFramePos", "ペアレント座標系でのジョイント位置"],
                        ["parentFrameOrn", "ペアレント座標系でのジョイント角度"],
                        ["parentIndex", "ペアレントリンクのインデックス（ベースであれば-1）"]
                        ]
    joint_state_index = [["jointPosition", "ジョイントの位置"],
                         ["jointVelocity", "ジョイントの速度"],
                         ["jointReactionForces", "トルクセンサーが有効になっている場合の反力"],
                         ["appliedJointMotorTorque", "直前のstepSimulationでのモータートルク。VELOCITY_CONTROLおよびPOSITION_CONTROLの時のみ有効。"
                                                     "TORQUE_CONTROLを使っている場合は、この値は与えている力と同じになる"]
                         ]

    constraint_state_index = [["parentBodyUniqueId", "ペアレントボディのユニークID"],
                              ["parentJointIndex", "ペアレント関節のインデックス"],
                              ["childBodyUniqueId", ""]
                              ["childLinkIndex", ""],
                              ["jointAxis", ""],
                              ["jointPivotInParent", ""],
                              ["jointPivotInChild", ""],
                              ["jointFrameOrientationParent", ""],
                              ["jointFrameOrientationChild", ""],
                              ["maxAppliedForce", ""]
                              ]


    for j in range(len(joint_info_index)):
        raw = joint_info_index[j] + [joint_info_list[i][j] for i in range(p.getNumJoints(cartId))]
        print(raw)
    for k in range(len(joint_state_index)):
        raw = joint_state_index[k] + [joint_state_list[i][k] for i in range(p.getNumJoints(cartId))]
        print(raw)

# シミュレータに接続する
physicsClient = p.connect(p.GUI)

# データを探索するパスを追加する
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
cartId = p.loadURDF("cartpole.urdf", cubeStartPos, cubeStartOrientation)
cubePos, cubeOrn = p.getBasePositionAndOrientation(cartId)

p.changeDynamics(cartId, -1, linearDamping=11, angularDamping=12)
p.changeDynamics(cartId, 0, linearDamping=1, angularDamping=1)
p.changeDynamics(cartId, 1, linearDamping=1, angularDamping=1)

p.setJointMotorControl2(cartId, 0, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(cartId, 1, p.VELOCITY_CONTROL, force=0)

#p.resetJointState(cartId, 0, 0.01, 0.02)
p.resetJointState(cartId, 1, 0.03, 0.04)

print("=== Confirm Joint Info ===")
showAllJointInfo()

p.setGravity(0, 0, -0.1)


maxStep = 2400

while 1:
    print("go 0.1")
    p.setJointMotorControl2(cartId, 1, p.POSITION_CONTROL, force=.1, targetPosition=0.1)
    for s in range(maxStep):
        p.stepSimulation()

    print("go -0.1")
    p.setJointMotorControl2(cartId, 1, p.POSITION_CONTROL, force=.1, targetPosition=-0.1)
    for s in range(maxStep):
        p.stepSimulation()


print("step finished")

