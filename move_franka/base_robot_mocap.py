import mujoco
import mujoco.viewer
import time
import numpy as np
from mocap_move import key_callback_data


def main():
    # Load the mujoco model basic.xml
    model = mujoco.MjModel.from_xml_path('base_scene.xml')
    data = mujoco.MjData(model)
    data.qpos[:22] = [-9.34612e-05, -5.75221e-06, 0.1561732, 1, 5.81129e-05, -0.000731868, 0.000297367, 0.000589517, 0.00427167, 0.000753209, 
                -1.97983, -0.00550771, 2.02638, -2.18097, -2.38219e-06, 2.65126e-06, -0.0343745, 0.0850201, 0, 0, 0, 0]

    def key_callback(key):
        key_callback_data(key, data)

    from scipy.spatial.transform import Rotation as R

    def xyaxes_to_quat_mjcf(xyaxes):
        """
        MuJoCo XML xyaxes (6개 실수) → MuJoCo 포맷 쿼터니언 [w x y z] 반환.

        Parameters:
        - xyaxes: (6,) 리스트 or np.ndarray [x_axis (3), y_axis (3)]

        Returns:
        - quat_mj: (4,) 리스트 [w x y z] MuJoCo 포맷
        """
        x_axis = np.array(xyaxes[:3])
        y_axis = np.array(xyaxes[3:])
        z_axis = np.cross(x_axis, y_axis)

        # 정규 직교화 (optional but recommended)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)

        R_mat = np.column_stack([x_axis, y_axis, z_axis])
        rot = R.from_matrix(R_mat)
        quat_xyzw = rot.as_quat()  # [x, y, z, w]
        quat_mj = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # MuJoCo: [w x y z]
        return quat_mj

    def camera_world_to_body_relative(pos_A, quat_A, pos_B, quat_B):
        """
        월드 기준 pose A를 기준으로 pose B를 상대 pose로 변환.

        Parameters:
        - pos_A, pos_B: (3,) np.ndarray, 위치 (world 좌표계)
        - quat_A, quat_B: (4,) np.ndarray, 회전 (MuJoCo 포맷 [w x y z])

        Returns:
        - rel_pos: (3,) np.ndarray, A 기준 상대 위치
        - rel_quat: (4,) np.ndarray, A 기준 상대 회전 (MuJoCo 포맷 [w x y z])
        """
        # 회전: MuJoCo → SciPy ([x y z w])
        r_A = R.from_quat([quat_A[1], quat_A[2], quat_A[3], quat_A[0]])
        r_B = R.from_quat([quat_B[1], quat_B[2], quat_B[3], quat_B[0]])

        # 상대 위치
        rel_pos = r_A.inv().apply(pos_B - pos_A)

        # 상대 회전
        rel_rot = r_A.inv() * r_B
        q_rel = rel_rot.as_quat()  # [x y z w]
        q_rel_mj = [q_rel[3], q_rel[0], q_rel[1], q_rel[2]]  # MuJoCo 포맷

        return rel_pos, q_rel_mj
    
    class EMAFilter:
        def __init__(self, alpha=0.2):
            self.alpha = alpha
            self.lin_vel = None
            self.ang_vel = None

        def update(self, lin, ang):
            if self.lin_vel is None or self.ang_vel is None:
                self.lin_vel = lin
                self.ang_vel = ang
            else:
                self.lin_vel = self.alpha * lin + (1 - self.alpha) * self.lin_vel
                self.ang_vel = self.alpha * ang + (1 - self.alpha) * self.ang_vel
            return self.lin_vel, self.ang_vel
    
    def get_base_vel_from_mujoco(model, data, ema_filter: EMAFilter = None):
        # left_qvel = data.qvel[model.joint_name2id("left_wheel_joint")]
        # right_qvel = data.qvel[model.joint_name2id("right_wheel_joint")]

        # # 오른쪽 바퀴 방향 보정 (필요시)
        # # right_qvel *= -1

        # r = 0.06   # 바퀴 반지름 [m]
        # b = 0.64   # 바퀴 중심 간 거리 [m]

        # v = (r / 2) * (left_qvel + right_qvel)
        # omega = (r / b) * (right_qvel - left_qvel)

        # 조인트 ID 획득
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")

        if left_id < 0 or right_id < 0:
            raise ValueError("joint name not found in model")

        # qvel 배열에서 각속도 추출
        left_qvel = data.qvel[left_id]
        right_qvel = data.qvel[right_id]

        # 오른쪽 바퀴 방향 보정 (필요 시)
        right_qvel *= -1

        # 모델 파라미터 설정
        wheel_radius = 0.06   # [m]
        wheel_base = 0.64     # 바퀴 중심 간 거리 [m]

        # 선속도 및 각속도 계산
        linear_velocity = wheel_radius * (left_qvel + right_qvel) / 2.0
        angular_velocity = wheel_radius * (right_qvel - left_qvel) / wheel_base

        # EMA 필터 적용
        if ema_filter is not None:
            linear_velocity, angular_velocity = ema_filter.update(linear_velocity, angular_velocity)


        return np.array([round(linear_velocity, 3), round(angular_velocity, 3)])  # [linear_velocity, angular_velocity]
    
    ema = EMAFilter(alpha=0.1)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            ## Camera pose mujoco -> xml
            # joint_name = "hand"
            # body_name = "hand"
            # joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            # body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            # qpos_index = model.jnt_qposadr[joint_id]
            # position = data.xpos[body_id]  # shape: (3,)
            # quat = data.xquat[body_id]
            # cam_quat = xyaxes_to_quat_mjcf([-0.004, 1.000, 0.000, -0.882, -0.004, 0.471])
            # cam_pos = camera_world_to_body_relative([0.822, -0.002, 0.783], cam_quat, position, quat)
            # print("camera_world_to_body_relative", cam_pos)

            lin_vel, ang_vel = get_base_vel_from_mujoco(model, data, ema_filter=ema)
            print("lin_vel",lin_vel)
            print("ang_vel", ang_vel)
            time.sleep(0.002)
            viewer.sync()


if __name__ == '__main__':
    main()
