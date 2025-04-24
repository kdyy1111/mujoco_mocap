import mujoco
import mujoco.viewer
import numpy as np
import time
from mocap_move import key_callback_data

from constants import START_ARM_POSE

def main():
    # Load the mujoco model basic.xml
    model = mujoco.MjModel.from_xml_path('panda_robot_scene.xml') # panda_robot_scene.xml
    data = mujoco.MjData(model)

    data.qpos[:9] = START_ARM_POSE

    def key_callback(key):
        key_callback_data(key, data)

    a = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            # print(data.ctrl)
            time.sleep(0.02)
            viewer.sync()

            # a=a+0.1
            # b = np.sin(a)/10
            # print(b)
            # np.copyto(data.mocap_pos[0], [3.99068696e-01+b, 4.94177457e-08, 4.17865547e-01])
            # joint_name = "hand"
            # joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            # body_id = model.jnt_bodyid[joint_id]
            # print("Joint positions:", data.xpos[body_id])
            # print(data.qpos)

if __name__ == '__main__':
    main()
