import mujoco
import mujoco.viewer

from mocap_move import key_callback_data


def main():
    # Load the mujoco model basic.xml
    model = mujoco.MjModel.from_xml_path('base_scene.xml')
    data = mujoco.MjData(model)
    # data.qpos[:9] = [-0.466832, 1.20202, 0.939347, 0.939147, -0.884983, -1.80357, -0.266836, 0.01, 0.01]
    def key_callback(key):
        key_callback_data(key, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == '__main__':
    main()
