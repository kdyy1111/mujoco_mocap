import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        # self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        # if self.right_trajectory[0]['t'] == self.step_count:
        #     self.curr_right_waypoint = self.right_trajectory.pop(0)
        # next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        # right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            # right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        # action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left]) #, action_right


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        # init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        
        box_info = np.array(ts_first.observation['env_state'])
        red_box_xyz = box_info[:3]
        box_quat = box_info[3:]
        blue_box_xyz = box_info[7:10]
        print("red: ",red_box_xyz, "blue: ", blue_box_xyz)
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90) # meet_left_quat.elements

        meet_xyz = np.array([3.99068696e-01, 4.94177457e-08, 0.2])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": red_box_xyz + np.array([0, 0, 0.2]), "quat": np.array([1, 0, 0, 0]), "gripper": 1}, # approach meet position
            {"t": 200, "xyz": red_box_xyz + np.array([0, 0, 0.07]), "quat": np.array([1, 0, 0, 0]), "gripper": 1}, # move to meet position
            {"t": 220, "xyz": red_box_xyz + np.array([0, 0, 0.07]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # close gripper
            {"t": 270, "xyz": red_box_xyz + np.array([0, 0, 0.2]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 350, "xyz": blue_box_xyz + np.array([0, 0, 0.2]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
            {"t": 370, "xyz": blue_box_xyz + np.array([0, 0, 0.115]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
            {"t": 400, "xyz": blue_box_xyz + np.array([0, 0, 0.115]), "quat": np.array([1, 0, 0, 0]), "gripper": 1}, # stay
        ]

        # self.left_trajectory = [
        #     {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        #     {"t": 60, "xyz": red_box_xyz + np.array([0, 0, 0.2]), "quat": np.array([1, 0, 0, 0]), "gripper": 1}, # approach meet position
        #     {"t": 130, "xyz": red_box_xyz + np.array([0, 0, 0.08]), "quat": np.array([1, 0, 0, 0]), "gripper": 1}, # move to meet position
        #     {"t": 180, "xyz": red_box_xyz + np.array([0, 0, 0.08]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # close gripper
        #     {"t": 220, "xyz": red_box_xyz + np.array([0, 0, 0.2]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
        #     {"t": 280, "xyz": blue_box_xyz + np.array([0, 0, 0.2]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
        #     {"t": 320, "xyz": blue_box_xyz + np.array([0, 0, 0.125]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},
        #     {"t": 360, "xyz": blue_box_xyz + np.array([0, 0, 0.125]), "quat": np.array([1, 0, 0, 0]), "gripper": 1},
        #     {"t": 400, "xyz": blue_box_xyz + np.array([0, 0, 0.125]), "quat": np.array([1, 0, 0, 0]), "gripper": 1}, # stay
        # ]

        # self.right_trajectory = [
        #     {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
        #     {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
        #     {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
        #     {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
        #     {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
        #     {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
        #     {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
        #     {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
        #     {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        # ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['top']) # angle
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['top']) # angle
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

