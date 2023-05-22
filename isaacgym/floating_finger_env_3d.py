import time

import gym
import numpy as np
import cv2
import copy
import misc_utils as mu
import pybullet as p
import pybullet_utils as pu
import math
import os
from math import radians
import itertools
from gym.utils import seeding
import pyrender
import trimesh
import rtde_control
import rtde_receive
import rospy
from std_msgs.msg import Bool
from tactile_sensor.msg import Contact
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion


class FloatingFingerEnv3D(gym.Env):
    def __init__(self,
                 max_ep_len=2000,
                 max_x=0.3,
                 max_y=0.3,
                 min_z=0.05,
                 max_z=0.3,
                 finger_urdf_path='assets/simplified_finger_urdf_origin_on_tip_36mm/urdf/roam_distal.urdf',
                 dataset='ycb_objects',
                 rot_step_size=15,
                 object_scale=1.0,
                 finger_height_index=3,
                 start_on_border=True,
                 reward_type='sparse',
                 reward_scale=1.0,
                 reward_contact=False,
                 reward_contact_magnitude=0.01,
                 num_orientations=20,
                 dof6_orientation=False,
                 translate_range=0.01,
                 translate_range_z=0,
                 render_pybullet=False,
                 render_ob=False,
                 render_spheres=False,
                 render_normals=False,
                 debug=False,
                 use_correctness=False,
                 exp_knob=None,
                 threshold=0.98,
                 sensor_noise=0,
                 localization_noise=0,
                 localization_noise_model='uniform',
                 max_non_collision_steps=1000,
                 # add_contact_normals = False,
                 keep_finger_downwards=False,
                 traj_pose_steps=10,
                 trans_step_size=0.01,
                 penalize_cylinder=False,
                 env_id=0,
                 real=False,
                 robot_ip='192.168.0.166',
                 workspace_origin_position=[-0.588, 0.128, -0.0395]):
        self.seed()

        if dof6_orientation:
            finger_height_index = 25
        # the height of the finger is min_z + finger_height_index * trans_step_size
        self.finger_height_index = finger_height_index
        self.r = 0.01875
        self.r_times_sqrt2 = self.r * np.sqrt(2)

        self.keep_finger_downwards = keep_finger_downwards
        self.traj_pose_steps = traj_pose_steps
        self.penalize_cylinder = penalize_cylinder

        if self.penalize_cylinder:
            self.contact_on_hemisphere = True

        self.max_ep_len = max_ep_len
        self.max_x = max_x
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z
        self.trans_step_size = trans_step_size
        self.rot_step_size = rot_step_size
        # discretize the workspace
        self.max_x_idx = round(self.max_x / self.trans_step_size)
        self.max_y_idx = round(self.max_y / self.trans_step_size)
        self.max_z_idx = round((self.max_z - self.min_z) / self.trans_step_size)
        self.finger_urdf_path = finger_urdf_path
        self.env_id = env_id
        self.object_scale = object_scale
        self.use_correctness = use_correctness
        # self.add_contact_normals = add_contact_normals
        self.num_classes = 10
        self.move_dim = len(mu.move_map_3d)
        # self.action_space = gym.spaces.Discrete(self.action_dim)
        self.action_space = gym.spaces.Dict({"move": gym.spaces.Discrete(self.move_dim),
                                             "prediction": gym.spaces.Discrete(self.num_classes),
                                             "probs": gym.spaces.Box(low=0, high=1, shape=(self.num_classes,)),
                                             "max_prob": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                             "done": gym.spaces.Discrete(2)})
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.object)
        self.start_on_border = start_on_border
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.reward_contact = reward_contact
        self.reward_contact_magnitude = reward_contact_magnitude
        self.exp_knob = exp_knob
        self.prob_mapping_function = mu.get_prob_mapping_function(reward_type, 0.1, threshold, exp_knob)
        self.num_orientations = num_orientations
        self.dof6_orientation = dof6_orientation
        self.translate_range = translate_range
        self.translate_range_z = translate_range_z
        self.render_ob = render_ob
        self.render_pybullet = render_pybullet
        self.render_spheres = render_spheres
        self.render_normals = render_normals
        self.normal_length = 0.1
        self.debug = debug
        self.finger_initial_z_loc = self.min_z + self.finger_height_index * self.trans_step_size
        self.finger_initial_loc = [[0, 0, self.finger_height_index],
                                   [round(180 / self.rot_step_size), round(300 / self.rot_step_size),
                                    round(45 / self.rot_step_size)]]
        # self.finger_initial_loc = [[15, 15, self.finger_height_index],
        #                            [round(180 / self.rot_step_size), 0, 0]]
        self.finger_corner_loc_min_x_max_y = [[0, self.max_y_idx, self.finger_height_index],
                                              [round(180 / self.rot_step_size), round(300 / self.rot_step_size),
                                               round(315 / self.rot_step_size)]]

        self.finger_initial_pose = self.get_pose_from_loc(self.finger_initial_loc)
        self.polygon_initial_quaternion = [0, 0, 0, 1]
        self.waitlist_position = [-1, -1, 0]
        self.sensor_noise = sensor_noise
        self.localization_noise = localization_noise
        self.localization_noise_model = localization_noise_model
        self.real = real

        print("location noise:{}, noise model:{}".format(self.localization_noise, self.localization_noise_model))

        if self.localization_noise > 0:
            self.localization_noise_std = np.sqrt(
                1 / 3) * self.localization_noise  # from the formula: 3 * sigma^2 = (localization_noise)^2

        self.scene = None
        self.points_node = None  # node for saving the point clouds for pyrender
        self.viewer = None  # the viewer for visualizing the point clouds

        # step related info
        self.gt_grids = None
        self.rendered_occupancy = False
        self.current_step = 0
        self.steps_non_collision = 0
        self.last_position = None
        self.current_loc = None
        self.polygon_id = None
        self.angle = None
        self.polygon_initial_position = None
        # y is the world y axis (in simulation), y is the width of the image, the second axis of the numpy array
        # x is the world x axis (in simulation), x is the height of the image, the first axis of the numpy array
        self.occupancy_grid = np.full((1, self.max_x_idx, self.max_y_idx), mu.unexplored, dtype=np.uint8)
        self.points = set()  # points is a set of tuples: {(x1, y1, z1), (x2, y2, z2), ...}
        self.all_points = set()  # a set of tuples: {((x1, y1, z1), 1), ((x2, y2, z2), 0), ...}
        self.sphere_bodies = []
        self.normal_ids = []

        self.ob = None
        self.done = None
        self.info = None
        self.reward = 0
        self.success = None
        self.discover = None
        self.max_prob = 0.1
        self.initial_explored_pixel = None
        self.finger_point = None

        self.dataset = dataset
        self.dataset_path = os.path.join('assets', 'datasets', dataset)
        self.object_urdf_folder = os.path.join(self.dataset_path, 'urdfs')

        self.client_id = pu.configure_pybullet(rendering=render_pybullet, debug=self.debug,
                                               target=(max_x / 2, max_y / 2, 0.05), dist=0.6)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=0,
                                     cameraTargetPosition=(max_x / 2, max_y / 2, 0.05),
                                     physicsClientId=self.client_id)
        time.sleep(2)
        # if this part goes to reset, it becomes 100X slower!
        # p.resetSimulation(physicsClientId=self.client_id)
        self.finger = FloatingFingerController([self.finger_initial_pose[0], self.finger_initial_pose[1]],
                                               self.finger_urdf_path,
                                               self.client_id,
                                               self.debug)
        if not self.real:
            # load all the polygons at a waiting location
            self.polygons = []  # list of pybullet object ids
            self.polygon_bodyinfos = []
            for i in range(self.num_classes):
                object_urdf_path = os.path.join(self.object_urdf_folder, f'{i}.urdf')
                object = p.loadURDF(object_urdf_path,
                                    basePosition=self.waitlist_position,
                                    baseOrientation=[0, 0, 0, 1],
                                    globalScaling=self.object_scale,
                                    useFixedBase=True,
                                    physicsClientId=self.client_id)
                self.polygons.append(object)

                polygon_name = p.getBodyInfo(object)[1].decode()  # get the string name of object
                object_mesh_path = os.path.join(os.path.dirname(self.object_urdf_folder), "meshes", polygon_name,
                                                "textured.obj")
                self.polygon_bodyinfos.append(trimesh.load(object_mesh_path).extents.tolist())

        self.max_non_collision_steps = max_non_collision_steps

        if self.keep_finger_downwards:
            poses_cube = self.legal_poses()
            print("poses_cube.sum()", poses_cube.sum())
            self.legal_actions = np.ones(12)

        if self.real:
            # This defines the position of workspace loc (0, 0, 0) in the robot frame
            self.workspace_origin_position = workspace_origin_position
            self.robot_ip = robot_ip
            # define the interfaces
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

            # ROS
            rospy.init_node('demo')
            self.collision_sub = rospy.Subscriber("/contact/collision", Bool, self.collision_callback)
            self.contact_sub = rospy.Subscriber("/contact/local/finger3", Contact, self.contact_callback)
            self.collision = False
            self.contact_pose = None

            self.finger_initial_pose_ws = self.get_pose_from_loc(self.finger_initial_loc)
            self.starting_pose_control = self.from_workspace_to_control_space(self.finger_initial_pose_ws)
            self.pre_starting_pose_control = copy.deepcopy(self.starting_pose_control)
            self.pre_starting_pose_control[2] += 0.1

            # move to an initial configuration
            self.initial_q = [-1.12, -1.34, 1.36, -1.60, -1.61, 0.11]
            self.rtde_c.moveJ(self.initial_q, speed=0.5, acceleration=0.5)

    def degree_2_radian(self, d):
        return [math.radians(i) for i in d]

    def calc_contact_normals(self, contact_position_on_finger):

        center_pos_in_tip = [0, 0, -self.r]
        center_ori_in_tip = [0, 0, 0, 1]
        tip_pos_in_world = [self.current_loc[0][0] * self.trans_step_size,
                            self.current_loc[0][1] * self.trans_step_size,
                            self.current_loc[0][2] * self.trans_step_size + self.min_z]
        tip_ori_in_world = p.getQuaternionFromEuler(
            self.degree_2_radian(np.array(self.current_loc[1]) * self.rot_step_size))

        center_pos_in_world, center_ori_in_world = p.multiplyTransforms(tip_pos_in_world, tip_ori_in_world,
                                                                        center_pos_in_tip, center_ori_in_tip)

        delta = np.array(center_pos_in_world) - np.array(contact_position_on_finger)
        delta_signs = (delta > 0) * 2.0 - 1
        contact_normals = delta ** 2
        contact_normals = contact_normals / contact_normals.sum()
        contact_normals = np.sqrt(contact_normals) * delta_signs

        return tuple(contact_normals)

    def check_collision(self, object_id=None):
        # For the 3D environment, we don't care about sensor noise flipping.

        object_id = object_id if object_id is not None else self.polygons[self.polygon_id]
        closest_points = pu.get_closest_potins(self.finger.id, object_id, 0, -1, -1, client=self.client_id)

        if len(closest_points) > 0:
            # This function returns at most 4 points. I will always use the first one
            point = closest_points[0]
            contact_position = point.positionOnB
            contact_position_on_finger = point.positionOnA

            if self.localization_noise > 0:
                if self.localization_noise_model == 'uniform':
                    # The first method: randomly sample points (not necessarily on the hemisphere surface)
                    rand_point = self.np_random.randn(3)
                    rand_point /= np.linalg.norm(rand_point)
                    rand_point *= self.np_random.uniform(0, self.localization_noise, 1)[0]
                    rand_point = np.array([rand_point[0], rand_point[1], rand_point[2]])
                    contact_position_with_localization_noise = tuple(np.array(contact_position_on_finger) + rand_point)
                    contact_normal = self.calc_contact_normals(contact_position_with_localization_noise)
                    #        print("contact normal", contact_normal)
                    return True, contact_position_with_localization_noise, contact_normal
            else:
                contact_normal = self.calc_contact_normals(contact_position_on_finger)
                return True, contact_position_on_finger, contact_normal
        else:
            return False, None, None

    def check_collision_real(self):
        if self.collision:
            return self.collision, self.contact_pose[0], self.normal_in_ws
        else:
            return False, None, None

    def step(self, action):
        print(action)
        move = action['move']
        prediction = action['prediction']
        max_prob = action['max_prob']
        probs = action['probs']
        done = action['done']

        num_explored = len(self.points)

        if self.steps_non_collision >= self.max_non_collision_steps:
            if self.start_on_border:
                # print("move towards center")
                # time.sleep(1)
                moves = self.generate_heuristic_moves_towards_center()
                num_moves = 0
                collision = False
                while not collision:
                    collision, contact_position, contact_normal = self.move(moves[num_moves])
                    num_moves += 1
                    self.current_step += 1
                    # time.sleep(0.2)
                self.steps_non_collision = 0
        else:
            collision, contact_position, contact_normal = self.move(move)

        self.current_step += 1
        self.success = self.check_success(prediction)
        self.done = done or self.current_step >= self.max_ep_len

        self.discover = True if len(self.points) > num_explored else False
        if self.discover and self.render_spheres:
            self.sphere_bodies.append(
                pu.draw_sphere_body(contact_position, radius=0.005, rgba_color=[1, 0, 0, 1], client=self.client_id))
        if self.discover and self.render_normals:
            self.normal_ids.append(pu.draw_line(start_pos=contact_position,
                                                end_pos=np.array(contact_position) + self.normal_length * np.array(contact_normal),
                                                rgb_color=(0, 0, 0),
                                                client=self.client_id))

        if not self.discover:
            self.steps_non_collision += 1

        if (not self.use_correctness and max_prob > self.max_prob) or \
                (self.use_correctness and max_prob > self.max_prob and prediction == self.polygon_id):
            old_mapped_prob = self.prob_mapping_function(self.max_prob)
            mapped_prob = self.prob_mapping_function(max_prob)
            self.reward = mapped_prob - old_mapped_prob
            self.max_prob = max_prob
        else:
            self.reward = 0

        if self.reward_contact and self.discover:
            self.reward += self.reward_contact_magnitude
        self.reward = self.reward * self.reward_scale

        if self.penalize_cylinder:
            if not self.contact_on_hemisphere:
                self.done = True
                self.reward = 0

                # the shape of the ob is (3, ), type is np.object

        self.ob = np.empty(4, dtype=np.object)
        self.ob[0] = list(zip(*list(self.all_points)))[0]
        self.ob[1] = list(zip(*list(self.all_points)))[1]
        self.ob[2] = list(zip(*list(self.all_points)))[2]
        self.ob[3] = copy.deepcopy(self.finger_point)

        self.info = {'discover': self.discover,
                     'contact_position': contact_position if self.discover else None,
                     'contact_normal': contact_normal if self.discover else None,
                     'num_points': len(self.points),
                     'num_gt': self.polygon_id,
                     'prediction': action['prediction'],
                     'success': self.success,
                     'angle': self.angle}
        if self.render_ob:
            self.render_points()
        return self.ob, self.reward, self.done, self.info

    def move(self, move):
        """ also return collision and collision position """
        goal_loc = self.compute_next_loc(self.current_loc, move)
        traj_poses = self.get_traj_poses(self.current_loc, goal_loc, step=self.traj_pose_steps)
        for idx, pose in enumerate(traj_poses):

            self.finger.set_pose_no_control(pose)
            if self.real:
                self.rtde_c.moveL(self.from_workspace_to_control_space(pose), speed=0.02, acceleration=0.05)

            collision, contact_position, contact_normal = \
                self.check_collision(self.polygons[self.polygon_id]) if not self.real else self.check_collision_real()

            if collision:
                self.finger.set_pose_no_control(traj_poses[0])
                if self.real:
                    self.rtde_c.moveL(self.from_workspace_to_control_space(traj_poses[0]), speed=0.02, acceleration=0.05)

                self.points.add(tuple(contact_position))
                self.all_points.add(tuple([tuple(contact_position), contact_normal, 1]))

                if self.penalize_cylinder:
                    self.contact_on_hemisphere = np.linalg.norm(
                        pose[0] - np.array(contact_position)) < self.r_times_sqrt2 + self.localization_noise + 0.01

                if self.keep_finger_downwards:
                    self.set_legal_actions()
                return True, tuple(contact_position), tuple(contact_normal)

        self.current_loc = goal_loc
        self.finger_point = tuple(pu.get_body_pos(self.finger.id, self.client_id))

        if self.keep_finger_downwards:
            self.set_legal_actions()
        return False, None, None

    def get_traj_poses(self, start_loc, goal_loc, step=5):
        start_pose = self.get_pose_from_loc(start_loc)
        goal_pose = self.get_pose_from_loc(goal_loc)

        start_pos, start_ori = start_pose[0], start_pose[1]
        goal_pos, goal_ori = goal_pose[0], goal_pose[1]

        # position trajectory
        traj_pos = np.linspace(start_pos, goal_pos, num=step + 1)

        # orientation trajectory
        processed_start_ori = copy.deepcopy(start_ori)
        # check if any of the angles is wrapped
        for i in range(3):
            if start_loc[1][i] == 0 and goal_loc[1][i] == 23:
                processed_start_ori[i] = 360
            elif start_loc[1][i] == 23 and goal_loc[1][i] == 0:
                processed_start_ori[i] = -15
            else:
                pass
        traj_ori = np.linspace(processed_start_ori, goal_ori, num=step + 1)

        traj_poses = []
        for pos, ori in zip(traj_pos, traj_ori):
            traj_poses.append([pos, ori])
        return traj_poses

    def reset(self, polygon_id=None, angle=None):
        if self.polygon_id is not None and not self.real:
            pu.set_point(self.polygons[self.polygon_id], self.waitlist_position, client=self.client_id)
        mu.draw_workspace([0, 0, self.min_z], [self.max_x, self.max_y, self.max_z])
        self.done = False
        self.info = None
        self.reward = 0
        self.success = False
        self.discover = True
        self.current_step = 0
        self.steps_non_collision = 0
        self.last_position = None
        self.max_prob = 0.1
        self.occupancy_grid = np.full((1, self.max_x_idx, self.max_y_idx), mu.unexplored, dtype=np.uint8)
        self.polygon_id = self.np_random.randint(low=0, high=10) if polygon_id is None else polygon_id
        self.finger.set_pose_no_control(self.finger_initial_pose)
        if self.penalize_cylinder:
            self.contact_on_hemisphere = True
        # different methods have different number of actions, calling the check_collision function different number of times,
        # then calling random choice different number of times
        # doing it this way make sure the sensor error happens at the same time (not the same location) for each episode across different models
        self.random_nums = self.np_random.uniform(size=3000)
        self.collision_cnt = 0
        self.points = set()  # points is a set of tuples
        self.all_points = set()
        # I am afriad I am gonna change a specific element of current_loc later
        self.current_loc = copy.deepcopy(self.finger_initial_loc)
        self.finger_point = tuple(pu.get_body_pos(self.finger.id, self.client_id))
        self.legal_actions = np.ones(12)

        # object orientation
        if angle is not None:
            self.angle = angle
        else:
            if self.num_orientations == -1:
                if self.dof6_orientation:
                    self.angle = [self.np_random.uniform(low=0, high=360),
                                  self.np_random.uniform(low=0, high=360),
                                  self.np_random.uniform(low=0, high=360)]
                else:
                    self.angle = [0, 0, self.np_random.uniform(low=0, high=360)]
            else:
                gap = int(360 / self.num_orientations)
                angles = [0 + i * gap for i in range(self.num_orientations)]
                if self.dof6_orientation:
                    angle_i = [self.np_random.choice(range(self.num_orientations)),
                               self.np_random.choice(range(self.num_orientations)),
                               self.np_random.choice(range(self.num_orientations))]
                    self.angle = [angles[angle_i[0]], angles[angle_i[1]], angles[angle_i[2]]]
                else:
                    angle_i = self.np_random.choice(range(self.num_orientations))
                    self.angle = [0, 0, angles[angle_i]]

        euler = (radians(self.angle[0]), radians(self.angle[1]), radians(self.angle[2]))

        self.polygon_initial_quaternion = pu.quaternion_from_euler(euler)

        if not self.real:
            # object position
            self.polygon_initial_position = self.sample_polygon_position()
            pu.set_pose(self.polygons[self.polygon_id], (self.polygon_initial_position, self.polygon_initial_quaternion),
                        client=self.client_id)

        # the finger initial location is guaranteed to be collision-free
        self.finger.set_pose_no_control(self.finger_initial_pose)

        if self.real:
            self.rtde_c.moveL(self.pre_starting_pose_control, speed=0.1, acceleration=0.5)
            self.rtde_c.moveL(self.starting_pose_control, speed=0.1, acceleration=0.5)

        if self.start_on_border:
            # always starts on boarder
            moves = self.generate_heuristic_moves()
            num_moves = 0
            collision = False
            while not collision:
                collision, _, _ = self.move(moves[num_moves])
                num_moves += 1
                if self.current_loc[0][0] == self.max_x_idx - 1 and self.current_loc[0][1] == self.max_y_idx - 1:
                    self.current_loc = self.finger_corner_loc_min_x_max_y

        # the shape of the ob is (3, ), type is an object
        self.ob = np.empty(4, dtype=np.object)
        # extract all the points location
        self.ob[0] = list(zip(*list(self.all_points)))[0]
        # extract all the features
        self.ob[1] = list(zip(*list(self.all_points)))[1]
        self.ob[2] = list(zip(*list(self.all_points)))[2]

        # extract the finger location
        self.ob[3] = copy.deepcopy(self.finger_point)

        self.initial_explored_pixel = np.count_nonzero(self.occupancy_grid != mu.unexplored)
        if self.render_ob:
            self.render_points()

        if self.render_spheres:
            if len(self.sphere_bodies) != 0:
                pu.remove_bodies(self.sphere_bodies, client=self.client_id)
                self.sphere_bodies = []
                # append the first collision point
            self.sphere_bodies.append(
                pu.draw_sphere_body(position=self.ob[0][0], radius=0.005, rgba_color=[1, 0, 0, 1],
                                    client=self.client_id))

        if self.render_normals:
            if len(self.normal_ids) != 0:
                pu.remove_markers(self.normal_ids)
                self.normal_ids = []
            self.normal_ids.append(pu.draw_line(start_pos=np.array(self.ob[0][0]),
                                                end_pos=np.array(self.ob[0][0]) + self.normal_length * np.array(self.ob[1][0]),
                                                rgb_color=(0, 0, 0),
                                                client=self.client_id))

        # print(f'env id: {self.env_id}\t client id: {self.client_id}\t polygon id: {self.polygon_id}')
        return self.ob

    def check_success(self, prediction):
        return prediction == self.polygon_id

    def get_pose_from_loc(self, loc):
        """
        loc is a list of lists: [[pos_loc], [ori_loc]]
        returned pose in euler angle degrees
        """
        pos = [loc[0][0] * self.trans_step_size, loc[0][1] * self.trans_step_size,
               self.min_z + loc[0][2] * self.trans_step_size]
        ori = [loc[1][0] * self.rot_step_size, loc[1][1] * self.rot_step_size, loc[1][2] * self.rot_step_size]
        return [pos, ori]

    def calculate_new_pose(self, move):
        current_position, current_orn = self.finger.get_pose()
        current_orn = self.finger_initial_quaternion
        if move == 0:
            # y + 1
            current_position[1] = current_position[1] + self.step_size \
                if current_position[1] + self.step_size <= self.max_y else current_position[1]
        elif move == 1:
            # x - 1
            current_position[0] = current_position[0] - self.step_size \
                if current_position[0] - self.step_size >= 0 else current_position[0]
        elif move == 2:
            # y - 1
            current_position[1] = current_position[1] - self.step_size \
                if current_position[1] - self.step_size >= 0 else current_position[1]
        elif move == 3:
            # x + 1
            current_position[0] = current_position[0] + self.step_size \
                if current_position[0] + self.step_size <= self.max_x else current_position[0]
        else:
            raise ValueError('unrecognized move')
        return [current_position, current_orn]

    def compute_next_loc(self, current_loc, move):
        next_loc = copy.deepcopy(current_loc)
        if move == mu.x_plus:
            next_loc[0][0] = current_loc[0][0] + 1 if current_loc[0][0] + 1 < self.max_x_idx else self.max_x_idx - 1
        elif move == mu.x_minus:
            next_loc[0][0] = current_loc[0][0] - 1 if current_loc[0][0] - 1 >= 0 else 0
        elif move == mu.y_plus:
            next_loc[0][1] = current_loc[0][1] + 1 if current_loc[0][1] + 1 < self.max_y_idx else self.max_y_idx - 1
        elif move == mu.y_minus:
            next_loc[0][1] = current_loc[0][1] - 1 if current_loc[0][1] - 1 >= 0 else 0
        elif move == mu.z_plus:
            next_loc[0][2] = current_loc[0][2] + 1 if current_loc[0][2] + 1 < self.max_z_idx else self.max_z_idx - 1
        elif move == mu.z_minus:
            next_loc[0][2] = current_loc[0][2] - 1 if current_loc[0][2] - 1 >= 0 else 0
        # For the angles, I am wrapping it for continuous rotation
        elif move == mu.alpha_plus:
            next_loc[1][0] = current_loc[1][0] + 1 if current_loc[1][0] + 1 < 24 else 0
        elif move == mu.alpha_minus:
            next_loc[1][0] = current_loc[1][0] - 1 if current_loc[1][0] - 1 >= 0 else 23
        elif move == mu.beta_plus:
            next_loc[1][1] = current_loc[1][1] + 1 if current_loc[1][1] + 1 < 24 else 0
        elif move == mu.beta_minus:
            next_loc[1][1] = current_loc[1][1] - 1 if current_loc[1][1] - 1 >= 0 else 23
        elif move == mu.theta_plus:
            next_loc[1][2] = current_loc[1][2] + 1 if current_loc[1][2] + 1 < 24 else 0
        elif move == mu.theta_minus:
            next_loc[1][2] = current_loc[1][2] - 1 if current_loc[1][2] - 1 >= 0 else 23
        else:
            raise ValueError('unrecognized move')
        return next_loc

    def render_grid(self, mode='human'):
        if not self.rendered_occupancy:
            cv2.namedWindow('image' + str(self.env_id), cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image' + str(self.env_id), 300, 300)
            self.rendered_occupancy = True
        if mode == 'rgb_array':
            return self.occupancy_grid  # return RGB frame suitable for video
        elif mode == 'human':
            # pop up a window for visualization
            cv2.imshow('image' + str(self.env_id), self.occupancy_grid[0])
            cv2.waitKey(1)
            return self.occupancy_grid
        else:
            super(FloatingFingerEnv3D, self).render(mode=mode)  # just raise an exception

    def render_points(self):
        if self.viewer is None:
            positions = np.array([
                [0, 0, 0],
                [0, self.max_y, 0],
                [self.max_x, self.max_y, 0],
                [self.max_x, 0, 0]
            ])
            workspace_mesh = pyrender.Mesh([pyrender.Primitive(positions=positions, mode=2)])
            workspace_node = pyrender.Node(mesh=workspace_mesh)
            self.scene = pyrender.Scene()
            self.scene.add_node(workspace_node)
            self.viewer = pyrender.Viewer(self.scene,
                                          run_in_thread=True,
                                          viewer_flags={'show_world_axis': True,
                                                        'use_raymond_lighting': True},
                                          render_flags={'vertex_normals': False,
                                                        'point_size': 10})
        self.viewer.render_lock.acquire()
        if self.points_node is not None:
            self.scene.remove_node(self.points_node)
        points = np.append(np.array(list(self.points)), np.array(self.finger_point)[None, ...], axis=0)
        # points = np.array(list(zip(*list(self.all_points)))[0])
        # normals = np.array(list(zip(*list(self.all_points)))[1])
        colors = np.array([mu.WHITE] * len(self.points) + [mu.RED])
        self.points_node = pyrender.Node(mesh=pyrender.Mesh.from_points(points, colors=colors))
        self.scene.add_node(self.points_node)
        self.viewer.render_lock.release()
        time.sleep(0.001)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_polygon_position(self):
        x = self.max_x / 2
        y = self.max_y / 2
        # compute z
        if self.dataset in ['ycb_objects_origin_at_center', 'ycb_objects_origin_at_center_vhacd']:
            if self.dof6_orientation:
                z = self.finger_initial_z_loc
            else:
                z = self.polygon_bodyinfos[self.polygon_id][2] / 2
        else:
            if self.dof6_orientation:
                z = self.finger_initial_z_loc - 0.05
            else:
                z = 0

        # some special objects
        if not self.dof6_orientation and pu.get_body_name(self.polygons[self.polygon_id]) == 'bowl':
            # raise the height of the bowl
            z += 0.05

        z_with_trans_noise = z + self.np_random.uniform(low=-self.translate_range_z, high=self.translate_range_z)

        # sample x, y
        if self.translate_range > 0:
            x = self.np_random.uniform(low=self.max_x / 2 - self.translate_range / 2,
                                       high=self.max_x / 2 + self.translate_range / 2)
            y = self.np_random.uniform(low=self.max_y / 2 - self.translate_range / 2,
                                       high=self.max_y / 2 + self.translate_range / 2)
        elif self.translate_range == 0:
            pass
        else:
            occupy_center = False
            while not occupy_center:
                x = self.np_random.uniform(low=0, high=self.max_x)
                y = self.np_random.uniform(low=0, high=self.max_y)
                z_with_trans_noise = z + self.np_random.uniform(low=-self.translate_range_z,
                                                                high=self.translate_range_z)

                pu.set_pose(self.polygons[self.polygon_id],
                            ([x, y, z_with_trans_noise], self.polygon_initial_quaternion), client=self.client_id)

                if p.rayTest([0.15, 0.15, -1], [0.15, 0.15, 1])[0][0] != -1:  # above the center
                    if \
                    p.rayTest([0.149, 0.149, self.finger_initial_z_loc], [0.151, 0.151, self.finger_initial_z_loc])[0][
                        0] != -1 or \
                            p.rayTest([0.149, 0.151, self.finger_initial_z_loc],
                                      [0.151, 0.149, self.finger_initial_z_loc])[0][
                                0] != -1:  # can be touched by heuristic move
                        occupy_center = True
        return [x, y, z_with_trans_noise]

    def generate_heuristic_trajectory(self):
        trajectory = [(0, 0)]
        current = (0, 0)
        while current != (self.max_x_idx - 1, self.max_y_idx - 1):
            # move down
            current = mu.compute_next_loc(current, 2, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
            # move right
            current = mu.compute_next_loc(current, 1, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
        while current != (0, self.max_y_idx - 1):
            # move up
            current = mu.compute_next_loc(current, 0, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
        while current != (self.max_x_idx - 1, 0):
            # move down
            current = mu.compute_next_loc(current, 2, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
            # move left
            current = mu.compute_next_loc(current, 3, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
        return trajectory

    def generate_heuristic_moves(self):
        # using a list of moves, so we can check the collision within the move function
        moves = []

        current = self.finger_initial_loc
        while current[0][0] != self.max_x_idx - 1 and current[0][1] != self.max_y_idx - 1:
            # x plus
            current = self.compute_next_loc(current, mu.x_plus)
            moves.append(mu.x_plus)
            # y plus
            current = self.compute_next_loc(current, mu.y_plus)
            moves.append(mu.y_plus)
        # move to the other corner
        current = self.finger_corner_loc_min_x_max_y
        while current[0][0] != self.max_x_idx - 1 and current[0][1] != 0:
            # x plus
            current = self.compute_next_loc(current, mu.x_plus)
            moves.append(mu.x_plus)
            # y minus
            current = self.compute_next_loc(current, mu.y_minus)
            moves.append(mu.y_minus)

        return moves

    def generate_heuristic_moves_towards_center(self):

        moves = []

        current_loc = copy.deepcopy(self.current_loc)
        # set finger orientations according to current position
        if self.current_loc[0][0] <= 30 and self.current_loc[0][1] <= 30:
            new_orientation = [12, 20, 3]
        elif self.current_loc[0][0] <= 30 and self.current_loc[0][1] > 30:
            new_orientation = [12, 20, 21]
        elif self.current_loc[0][0] > 30 and self.current_loc[0][1] <= 30:
            new_orientation = [12, 4, 3]
        elif self.current_loc[0][0] > 30 and self.current_loc[0][1] > 30:
            new_orientation = [12, 4, 21]

        delta_x = int(self.max_x_idx / 2) - self.current_loc[0][0]
        delta_y = int(self.max_y_idx / 2) - self.current_loc[0][1]
        delta_z = self.finger_height_index - self.current_loc[0][2]

        alpha_add = new_orientation[0] - self.current_loc[1][0]
        if alpha_add > 0:
            alpha_minus = new_orientation[0] - self.current_loc[1][0] - 24
        else:
            alpha_minus = new_orientation[0] - self.current_loc[1][0] + 24

        beta_add = new_orientation[1] - self.current_loc[1][1]
        if beta_add > 0:
            beta_minus = new_orientation[1] - self.current_loc[1][1] - 24
        else:
            beta_minus = new_orientation[1] - self.current_loc[1][1] + 24

        theta_add = new_orientation[2] - self.current_loc[1][2]
        if theta_add > 0:
            theta_minus = new_orientation[2] - self.current_loc[1][2] - 24
        else:
            theta_minus = new_orientation[2] - self.current_loc[1][2] + 24

        delta_alpha = alpha_add if abs(alpha_add) < abs(alpha_minus) else alpha_minus
        delta_beta = beta_add if abs(beta_add) < abs(beta_minus) else beta_minus
        delta_theta = theta_add if abs(theta_add) < abs(theta_minus) else theta_minus

        while delta_theta > 0:
            current_loc = self.compute_next_loc(current_loc, mu.theta_plus)
            moves.append(mu.theta_plus)
            delta_theta -= 1
        while delta_theta < 0:
            current_loc = self.compute_next_loc(current_loc, mu.theta_minus)
            moves.append(mu.theta_minus)
            delta_theta += 1

        while delta_beta > 0:
            current_loc = self.compute_next_loc(current_loc, mu.beta_plus)
            moves.append(mu.beta_plus)
            delta_beta -= 1
        while delta_beta < 0:
            current_loc = self.compute_next_loc(current_loc, mu.beta_minus)
            moves.append(mu.beta_minus)
            delta_beta += 1

        while delta_alpha > 0:
            current_loc = self.compute_next_loc(current_loc, mu.alpha_plus)
            moves.append(mu.alpha_plus)
            delta_alpha -= 1
        while delta_alpha < 0:
            current_loc = self.compute_next_loc(current_loc, mu.alpha_minus)
            moves.append(mu.alpha_minus)
            delta_alpha += 1

        while delta_z > 0:
            current_loc = self.compute_next_loc(current_loc, mu.z_plus)
            moves.append(mu.z_plus)
            delta_z -= 1
        while delta_z < 0:
            current_loc = self.compute_next_loc(current_loc, mu.z_minus)
            moves.append(mu.z_minus)
            delta_z += 1

        while delta_y > 0:
            current_loc = self.compute_next_loc(current_loc, mu.y_plus)
            moves.append(mu.y_plus)
            delta_y -= 1
        while delta_y < 0:
            current_loc = self.compute_next_loc(current_loc, mu.y_minus)
            moves.append(mu.y_minus)
            delta_y += 1

        while delta_x > 0:
            current_loc = self.compute_next_loc(current_loc, mu.x_plus)
            moves.append(mu.x_plus)
            delta_x -= 1
        while delta_x < 0:
            current_loc = self.compute_next_loc(current_loc, mu.x_minus)
            moves.append(mu.x_minus)
            delta_x += 1

        return moves

    def legal_poses(self):

        alphas = np.arange(0, 24)
        betas = np.arange(0, 24)
        thetas = np.arange(0, 24)
        self.poses_cube = np.zeros([24, 24, 24])

        center_pos_in_tip = [0, 0, -self.r]
        center_ori_in_tip = [0, 0, 0, 1]

        for alpha in alphas:
            for beta in betas:
                for theta in thetas:
                    current_loc = [[0, 0, 0], [alpha, beta, theta]]
                    tip_pos_in_world = [0, 0, 0]
                    tip_ori_in_world = p.getQuaternionFromEuler(
                        self.degree_2_radian(np.array(current_loc[1]) * self.rot_step_size))
                    center_pos_in_world, center_ori_in_world = p.multiplyTransforms(tip_pos_in_world, tip_ori_in_world,
                                                                                    center_pos_in_tip,
                                                                                    center_ori_in_tip)
                    if center_pos_in_world[2] > -1e-6 and center_pos_in_world[2] <= self.r / np.sqrt(2):
                        self.poses_cube[alpha, beta, theta] = 1

        return self.poses_cube

    def set_legal_actions(self):
        self.legal_actions[6] = self.poses_cube[
            (self.current_loc[1][0] + 1) % 24, self.current_loc[1][1], self.current_loc[1][2]]
        self.legal_actions[7] = self.poses_cube[
            (self.current_loc[1][0] - 1) % 24, self.current_loc[1][1], self.current_loc[1][2]]
        self.legal_actions[8] = self.poses_cube[
            self.current_loc[1][0], (self.current_loc[1][1] + 1) % 24, self.current_loc[1][2]]
        self.legal_actions[9] = self.poses_cube[
            self.current_loc[1][0], (self.current_loc[1][1] - 1) % 24, self.current_loc[1][2]]
        self.legal_actions[10] = self.poses_cube[
            self.current_loc[1][0], self.current_loc[1][1], (self.current_loc[1][2] + 1) % 24]
        self.legal_actions[11] = self.poses_cube[
            self.current_loc[1][0], self.current_loc[1][1], (self.current_loc[1][2] - 1) % 24]

    # real-robot related
    def close(self):
        super().close()
        if self.real:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()

    def get_finger_tip_pose_ws(self):
        return self.from_control_space_to_workspace(self.rtde_r.getActualTCPPose())

    def cartesian_control(self, axis, distance):
        current_position = self.rtde_r.getActualTCPPose()[:3]
        current_orientation = self.rtde_r.getActualTCPPose()[:3]
        # computer new pose
        new_position = copy.deepcopy(current_position)
        if axis == 'x':
            new_position[0] += distance
        elif axis == 'y':
            new_position[1] += distance
        elif axis == 'z':
            new_position[2] += distance
        else:
            raise TypeError('unrecognized axis')
        new_pose = new_position + current_orientation
        self.rtde_c.moveL(new_pose)

    def from_workspace_to_robot(self, pose):
        """
        Transform a pose in workspace frame to robot frame
        pose: [pos, euler (in degrees)]
        """
        workspace_pos_in_robot = self.workspace_origin_position
        workspace_ori_in_robot = p.getQuaternionFromEuler(mu.degree_2_radian([0, 0, 180]))
        pos_in_workspace = pose[0]
        ori_in_workspace = p.getQuaternionFromEuler(mu.degree_2_radian(pose[1]))
        pos_in_robot, ori_in_robot = p.multiplyTransforms(workspace_pos_in_robot, workspace_ori_in_robot, pos_in_workspace, ori_in_workspace)
        ori_in_robot = mu.radian_2_degree(p.getEulerFromQuaternion(ori_in_robot))
        return [list(pos_in_robot), ori_in_robot]

    def from_robot_to_workspace(self, pose):
        """
        Transform a pose in robot frame to workspace frame
        pose: [pos, euler (in degrees)]
        """
        # ws_P = ws_T_robot * robot_P
        robot_pos_in_workspace = self.workspace_origin_position[:2] + [-self.workspace_origin_position[2]]
        robot_ori_in_workspace = p.getQuaternionFromEuler(mu.degree_2_radian([0, 0, 180]))
        pos_in_robot = pose[0]
        ori_in_robot = p.getQuaternionFromEuler(mu.degree_2_radian(pose[1]))
        pos_in_workspace, ori_in_workspace = p.multiplyTransforms(robot_pos_in_workspace, robot_ori_in_workspace, pos_in_robot, ori_in_robot)
        ori_in_workspace = mu.radian_2_degree((p.getEulerFromQuaternion(ori_in_workspace)))
        return [list(pos_in_workspace), ori_in_workspace]

    def from_workspace_to_control_space(self, pose_ws):
        """
        first convert the pose in workspace to robot space, but this pose is the finger tip pose.
        then convert finger tip pose to tcp pose

        Args:
            pose_ws: [pos, euler (degrees)]

        Returns:
            pose_control: [x, y, z, rx, ry, rz (radians)]
        """
        ft_pose_robot = self.from_workspace_to_robot(pose_ws)
        tcp_pose_robot = mu.get_tcp_pose(ft_pose_robot)
        return mu.from_mine_to_ur5(tcp_pose_robot)

    def from_control_space_to_workspace(self, pose_control):
        """
        first convert the pose in robot space to workspace, but this pose is the tcp pose.
        then convert tcp pose to finger tip pose

        Args:
            pose_control: [x, y, z, rx, ry, rz (radians)]

        Returns:
            pose_ws: [pos, euler (degrees)]
        """
        pose_control = mu.from_ur5_to_mine(pose_control)
        tcp_pose_ws = self.from_robot_to_workspace(pose_control)
        return mu.get_finger_tip_pose(tcp_pose_ws)

    def collision_callback(self, msg):
        self.collision = msg.data

    def contact_callback(self, msg):
        position = msg.position
        position = [position.x, position.y, position.z]
        quaternion = msg.quaternion
        quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        normal = msg.normal
        normal = [normal.x, normal.y, normal.z]
        force = msg.force
        self.contact_pose = mu.convert_contact_pose(
            [position, mu.radian_2_degree(p.getEulerFromQuaternion(quaternion))],
            self.get_finger_tip_pose_ws()
        )
        finger_sphere_center = mu.convert_contact_pose([[0, 0, 0], [0, 0, 0]], self.get_finger_tip_pose_ws())
        normal_in_ws = np.array(finger_sphere_center[0]) - np.array(self.contact_pose[0])
        normal_in_ws = normal_in_ws / np.linalg.norm(normal_in_ws)
        self.normal_in_ws = tuple(normal_in_ws)


class FloatingFingerController:
    def __init__(self, initial_pose, urdf_path, client_id, visualize_frame):
        self.initial_pose = initial_pose
        self.urdf_path = urdf_path
        self.client_id = client_id
        self.visualize_frame = visualize_frame
        self.id = p.loadURDF(self.urdf_path,
                             initial_pose[0],
                             quaternion_from_euler_degrees(initial_pose[1]),
                             physicsClientId=self.client_id)

        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=initial_pose[0],
                                      childFrameOrientation=quaternion_from_euler_degrees(initial_pose[1]),
                                      physicsClientId=self.client_id)

        if self.visualize_frame:
            self.frame_id = pu.show_link_frame(self.id, -1, length=0.05)

    def set_pose_no_control(self, pose):
        """ pose is in euler degrees """
        # we should not change the pose elements directly, we should make a copy
        pose_ = copy.deepcopy(pose)
        pose_[1] = quaternion_from_euler_degrees(pose[1])
        pu.set_pose(self.id, pose_, client=self.client_id)

    def set_pose(self, pose):
        """ pose is in euler degrees """
        # we should not change the pose elements directly, we should make a copy
        pose_ = copy.deepcopy(pose)
        pose_[1] = quaternion_from_euler_degrees(pose[1])
        pu.set_pose(self.id, pose_, client=self.client_id)
        self.control_pose(pose_)

    def get_pose(self):
        """ returned pose is in euler degrees """
        pose = pu.get_body_pose(self.id, client=self.client_id)
        pose[1] = euler_degrees_from_quaternion(pose[1])
        return pose

    def control_pose(self, pose):
        """ pose is in euler degrees """
        p.changeConstraint(self.cid,
                           jointChildPivot=pose[0],
                           jointChildFrameOrientation=quaternion_from_euler_degrees(pose[1]),
                           physicsClientId=self.client_id)


def quaternion_from_euler_degrees(euler_degrees):
    euler = [math.radians(x) for x in euler_degrees]
    return pu.quaternion_from_euler(euler)


def euler_degrees_from_quaternion(quat):
    euler = pu.euler_from_quaternion(quat)
    return [math.degrees(x) for x in euler]
