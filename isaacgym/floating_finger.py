import gym
import numpy as np
from isaacgym import gymapi

# initialize gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SimType.SIM_FLEX, sim_params)

asset_options = gymapi.AssetOptions()
# asset_options.flip_mesh_winding = True
# asset_options.fix_base_link = True
# asset_options.armature = 0.01

asset_root = "/home/tasha/PycharmProjects/robotics/assets"
finger_asset = gym.load_asset(sim, asset_root, 'finger.urdf', asset_options)
print("worked")
# object_asset = gym.load_asset(sim, asset_root, '/home/tasha/Downloads/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj', asset_options)
#object_asset = gym.load_asset(sim, '/home/tasha/Downloads/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models', 'model_normalized.obj', asset_options)

# in order to add actors, you must create an environment
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 8)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

finger_actor = gym.create_actor(env, finger_asset, pose, "Finger", 0, 1)
#object_actor = gym.create_actor(env, object_asset, pose, "Object", 0, 1)
# finger_actor = gym.create_actor(sim, finger_asset, pose)
# object_actor = gym.create_actor(sim, object_asset, pose)

while True:
    gym.simulate(sim)
    gym.fetch_results(sim, True)




# # create a small step for translation and rotation
# trans_step = 0.01  # 1cm
# rot_step = np.deg2rad(15)  # 15 degrees
#
# # define the 12 possible movements
# movements = [
#     np.array([trans_step, 0, 0]),  # increase x
#     np.array([-trans_step, 0, 0]),  # decrease x
#     np.array([0, trans_step, 0]),  # increase y
#     np.array([0, -trans_step, 0]),  # decrease y
#     np.array([0, 0, trans_step]),  # increase z
#     np.array([0, 0, -trans_step]),  # decrease z
#     np.array([rot_step, 0, 0]),  # increase roll
#     np.array([-rot_step, 0, 0]),  # decrease roll
#     np.array([0, rot_step, 0]),  # increase pitch
#     np.array([0, -rot_step, 0]),  # decrease pitch
#     np.array([0, 0, rot_step]),  # increase yaw
#     np.array([0, 0, -rot_step])  # decrease yaw
# ]
#
# # perform the simulation
# while True:
#     # randomly choose a movement
#     movement = np.random.choice(movements)
#
#     # apply the movement to the finger
#     # (you'll need to implement the `move_finger` function)
#     move_finger(finger_handle, movement)
#
#     # step the simulation
#     gym.simulate()
#     gym.fetch_results()
#
#     # check for collisions between the finger and the object
#     # (you'll need to implement the `check_collision` function)
#     if check_collision(finger_handle, object_handle):
#         # if a collision occurred, record the contact points and normals
#         # (you'll need to implement the `record_contact` function)
#         record_contact(finger_handle, object_handle)
