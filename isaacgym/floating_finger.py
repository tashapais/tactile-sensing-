import random
import gym
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from pyquaternion import Quaternion

# initialize gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SimType.SIM_FLEX, sim_params)

# #change up axis to z instead of y
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 1, 0) # z-up!
plane_params.distance = 0 #distance of the plane from the origin
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0 #used to control the elasticity of collisions with the ground plane (amount of bounce)

# create the ground plane
gym.add_ground(sim, plane_params)

# in order to add actors, you must create an environment
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 8)

# in order to add actor to environment, these must be specified
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107) #arguments are (x, y, z, w)
height = random.uniform(1.0, 2.5)

# create assets
asset_root = "/home/tasha/PycharmProjects/robotics/assets"
finger_asset = gym.load_asset(sim, asset_root, 'finger.urdf')
object_asset = gym.load_asset(sim, asset_root, 'object.urdf')
finger_actor = gym.create_actor(env, finger_asset, pose, "Finger", 0, 1)
object_actor = gym.create_actor(env, object_asset, pose, "Object", 0, 1)

# create a small step for translation and rotation
trans_step = 0.01  # 1cm
rot_step = np.deg2rad(15)  # 15 degrees

# define the 12 possible movements
translations = [
    np.array([trans_step, 0, 0]),  # increase x
    np.array([-trans_step, 0, 0]),  # decrease x
    np.array([0, trans_step, 0]),  # increase y
    np.array([0, -trans_step, 0]),  # decrease y
    np.array([0, 0, trans_step]),  # increase z
    np.array([0, 0, -trans_step])  # decrease z
]

rotations = [
    np.array([rot_step, 0, 0]),  # increase roll
    np.array([-rot_step, 0, 0]),  # decrease roll
    np.array([0, rot_step, 0]),  # increase pitch
    np.array([0, -rot_step, 0]),  # decrease pitch
    np.array([0, 0, rot_step]),  # increase yaw
    np.array([0, 0, -rot_step])  # decrease yaw
]


def move_finger(finger_handle, trans, rot):
    # Get the current state of the finger
    body_states = gym.get_actor_rigid_body_states(env, finger_handle, gymapi.STATE_ALL)

    # Convert the movement to numpy arrays
    trans = np.array(trans)
    rot = np.array(rot)

    # Update the position
    body_states["pose"]["p"]["x"] += trans[0]
    body_states["pose"]["p"]["y"] += trans[1]
    body_states["pose"]["p"]["z"] += trans[2]

    # Get the current rotation as a quaternion
    current_quat = Quaternion(body_states["pose"]["r"]["w"][0], body_states["pose"]["r"]["x"][0], body_states["pose"]["r"]["y"][0], body_states["pose"]["r"]["z"][0])

    # Get the rotation as a quaternion
    rotation_quat = Quaternion(axis=[1, 0, 0], angle=rot[0]) * Quaternion(axis=[0, 1, 0], angle=rot[1]) * Quaternion(axis=[0, 0, 1], angle=rot[2])

    # Apply the rotation
    new_quat = current_quat * rotation_quat

    # Update the rotation
    body_states["pose"]["r"]["x"][0] = new_quat.x
    body_states["pose"]["r"]["y"][0] = new_quat.y
    body_states["pose"]["r"]["z"][0] = new_quat.z
    body_states["pose"]["r"]["w"][0] = new_quat.w

    # Set the new state of the finger
    gym.set_actor_rigid_body_states(env, finger_handle, body_states, gymapi.STATE_ALL)




def check_collision(finger_handle, object_handle):
    # Get the net contact force tensor
    _net_cf = gym.acquire_net_contact_force_tensor(sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)

    # Refresh the tensor with the latest data from the simulation
    gym.refresh_net_contact_force_tensor(sim)

    # Get the forces on the finger and the object
    finger_force = net_cf[finger_handle]
    object_force = net_cf[object_handle]

    # Check if there is a non-zero force on either the finger or the object
    if torch.norm(finger_force) > 0 or torch.norm(object_force) > 0:
        return True

    return False


def record_contact(finger_handle, object_handle):
    print('Contact detected between finger and object')


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
while not gym.query_viewer_has_closed(viewer): #terminate the simulation when the viewer window is closed
    # randomly choose a translation and a rotation
    trans_index = np.random.choice(len(translations))  # Choose a random index for translation
    rot_index = np.random.choice(len(rotations))  # Choose a random index for rotation

    trans = translations[trans_index]  # Get the corresponding translation
    rot = rotations[rot_index]  # Get the corresponding rotation

    # apply the movement to the finger
    move_finger(finger_actor, trans, rot)

    #step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if check_collision(finger_actor, object_actor):
        record_contact(finger_actor, object_actor)

    gym.step_graphics(sim) #synchronizes the visual representation of the simulation with the physics state
    gym.draw_viewer(viewer, sim, True) #renders the latest snapshot in the viewer
    gym.sync_frame_time(sim) #synchronize the visual update frequency with real time
