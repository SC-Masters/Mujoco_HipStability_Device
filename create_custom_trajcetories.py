import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import mujoco

from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.environments import MyoSkeleton
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid

"""
This is a minimal example explaining the trajectory interface.
Here a standing still trajectory with a sine wave on the elbow is created and replayed in the Unitree H1 environment

The example shows how to:
    - create a trajectory 
    - save the trajectory
    - load the trajectory
    - replay the trajectory

"""

N_steps = 5

# create the environment
env = MyoSkeleton(init_state_type="DefaultInitialStateHandler")

# reset the env

# key = jax.random.PRNGKey(0)

env.reset()

# get the model and data of the environment
model = env.get_model()
data = env.get_data()

# get the initial qpos and qvel of the environment
qpos = data.qpos
qvel = data.qvel

# stack qpos and qvel to a trajectory
qpos = np.tile(qpos, (N_steps, 1))
qvel = np.tile(qvel, (N_steps, 1))

# add a sine wave to the elbow joint to make it more interesting
elbow_joint_left_ind = mj_jntname2qposid("knee_angle_l", model)
test = 0.01 * np.sin(np.linspace(0, 2 * np.pi, N_steps))
tes3 = qpos[:, elbow_joint_left_ind]
qpos[:, elbow_joint_left_ind] += 0.5 * np.sin(np.linspace(0, 20 * np.pi, N_steps)).reshape(-1, 1)

# since the elbow qpos is updated, qvel needs to be updated as well
qvel = qvel[1:-1]
qvel[:, elbow_joint_left_ind] = (qpos[2:, elbow_joint_left_ind] - qpos[:-2, elbow_joint_left_ind]) / (2 * env.dt)
qpos = qpos[1:-1]

# create a trajectory info -- this stores basic information about the trajectory
njnt = model.njnt
jnt_type = model.jnt_type
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]
traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=1 / env.dt)

# create a trajectory data -- this stores the actual trajectory data
traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, N_steps]))

# combine them to a trajectory
traj = Trajectory(traj_info, traj_data)

# print(traj)

#print(number_of_traj)

# example: save the trajectory
np.save("C:\\Users\\Masked Gentleman\\PycharmProjects\\PythonProject\\custom traj\\new_trajectory", traj, allow_pickle=True)

# example: load the trajectory
traj = np.load("new_trajectory.npy", allow_pickle=True)

# add the trajectory to the environment
# env.load_trajectory(traj)

# replay the trajectory
# env.play_trajectory(n_steps_per_episode=N_steps)

#np.savetxt("new_trajectory.csv", traj, delimiter=",")

one_traj = traj.reshape(1,1)
df = pd.DataFrame(one_traj)
df.to_csv("new_trajectory.csv", delimiter=' ' index=True)


# Alternatively, create an imitation learning environment and pass a custom trajectory
#env = ImitationFactory.make("UnitreeH1", custom_dataset_conf=CustomDatasetConf(traj))
#env.play_trajectory(n_steps_per_episode=N_steps)


# np.load("C:\\Users\\Masked Gentleman\\PycharmProjects\\PythonProject\\trajectory.npz", mmap_mode='r', fix_imports=True,  allow_pickle=True)