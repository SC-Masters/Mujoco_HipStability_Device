import numpy as np
import loco_mujoco
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf
import gymnasium as gym

# note: we do not support parallel environments in gymnasium yet!
env = gym.make("LocoMujoco", env_name="SkeletonTorque", render_mode="human",
               default_dataset_conf=DefaultDatasetConf("walk"),
               lafan1_dataset_conf=LAFAN1DatasetConf("walk1_subject1"),
               goal_type="GoalTrajMimicv2", goal_params=dict(visualize_goal=True))

action_dim = env.action_space.shape[0]

seed = 1

env.reset()
img = env.render()
absorbing = False
i = 0   

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, _, absorbing, _,  _ = env.step(action)

    env.render()
    i += 1