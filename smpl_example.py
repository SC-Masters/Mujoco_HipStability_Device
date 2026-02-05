from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf

# Create an environment with retargeted AMASS motion
env = ImitationFactory.make(
    "UnitreeH1",
    amass_dataset_conf=AMASSDatasetConf([
        "C:/Users/Masked Gentleman/PycharmProjects/PythonProject/.venv/Lib/site-packages/loco_mujoco/datasets/amass/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v3_C3D_poses"
    ]),
    n_substeps=20
)

traj = env.th.traj

# Play the retargeted trajectory
env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)