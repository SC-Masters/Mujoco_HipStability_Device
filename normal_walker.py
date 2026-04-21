import jax.numpy as jnp
import mujoco
import numpy as np
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid

# -----------------------------
# ENVIRONMENT SETUP
# -----------------------------
env = ImitationFactory.make(
    "MyoSkeleton",
    default_dataset_conf=DefaultDatasetConf(["walk"]),
    n_substeps=20
)
traj = env.th.traj
data = traj.data
qpos = data.qpos
qvel = data.qvel
dt = env.dt
model = env._model
simdata = env.data

# -----------------------------
# 10 SECOND TRAJECTORY SLICE
# -----------------------------
max_steps = int(10 / dt)
qpos = qpos[:max_steps]
qvel = qvel[:max_steps]
N = qpos.shape[0]

# -----------------------------
# TOE OFF DETECTION
# -----------------------------
foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "talus_l")
foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "talus_r")
pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

foot_height = []
for i in range(N):
    simdata.qpos[:] = qpos[i]
    simdata.qvel[:] = qvel[i]
    mujoco.mj_forward(model, simdata)
    foot_height.append(simdata.xpos[foot_l_id][2])
foot_height = np.array(foot_height)
threshold = 0.01
toe_off = next(
    (i for i in range(1, N) if foot_height[i-1] < threshold and foot_height[i] > threshold),
    N // 2
)
print("Toe-off detected at frame:", toe_off)

# -----------------------------
# PERTURBATION PARAMETERS
# -----------------------------
mass = 70
g = 9.81
force_mag = 0.12 * mass * g
acc_mag = force_mag / mass
preblend_steps = 1
perturb_steps = 8
step_frames = 1
foot_lock_frames = 8
postblend_steps = 8
resume_offset = 8
num_ramp_frames = 10
direction = 0.0
time = np.linspace(0, np.pi, perturb_steps)
acc_profile = acc_mag * np.sin(time)

# =========================================================
# PERTURBATION DIRECTION CONTROL
# =========================================================
#  0.0 = pure AP
#  1.0 = pure ML
# -1.0 = opposite AP direction (optional)
# =========================================================

PERTURB_DIR = 0.0   # <- YOU TUNE THIS

# -----------------------------
# AUTO-TUNED GAINS
# -----------------------------
AP_GAIN = 6.0      # keep your original scaling
ML_GAIN = 1.0       # tuned for ~0.3 m lateral

# -----------------------------
# DIRECTION LOCK
# -----------------------------
locked_direction = 0.0
direction_locked = False
lock_delay = 2  # frames after perturbation starts

# -----------------------------
# DOF INDICES
# -----------------------------
pelvis_lat_idx = 1
pelvis_fwd_idx = 0

joint_names = ["femur_l", "tibia_l", "calcn_l"]
for name in joint_names:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_adr = model.jnt_qposadr[j_id]
    dof_adr = model.jnt_dofadr[j_id]

# -----------------------------
# RECOVERY DYNAMICS
# -----------------------------
recovery_frames = 40
osc_freq = 1.7       # shakiness recovery (shakier when higher)
osc_decay = 3.0      # wobble of the body (decreased when lower)
recovery_gain = 0.35
RECOVERY_EFFICIENCY = 0.5  # 0.5 = half as effective
DRIFT_GAIN = 0.15           # small bias to keep instability alive

# -----------------------------
# DEBUG JOINT INDICES
# -----------------------------
femur_l_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "femur_l")
femur_r_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "femur_r")
tibia_l_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tibia_l")
calcn_l_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "calcn_l")

# -----------------------------
# TRAJECTORY STORAGE
# -----------------------------
qpos_new = np.array(qpos)
qvel_new = np.array(qvel)

# -----------------------------
# TIMING OFFSET
# -----------------------------
perturb_offset = -10  # 🔥 tune this (-10 to -40)

# =========================================================
# TRACKING ERROR STORAGE (replaces max_displacements)
# =========================================================
max_errors = {
    "talus_l": {"AP": 0.0, "ML": 0.0},
    "talus_r": {"AP": 0.0, "ML": 0.0}
}

# =========================================================
# PEAK ERROR (at target reach)
# =========================================================
peak_error = {
    "talus_l": {"AP": None, "ML": None},
    "talus_r": {"AP": None, "ML": None}
}

# -----------------------------
# REBUILD TRAJECTORY
# -----------------------------
new_data = data.replace(
    qpos=jnp.array(qpos_new),
    qvel=jnp.array(qvel_new),
    split_points=jnp.array([0, N])
)
traj.data = new_data
env.load_trajectory(traj)

# -----------------------------
# SAVE TRAJECTORY
# -----------------------------
traj.save("normal_trajectory.npz")
print("Saved normal_trajectory.npz!!!")

# -----------------------------
# PLAYBACK
# -----------------------------
env.play_trajectory(n_steps_per_episode=N)