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

PERTURB_DIR = 1.0   # <- YOU TUNE THIS

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

def get_body_jac(model, data, body_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jacp

# =========================================================
# PERTURBATION WINDOW ERROR TRACKING
# =========================================================
err_log_l = []
err_log_r = []

# -----------------------------
# INTEGRAL ERROR STORAGE (NEW)
# -----------------------------
err_int_l = np.zeros(2)  # [AP, ML]

err_int_r = np.zeros(2)  # [AP, ML]

# =========================================================
# MOS INPUT (FROM YOUR OTHER SCRIPT)
# =========================================================

# Replace these with real-time values later
mos_ap_percent = -88.6  # + means unstable forward
mos_ml_percent = -57.3  # + means unstable lateral

# ---------------------------------------------------------
# CLAMP + NORMALIZE
# ---------------------------------------------------------
def clamp(x, lo=-100.0, hi=100.0):
    return max(lo, min(hi, x))

# =========================================================
# MOS → FOOT TARGET SHIFT (DECOUPLED GAINS)
# =========================================================

MOS_GAIN_AP = 0.5   # reduce AP sensitivity
MOS_GAIN_ML = 0.5   # increase ML sensitivity (IMPORTANT)

mos_ap = clamp(mos_ap_percent) / 100.0
mos_ml = clamp(mos_ml_percent) / 100.0

mos_target_shift = np.array([
    -MOS_GAIN_AP * mos_ap,   # AP correction (smaller)
    -MOS_GAIN_ML * mos_ml    # ML correction (stronger)
])

print(mos_target_shift)

# -----------------------------
# SIMULATION LOOP WITH COUNTER-STEP & AUTO GAINS
# -----------------------------
for frame in range(toe_off, N):
    i = frame - toe_off
    i_shifted = i + perturb_offset
    a_ext = 0.0

    # -----------------------------
    # DEFINE REFERENCE AT TOE-OFF
    # -----------------------------
    if frame == toe_off:
        simdata.qpos[:] = qpos_new[frame]
        simdata.qvel[:] = qvel_new[frame]
        mujoco.mj_forward(model, simdata)

        # --- COM reference ---
        com_AP_ref = simdata.xpos[pelvis_id][0]
        com_ML_ref = simdata.xpos[pelvis_id][1]

        foot_l_AP_0 = simdata.xpos[foot_l_id][0]
        foot_l_ML_0 = simdata.xpos[foot_l_id][1]

        foot_r_AP_0 = simdata.xpos[foot_r_id][0]
        foot_r_ML_0 = simdata.xpos[foot_r_id][1]

        foot_l_start = np.array([foot_l_AP_0 - com_AP_ref,
                                 foot_l_ML_0 - com_ML_ref])

        foot_r_start = np.array([foot_r_AP_0 - com_AP_ref,
                                 foot_r_ML_0 - com_ML_ref])

    # -----------------------------
    # PREBLEND
    # -----------------------------
    if i < preblend_steps:
        ramp = (i + 1) / preblend_steps
        for idx in [pelvis_lat_idx, pelvis_fwd_idx]:
            qpos_new[frame, idx] = (1 - ramp) * qpos_new[frame-1, idx] + ramp * qpos[frame, idx]
            qvel_new[frame, idx] = (qpos_new[frame, idx] - qpos_new[frame-1, idx]) / dt
        continue

    # -----------------------------
    # PERTURBATION
    # -----------------------------
    if preblend_steps <= i_shifted < preblend_steps + perturb_steps:
        idx = i_shifted - preblend_steps
        # =========================================================
        # DIRECTIONAL PERTURBATION FORCE
        # =========================================================

        a_ext = acc_profile[idx]
        ap_component = (1.0 - abs(PERTURB_DIR)) * a_ext
        ml_component = PERTURB_DIR * a_ext

        perturb_force = np.array([ap_component, ml_component])

        # =========================================================
        # PELVIS PERTURBATION (FIXED - SINGLE INTEGRATION)
        # =========================================================

        acc = np.array([ap_component, ml_component])

        qvel_new[frame, pelvis_fwd_idx] = (
                qvel_new[frame - 1, pelvis_fwd_idx] + AP_GAIN * acc[0] * dt
        )

        qvel_new[frame, pelvis_lat_idx] = (
                qvel_new[frame - 1, pelvis_lat_idx] + ML_GAIN * acc[1] * dt
        )

        qpos_new[frame, pelvis_fwd_idx] = (
                qpos_new[frame - 1, pelvis_fwd_idx] + qvel_new[frame, pelvis_fwd_idx] * dt
        )

        qpos_new[frame, pelvis_lat_idx] = (
                qpos_new[frame - 1, pelvis_lat_idx] + qvel_new[frame, pelvis_lat_idx] * dt
        )

        # =========================================================
        # TRUNK RESPONSE (WITH RECOVERY)
        # =========================================================
        trunk_active = (
                preblend_steps <= i_shifted <
                preblend_steps + perturb_steps + recovery_frames
        )
        if trunk_active:
            if i_shifted < preblend_steps + perturb_steps:
                # ---- NORMAL RESPONSE ----
                idx = i_shifted - preblend_steps
                t_norm = idx / perturb_steps
                trunk_phase = np.sin(np.pi * t_norm) ** 1.5
                ramp = np.sin(np.pi * t_norm / 2) ** 2
            else:
                # ---- RECOVERY / STUMBLE ----
                t_tail = (i_shifted - (preblend_steps + perturb_steps)) / recovery_frames
                oscillation = np.sin(2 * np.pi * osc_freq * t_tail)
                decay = np.exp(-osc_decay * t_tail)
                trunk_phase = RECOVERY_EFFICIENCY * recovery_gain * oscillation * decay
                ramp = 1.0

            # ---- TRUNK JOINTS ----
            lat_bend_joints = [
                "L5_S1_Lat_Bending", "L4_L5_Lat_Bending", "L3_L4_Lat_Bending",
                "L2_L3_Lat_Bending", "L1_L2_Lat_Bending", "L1_T12_Lat_Bending",
                "c7_c6_LB", "c6_c5_LB", "c5_c4_LB", "c4_c3_LB", "c3_c2_LB",
                "c2_c1_LB", "c1_skull_LB"
            ]
            axial_rot_joints = [
                "L5_S1_axial_rotation", "L4_L5_axial_rotation", "L3_L4_axial_rotation",
                "L2_L3_axial_rotation", "L1_L2_axial_rotation", "L1_T12_axial_rotation",
                "c7_c6_AR", "c6_c5_AR", "c5_c4_AR", "c4_c3_AR", "c3_c2_AR",
                "c2_c1_AR", "c1_skull_AR"
            ]
            flex_ext_joints = [
                "L5_S1_Flex_Ext", "L4_L5_Flex_Ext", "L3_L4_Flex_Ext",
                "L2_L3_Flex_Ext", "L1_L2_Flex_Ext", "L1_T12_Flex_Ext",
                "c7_c6_FE", "c6_c5_FE", "c5_c4_FE", "c4_c3_FE", "c3_c2_FE",
                "c2_c1_FE", "c1_skull_FE"
            ]

            lat_idx = [mj_jntname2qposid(j, model) for j in lat_bend_joints]
            rot_idx = [mj_jntname2qposid(j, model) for j in axial_rot_joints]
            fe_idx = [mj_jntname2qposid(j, model) for j in flex_ext_joints]
            weights = np.linspace(0.1, 0.1, len(lat_idx))

            LAT_GAIN = 1.0
            ROT_GAIN = 0.15
            FE_GAIN = 0.1

            for w, lb, ar, fe in zip(weights, lat_idx, rot_idx, fe_idx):
                qpos_new[frame, lb] += direction * LAT_GAIN * w * trunk_phase * ramp
                qpos_new[frame, ar] += -direction * ROT_GAIN * w * trunk_phase * ramp
                qpos_new[frame, fe] += -direction * FE_GAIN * w * trunk_phase * ramp
                for idx in [lb, ar, fe]:
                    qvel_new[frame, idx] = (qpos_new[frame, idx] - qpos_new[frame - 1, idx]) / dt

            # =========================================================
            # FOOT PERTURBATION CONTROL (IK-LITE TASK SPACE CONTROLLER)
            # =========================================================

            # normalized time in perturbation window [0,1]
            t_norm = (i_shifted - preblend_steps) / perturb_steps
            t_norm = np.clip(t_norm, 0.0, 1.0)

            # smooth activation (bell-shaped, stable for dynamics)
            # peak at mid-perturbation, avoids impulsive jerk
            ramp = np.sin(np.pi * t_norm)

            # =========================================================
            # TARGET FOOT DISPLACEMENTS (COM-RELATIVE SPACE)
            # =========================================================
            target_l = np.array([-0.3, -0.2])  # LEFT foot (trailing)
            base_target_r = np.array([0.3, 0.3])

            # NEW: MoS-corrected target
            target_r = base_target_r + mos_target_shift  # RIGHT foot (stepping)

            print("target_r", target_r)
            print("base_target_r", base_target_r)

            # =========================================================
            # FORWARD KINEMATICS UPDATE (CURRENT STATE)
            # =========================================================
            simdata.qpos[:] = qpos_new[frame]
            simdata.qvel[:] = qvel_new[frame]
            mujoco.mj_forward(model, simdata)

            foot_l = np.array([
                simdata.xpos[foot_l_id][0] - com_AP_ref,
                simdata.xpos[foot_l_id][1] - com_ML_ref
            ])

            foot_r = np.array([
                simdata.xpos[foot_r_id][0] - com_AP_ref,
                simdata.xpos[foot_r_id][1] - com_ML_ref
            ])

            # =========================================================
            # TASK-SPACE ERROR
            # =========================================================
            err_l = target_l - foot_l
            err_r = target_r - foot_r

            # log only during perturbation
            if 0.0 < t_norm < 1.0:
                err_log_l.append(err_l.copy())
                err_log_r.append(err_r.copy())

            # =========================================================
            # IK-LITE GAINS (MAIN TUNING PARAMETERS)
            # =========================================================
            K_AP = 2.5  # sagittal correction (forward/back)
            K_ML = 1.5  # frontal correction (lateral)

            dt_scale = dt

            # =========================================================
            # JOINT MAPPING
            # =========================================================
            hip_add_r = mj_jntname2qposid("hip_adduction_r", model)
            hip_flex_r = mj_jntname2qposid("hip_flexion_r", model)

            hip_add_l = mj_jntname2qposid("hip_adduction_l", model)
            hip_flex_l = mj_jntname2qposid("hip_flexion_l", model)

            knee_l = mj_jntname2qposid("knee_angle_l", model)

            # =========================================================
            # TASK-SPACE → JOINT-SPACE INJECTION
            # (simple Jacobian-free IK approximation)
            # =========================================================

            # RIGHT LEG (step response)
            qpos_new[frame, hip_flex_r] += K_AP * err_r[0] * ramp * dt_scale
            qpos_new[frame, hip_add_r] += K_ML * err_r[1] * ramp * dt_scale

            # LEFT LEG (support leg compensation)
            qpos_new[frame, hip_flex_l] += K_AP * err_l[0] * ramp * dt_scale
            qpos_new[frame, hip_add_l] += K_ML * err_l[1] * ramp * dt_scale

            # small knee compliance (stability only)
            qpos_new[frame, knee_l] += 0.15 * err_l[0] * ramp * dt_scale

            # =========================================================
            # VELOCITY CONSISTENCY UPDATE
            # =========================================================
            for idx in [hip_add_r, hip_flex_r, hip_add_l, hip_flex_l, knee_l]:
                qvel_new[frame, idx] = (
                                               qpos_new[frame, idx] - qpos_new[frame - 1, idx]
                                       ) / dt

                # =========================================================
                # COUNTER-STEP CANCELLATION (HIP STRATEGY DEMO)
                # =========================================================

                # activation AFTER perturbation peak (so it looks like recovery)
                cancel_start = 0.1  # start halfway through perturbation
                cancel_gain = 1.0  # strength of correction (tune 1.0–3.0)

                if t_norm > cancel_start:

                    cancel_phase = (t_norm - cancel_start) / (1.0 - cancel_start)
                    cancel_ramp = np.sin(np.pi * cancel_phase / 2) ** 2  # smooth ramp-in

                    # -----------------------------------------------------
                    # TARGET = NEGATIVE OF COUNTER STEP (bring foot back)
                    # -----------------------------------------------------
                    cancel_target_r = -target_r

                    # current foot position (already computed earlier)
                    # foot_r = [...]

                    cancel_err_r = cancel_target_r - foot_r

                    # -----------------------------------------------------
                    # APPLY HIP CORRECTION (RIGHT LEG ONLY)
                    # -----------------------------------------------------
                    qpos_new[frame, hip_flex_r] += (
                            cancel_gain * K_AP * cancel_err_r[0] * cancel_ramp * dt_scale
                    )

                    qpos_new[frame, hip_add_r] += (
                            cancel_gain * K_ML * cancel_err_r[1] * cancel_ramp * dt_scale
                    )

                    # velocity consistency
                    for idx in [hip_flex_r, hip_add_r]:
                        qvel_new[frame, idx] = (
                                                       qpos_new[frame, idx] - qpos_new[frame - 1, idx]
                                               ) / dt

# log ONLY during true perturbation window
perturb_start = toe_off + preblend_steps
perturb_end   = perturb_start + perturb_steps

if perturb_start <= frame < perturb_end:
    err_log_l.append(err_l.copy())
    err_log_r.append(err_r.copy())

mean_err_l = np.mean(np.abs(np.array(err_log_l)), axis=0).reshape(-1)
mean_err_r = np.mean(np.abs(np.array(err_log_r)), axis=0).reshape(-1)

peak_err_l = np.max(np.abs(np.array(err_log_l)), axis=0).reshape(-1)
peak_err_r = np.max(np.abs(np.array(err_log_r)), axis=0).reshape(-1)

print("Logged frames:", len(err_log_l))



# -----------------------------
# PRINT MAX FOOT DISPLACEMENTS (relative to COM)
# -----------------------------
print("\n==============================")
print("MAX FOOT DISPLACEMENTS (relative to COM)")
print("==============================")
print("**ML (medial-lateral) perturbation targets**:")
print("Counter (stepping) foot: +0.3 m anterior, +0.3 m lateral")
print("Stance (trailing) foot: -0.3 m posterior, -0.2 m medial")
print("**AP (anterior-posterior) perturbation targets**:")
print("Counter (stepping) foot: +0.5 m anterior, +0.1 m lateral")
print("Trailing (other) foot: -0.6 m posterior, -0.2 m medial\n")

# =========================================================
# FINAL ERROR REPORT (CLEAN + STABLE)
# =========================================================

print("\n==============================")
print("ERROR AT PEAK PERTURBATION")
print("==============================")

def safe(v):
    return 0.0 if v is None else v

print(f"Trailing (LEFT) error -> "
      f"AP: {safe(peak_error['talus_l']['AP']):.3f}, "
      f"ML: {safe(peak_error['talus_l']['ML']):.3f}")

print(f"Counter (RIGHT) error -> "
      f"AP: {safe(peak_error['talus_r']['AP']):.3f}, "
      f"ML: {safe(peak_error['talus_r']['ML']):.3f}")

print("Peak displacement L:", np.max(np.abs(np.array(err_log_l)), axis=0))
print("Peak displacement R:", np.max(np.abs(np.array(err_log_r)), axis=0))

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
traj.save(r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\ML_mimic\ML_mimic_trajectory.npz")
print("Saved ML_mimic_trajectory.npz!!!")

# -----------------------------
# PLAYBACK
# -----------------------------
env.play_trajectory(n_steps_per_episode=N)