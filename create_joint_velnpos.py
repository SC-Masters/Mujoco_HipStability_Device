import mujoco
import numpy as np
import pandas as pd

# =========================
# USER PATHS
# =========================
# XML_PATH = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\.venv\Lib\site-packages\loco_mujoco\models\myo_model\myoskeleton\myoskeleton.xml"
JOINT_NAMES_PATH = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\normal_npy\joint_names.npy"
QPOS_PATH = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\normal_npy\qpos.npy"
QVEL_PATH = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\normal_npy\qvel.npy"
OUTPUT_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\normal_data\normal_vals.xlsx"

DT = 1 # seconds (adjust if needed)

# =========================
# LOAD DATA
# =========================
joint_names = np.load(JOINT_NAMES_PATH, allow_pickle=True).tolist()
qpos = np.load(QPOS_PATH)   # (T, nq)
qvel = np.load(QVEL_PATH)   # (T, nv)

T = qpos.shape[0]
time = np.arange(T) * DT

# =========================
# BUILD TABLE
# =========================
data = {}
data["time_s"] = time

n_joints = 2*len(joint_names)

print(n_joints)

for i, name in enumerate(joint_names):

    # Position
    if i < qpos.shape[1]:
        data[f"{name}_q"] = qpos[:, i]

    # Velocity (may be shorter by 1)
    if i < qvel.shape[1]:
        data[f"{name}_v"] = qvel[:, i]

# =========================
# SANITY CHECK
# =========================
print("joint_names:", len(joint_names))
print("qpos DOFs:", qpos.shape[1])
print("qvel DOFs:", qvel.shape[1])
print("exported columns (no time):", len(data) - 1)

df = pd.DataFrame(data)
df.to_excel(OUTPUT_XLSX, index=False)

print(f"Saved {OUTPUT_XLSX} ✅")