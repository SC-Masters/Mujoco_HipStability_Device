import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
NORMAL_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\normal_data\full_normal_stability_results.xlsx"
PERTURBED_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\AP_data\full_AP_stability_results.xlsx"
OUTPUT_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\comparison_data\AP_stability_comparison_output.xlsx"
PLOT_PATH = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\comparison_data\AP_stability_comparison_plot.png"

g = 9.81

# =========================
# PARAMETERS
# =========================
SHOE_LENGTH = 0.25
SHOE_WIDTH = 0.10
TRANSIENT_TIME = 1.0

# =========================
# SUBJECT PARAMETERS
# =========================
Height = 1.5
Weight = 70

# =========================
# ANGULAR MOMENTUM FUNCTIONS
# =========================
def compute_segment_omega(df):
    hip = df[["hip_flexion_r_wx", "hip_flexion_r_wy"]].values
    knee = df[["knee_angle_r_wx", "knee_angle_r_wy"]].values
    ankle = df[["ankle_angle_r_wx", "ankle_angle_r_wy"]].values

    return {
        "thigh": hip,
        "shank": hip + knee,
        "foot": hip + knee + ankle,
        "HAT": hip
    }

def compute_rotational_L(omega):
    # Simplified rotational inertia already baked into your previous pipeline
    L_total = None
    for seg in omega:
        L_seg = omega[seg]
        L_total = L_seg if L_total is None else L_total + L_seg
    return L_total

# =========================
# PROCESS FUNCTION (UPDATED)
# =========================
def process_df(df):

    # ---- Remove transient ----
    df = df[df["time_s"] > TRANSIENT_TIME].reset_index(drop=True)

    # =========================
    # LOCAL FRAME (CRITICAL FIX)
    # =========================
    origin_x = (df["ankle_angle_l_x"] + df["ankle_angle_r_x"]) / 2
    origin_y = (df["ankle_angle_l_y"] + df["ankle_angle_r_y"]) / 2

    CoM_x_local = df["CoM_x"] - origin_x
    CoM_y_local = df["CoM_y"] - origin_y

    calcn_l_x = df["ankle_angle_l_x"] - origin_x
    calcn_l_y = df["ankle_angle_l_y"] - origin_y
    calcn_r_x = df["ankle_angle_r_x"] - origin_x
    calcn_r_y = df["ankle_angle_r_y"] - origin_y

    # =========================
    # XCoM
    # =========================
    l = np.mean(df["CoM_z"] - df["ankle_angle_r_z"])
    omega0 = np.sqrt(g / l)

    df["XCoM_AP"] = CoM_x_local + df["CoM_vx"] / omega0
    df["XCoM_ML"] = CoM_y_local + df["CoM_vy"] / omega0

    # =========================
    # BASE OF SUPPORT (LOCAL)
    # =========================
    dist_x = np.abs(calcn_l_x - calcn_r_x)
    dist_y = np.abs(calcn_l_y - calcn_r_y)

    bos_ml_size = dist_x + SHOE_WIDTH
    bos_ap_size = dist_y + SHOE_LENGTH

    bos_center_x = (calcn_l_x + calcn_r_x) / 2
    bos_center_y = (calcn_l_y + calcn_r_y) / 2

    BoS_lat = bos_center_x + bos_ml_size / 2
    BoS_med = bos_center_x - bos_ml_size / 2
    BoS_ant = bos_center_y + bos_ap_size / 2
    BoS_post = bos_center_y - bos_ap_size / 2

    # =========================
    # MoS (CORRECT)
    # =========================
    df["MoS_AP"] = np.minimum(BoS_ant - df["XCoM_AP"], df["XCoM_AP"] - BoS_post)
    df["MoS_ML"] = np.minimum(BoS_lat - df["XCoM_ML"], df["XCoM_ML"] - BoS_med)

    # =========================
    # ANGULAR MOMENTUM (CONSISTENT FRAME)
    # =========================

    omega_segments = compute_segment_omega(df)
    L_rot = compute_rotational_L(omega_segments)

    # ---- Translational (LOCAL FRAME FIX) ----
    r_x = CoM_x_local.values - bos_center_x.values
    r_y = CoM_y_local.values - bos_center_y.values

    v_x = df["CoM_vx"].values
    v_y = df["CoM_vy"].values

    L_linear = np.zeros((len(df), 2))
    L_linear[:, 0] = r_y * (Weight * v_x) - r_x * (Weight * v_y)
    L_linear[:, 1] = r_x * (Weight * v_y) - r_y * (Weight * v_x)

    L_total = L_rot + L_linear

    df["ML_Angular_Momentum"] = L_total[:, 0]
    df["AP_Angular_Momentum"] = L_total[:, 1]

    return df

# =========================
# LOAD DATA
# =========================
df_norm = pd.read_excel(NORMAL_XLSX)
df_pert = pd.read_excel(PERTURBED_XLSX)

df_norm = process_df(df_norm)
df_pert = process_df(df_pert)

# =========================
# BASELINE (NORMAL)
# =========================
def get_bounds(series, k=2):
    return series.mean() - k*series.std(), series.mean() + k*series.std()

mos_ap_min, mos_ap_max = get_bounds(df_norm["MoS_AP"])
mos_ml_min, mos_ml_max = get_bounds(df_norm["MoS_ML"])
ap_ang_min, ap_ang_max = get_bounds(df_norm["AP_Angular_Momentum"])
ml_ang_min, ml_ang_max = get_bounds(df_norm["ML_Angular_Momentum"])

# =========================
# OUT-OF-RANGE
# =========================
df_pert["MoS_AP_out"] = (df_pert["MoS_AP"] < mos_ap_min) | (df_pert["MoS_AP"] > mos_ap_max)
df_pert["MoS_ML_out"] = (df_pert["MoS_ML"] < mos_ml_min) | (df_pert["MoS_ML"] > mos_ml_max)
df_pert["AP_AngMom_out"] = (df_pert["AP_Angular_Momentum"] < ap_ang_min) | (df_pert["AP_Angular_Momentum"] > ap_ang_max)
df_pert["ML_AngMom_out"] = (df_pert["ML_Angular_Momentum"] < ml_ang_min) | (df_pert["ML_Angular_Momentum"] > ml_ang_max)

# =========================
# PERCENT DIFFERENCE ANALYSIS (PERTURBED vs NORMAL)
# =========================
def percent_diff_safe(a, b):
    return 100 * np.abs(a - b) / (np.maximum(np.abs(b), 0.01))

# Compute percent differences
df_pert["MoS_ML_pct_diff"] = percent_diff_safe(df_pert["MoS_ML"], df_norm["MoS_ML"])
df_pert["MoS_AP_pct_diff"] = percent_diff_safe(df_pert["MoS_AP"], df_norm["MoS_AP"])
df_pert["ML_L_pct_diff"]   = percent_diff_safe(df_pert["ML_Angular_Momentum"], df_norm["ML_Angular_Momentum"])
df_pert["AP_L_pct_diff"]   = percent_diff_safe(df_pert["AP_Angular_Momentum"], df_norm["AP_Angular_Momentum"])

# =========================
# SUMMARY TABLE EXPORT
# =========================

summary_df = pd.DataFrame({
    "Metric": [
        "MoS_ML",
        "MoS_AP",
        "Angular_Momentum_ML",
        "Angular_Momentum_AP"
    ],
    "Max_%_Difference": [
        df_pert["MoS_ML_pct_diff"].max(),
        df_pert["MoS_AP_pct_diff"].max(),
        df_pert["ML_L_pct_diff"].max(),
        df_pert["AP_L_pct_diff"].max()
    ],
    "Mean_%_Difference": [
        df_pert["MoS_ML_pct_diff"].mean(),
        df_pert["MoS_AP_pct_diff"].mean(),
        df_pert["ML_L_pct_diff"].mean(),
        df_pert["AP_L_pct_diff"].mean()
    ]
})

print("\n==============================")
print("SUMMARY TABLE")
print("==============================")
print(summary_df)

# Save into same Excel file (new sheet)
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    summary_df.to_excel(writer, sheet_name="Perturbation_Summary", index=False)

# =========================
# SUMMARY OF MAX DIFFERENCES
# =========================
print("\n==============================")
print("MAX PERCENT DIFFERENCE (PERTURBED vs NORMAL)")
print("==============================")

print(f"MoS ML: {df_pert['MoS_ML_pct_diff'].max():.2f}%")
print(f"MoS AP: {df_pert['MoS_AP_pct_diff'].max():.2f}%")
print(f"ML Angular Momentum: {df_pert['ML_L_pct_diff'].max():.2f}%")
print(f"AP Angular Momentum: {df_pert['AP_L_pct_diff'].max():.2f}%")

# =========================
# OPTIONAL: MEAN DIFFERENCE
# =========================
print("\n--- Mean Percent Difference ---")

print(f"MoS ML: {df_pert['MoS_ML_pct_diff'].mean():.2f}%")
print(f"MoS AP: {df_pert['MoS_AP_pct_diff'].mean():.2f}%")
print(f"ML Angular Momentum: {df_pert['ML_L_pct_diff'].mean():.2f}%")
print(f"AP Angular Momentum: {df_pert['AP_L_pct_diff'].mean():.2f}%")

# =========================
# PLOTTING (PERTURBED vs NORMAL)
# =========================
time = df_pert["time_s"]
time_norm = df_norm["time_s"]

plt.figure(figsize=(12,12))

# -------------------------
# ML MoS
# -------------------------
plt.subplot(4,1,1)
plt.plot(time, df_pert["MoS_ML"], label="Perturbed")
plt.plot(time_norm, df_norm["MoS_ML"],
         linestyle='--', color='gray', alpha=0.6, label="Normal")
plt.title("ML MoS (m)")
plt.ylabel("Meters (m)")
plt.legend()
plt.grid()

# -------------------------
# AP MoS
# -------------------------
plt.subplot(4,1,2)
plt.plot(time, df_pert["MoS_AP"], label="Perturbed")
plt.plot(time_norm, df_norm["MoS_AP"],
         linestyle='--', color='gray', alpha=0.6, label="Normal")
plt.title("AP MoS (m)")
plt.ylabel("Meters (m)")
plt.legend()
plt.grid()

# -------------------------
# ML Angular Momentum
# -------------------------
plt.subplot(4,1,3)
plt.plot(time, df_pert["ML_Angular_Momentum"], label="Perturbed")
plt.plot(time_norm, df_norm["ML_Angular_Momentum"],
         linestyle='--', color='gray', alpha=0.6, label="Normal")
plt.title("ML Angular Momentum (kg·m²/s)")
plt.ylabel("kg·m²/s")
plt.legend()
plt.grid()

# -------------------------
# AP Angular Momentum
# -------------------------
plt.subplot(4,1,4)
plt.plot(time, df_pert["AP_Angular_Momentum"], label="Perturbed")
plt.plot(time_norm, df_norm["AP_Angular_Momentum"],
         linestyle='--', color='gray', alpha=0.6, label="Normal")
plt.title("AP Angular Momentum (kg·m²/s)")
plt.ylabel("kg·m²/s")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

# =========================
# SUMMARY PLOT (MAX EFFECT SIZE)
# =========================

plt.figure(figsize=(8,5))

plt.bar(
    summary_df["Metric"],
    summary_df["Max_%_Difference"]
)

plt.title("Maximum Perturbation Effect (Mimicry vs Normal)")
plt.ylabel("Max % Difference")
plt.xticks(rotation=30)
plt.grid(axis="y")

summary_plot_path = PLOT_PATH.replace(".png", "_summary_bar.png")
plt.tight_layout()
plt.savefig(summary_plot_path)
plt.show()

print(f"\nSaved summary plot → {summary_plot_path}")

# =========================
# EXPORT
# =========================
df_pert.to_excel(OUTPUT_XLSX, index=False)
print("\nResults exported ✅")