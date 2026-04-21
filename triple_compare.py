import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
MIMIC_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\ML_mimic\full_MLmimic_stability_results.xlsx"  # same as perturbed if already embedded
NORMAL_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\normal_data\full_normal_stability_results.xlsx"
PERTURBED_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\ML_data\full_ML_stability_results.xlsx"
OUTPUT_XLSX = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\comparison_data\MLmimic_stability_comparison_output.xlsx"
PLOT_PATH = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\comparison_data\MLmimic_stability_comparison_plot_better.png"

g = 9.81

# =========================
# PARAMETERS
# =========================
SHOE_LENGTH = 0.25
SHOE_WIDTH = 0.10
TRANSIENT_TIME = 1.0

Height = 1.5
Weight = 70

# =========================================================
# SAME PROCESS FUNCTION (UNCHANGED CORE LOGIC)
# =========================================================
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
    L_total = None
    for seg in omega:
        L_seg = omega[seg]
        L_total = L_seg if L_total is None else L_total + L_seg
    return L_total


def process_df(df):

    df = df[df["time_s"] > TRANSIENT_TIME].reset_index(drop=True)

    origin_x = (df["ankle_angle_l_x"] + df["ankle_angle_r_x"]) / 2
    origin_y = (df["ankle_angle_l_y"] + df["ankle_angle_r_y"]) / 2

    CoM_x_local = df["CoM_x"] - origin_x
    CoM_y_local = df["CoM_y"] - origin_y

    calcn_l_x = df["ankle_angle_l_x"] - origin_x
    calcn_l_y = df["ankle_angle_l_y"] - origin_y
    calcn_r_x = df["ankle_angle_r_x"] - origin_x
    calcn_r_y = df["ankle_angle_r_y"] - origin_y

    l = np.mean(df["CoM_z"] - df["ankle_angle_r_z"])
    omega0 = np.sqrt(g / l)

    df["XCoM_AP"] = CoM_x_local + df["CoM_vx"] / omega0
    df["XCoM_ML"] = CoM_y_local + df["CoM_vy"] / omega0

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

    df["MoS_AP"] = np.minimum(BoS_ant - df["XCoM_AP"], df["XCoM_AP"] - BoS_post)
    df["MoS_ML"] = np.minimum(BoS_lat - df["XCoM_ML"], df["XCoM_ML"] - BoS_med)

    omega_segments = compute_segment_omega(df)
    L_rot = compute_rotational_L(omega_segments)

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
# LOAD DATA (3 CONDITIONS)
# =========================
df_norm = process_df(pd.read_excel(NORMAL_XLSX))
df_pert = process_df(pd.read_excel(PERTURBED_XLSX))
df_mimic = process_df(pd.read_excel(MIMIC_XLSX))

# align length safety
min_len = min(len(df_norm), len(df_pert), len(df_mimic))
df_norm = df_norm.iloc[:min_len]
df_pert = df_pert.iloc[:min_len]
df_mimic = df_mimic.iloc[:min_len]


# =========================
# METRICS FUNCTION
# =========================
def metrics(df):
    return {
        "MoS_ML": df["MoS_ML"].values,
        "MoS_AP": df["MoS_AP"].values,
        "ML_L": df["ML_Angular_Momentum"].values,
        "AP_L": df["AP_Angular_Momentum"].values
    }


n = metrics(df_norm)
p = metrics(df_pert)
m = metrics(df_mimic)


# =========================
# SUMMARY TABLE (3-WAY)
# =========================
summary = pd.DataFrame({
    "Metric": ["MoS_ML", "MoS_AP", "AngMom_ML", "AngMom_AP"],

    "Normal_Max": [
        np.max(np.abs(n["MoS_ML"])),
        np.max(np.abs(n["MoS_AP"])),
        np.max(np.abs(n["ML_L"])),
        np.max(np.abs(n["AP_L"]))
    ],

    "Perturbed_Max": [
        np.max(np.abs(p["MoS_ML"])),
        np.max(np.abs(p["MoS_AP"])),
        np.max(np.abs(p["ML_L"])),
        np.max(np.abs(p["AP_L"]))
    ],

    "Mimic_Max": [
        np.max(np.abs(m["MoS_ML"])),
        np.max(np.abs(m["MoS_AP"])),
        np.max(np.abs(m["ML_L"])),
        np.max(np.abs(m["AP_L"]))
    ],
})

summary["Perturbation_Impact"] = summary["Perturbed_Max"] - summary["Normal_Max"]
summary["Mimic_Reduction"] = summary["Perturbed_Max"] - summary["Mimic_Max"]

print(summary)

summary.to_excel(OUTPUT_XLSX, index=False)


# =========================
# PLOTS (3 CONDITIONS)
# =========================
time = df_norm["time_s"]

plt.figure(figsize=(12,10))

labels = ["Normal", "Perturbed", "Mimic"]
colors = ["black", "red", "green"]

for df, c, l in zip([df_norm, df_pert, df_mimic], colors, labels):

    # -------------------------
    # ML MoS
    # -------------------------
    plt.subplot(4,1,1)
    plt.plot(time, df["MoS_ML"], color=c, label=l)
    plt.title("Mediolateral Margin of Stability")
    plt.ylabel("MoS ML (m)")

    # -------------------------
    # AP MoS
    # -------------------------
    plt.subplot(4,1,2)
    plt.plot(time, df["MoS_AP"], color=c, label=l)
    plt.title("Anteroposterior Margin of Stability")
    plt.ylabel("MoS AP (m)")

    # -------------------------
    # ML Angular Momentum
    # -------------------------
    plt.subplot(4,1,3)
    plt.plot(time, df["ML_Angular_Momentum"], color=c, label=l)
    plt.title("Mediolateral Angular Momentum")
    plt.ylabel("Angular Momentum (kg·m²/s)")

    # -------------------------
    # AP Angular Momentum
    # -------------------------
    plt.subplot(4,1,4)
    plt.plot(time, df["AP_Angular_Momentum"], color=c, label=l)
    plt.title("Anteroposterior Angular Momentum")
    plt.ylabel("Angular Momentum (kg·m²/s)")
    plt.xlabel("Time (s)")

# -------------------------
# Styling
# -------------------------
for i in range(1,5):
    plt.subplot(4,1,i)
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()


print("\nDONE ✔ 3-condition stability comparison generated")