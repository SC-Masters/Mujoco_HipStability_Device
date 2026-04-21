import numpy as np
from pathlib import Path

npz_path = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\perturbed_npy_arrays\ML_mimic\ML_mimic_trajectory.npz"
out_dir = Path(r"perturbed_npy_arrays\ML_mimic")
out_dir.mkdir(exist_ok=True)

data = np.load(npz_path, allow_pickle=True)

for key in data.files:
    np.save(out_dir / f"{key}.npy", data[key], allow_pickle=True)

print("Done ✅")