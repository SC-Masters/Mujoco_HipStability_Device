import numpy as np
from pathlib import Path

npz_path = r"C:\Users\Masked Gentleman\PycharmProjects\PythonProject\.venv\Lib\site-packages\loco_mujoco\datasets\walk.npz"
out_dir = Path("npy_arrays")
out_dir.mkdir(exist_ok=True)

data = np.load(npz_path, allow_pickle=True)

for key in data.files:
    np.save(out_dir / f"{key}.npy", data[key], allow_pickle=True)

print("Done ✅")