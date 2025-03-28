import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy

from lib.dataset.load_BP2004_model import load_BP2004_model
from lib.misc.paths import datasets_root_path
from lib.signal_processing.misc import zoom_and_crop

segy_file_path = datasets_root_path.joinpath("BP2004/vel_z6.25m_x12.5m_exact.segy")
velocity_model = load_BP2004_model(segy_file_path) / 1000.0
assert 1.4 < np.mean(velocity_model) < 4.8

size = (276, 76)

resized_velocity_model = zoom_and_crop(velocity_model[:, 1800:-1500], (size[1] * 2, size[0]))[::2]

# 縦に二倍して表示
plt.imshow(np.repeat(resized_velocity_model, 2, axis=0), cmap="coolwarm")
plt.colorbar(label="Velocity (m/s)")
plt.xlabel("Distance (samples)")
plt.ylabel("Depth (samples)")
plt.title("BP2004 Velocity Model from SEGY")
plt.show()

# 速度の統計情報
mean_velocity = np.mean(velocity_model)
max_velocity = np.max(velocity_model)
min_velocity = np.min(velocity_model)

print(f"平均速度: {mean_velocity:.2f} m/s")
print(f"最大速度: {max_velocity:.2f} m/s")
print(f"最小速度: {min_velocity:.2f} m/s")
