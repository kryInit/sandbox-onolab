import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lib.misc import output_path


def load(filename: str, label: str) -> npt.NDArray:
    path = output_path.joinpath(filename)
    data = np.load(path)
    return data[label]


filename0 = "gamma1=1e-05,gamma2=10,2024-07-21_02-58-12.npz"
filename1 = "gamma1=1e-05,gamma2=0.1,2024-07-21_11-23-59.npz"
filename2 = "gamma1=1e-05,gamma2=0,2024-07-22_23-18-50.npz"
filename3 = "gamma1=0.00001,gamma2=1.npz"

vm_diff0 = load(filename0, "arr_2")
vm_diff1 = load(filename1, "arr_2")
vm_diff2 = load(filename2, "arr_2")
vm_diff3 = load(filename3, "arr_2")

objective0 = load(filename0, "arr_3")
objective1 = load(filename1, "arr_3")
objective2 = load(filename2, "arr_3")
objective3 = load(filename3, "arr_3")

plt.plot(objective0, label="gamma2=10")
plt.plot(objective3, label="gamma2=1")
plt.plot(objective1, label="gamma2=0.1")
plt.plot(objective2, label="gradient method")
plt.title("objetive(residual observed waveform), gamma1=1e-5")
plt.xlim(300, 3000)
plt.ylim(0, 7000)
plt.legend()
plt.show()

plt.plot(vm_diff0, label="gamma2=10")
plt.plot(vm_diff3, label="gamma2=1")
plt.plot(vm_diff1, label="gamma2=0.1")
plt.plot(vm_diff2, label="gradient method")
plt.xlim(300, 4500)
plt.ylim(115, 180)
plt.title("velocity model difference l2 norm, gamma1=1e-5")
plt.xlabel("iterations")
plt.legend()
plt.show()
