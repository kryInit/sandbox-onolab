import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lib.misc import output_path


def load(filename: str, label: str) -> npt.NDArray:
    path = output_path.joinpath(filename)
    data = np.load(path)
    return data[label]


files = {
    # 'gamma2=1': '2024-08-04_21-58-09,nshots=10,gamma1=1e-05,gamma2=1.npz',
    # 'gamma2=0.1': '2024-08-04_23-12-04,nshots=10,gamma1=1e-05,gamma2=0.1.npz',
    # 'gamma2=0.05': '2024-08-04_22-53-58,nshots=10,gamma1=1e-05,gamma2=0.05.npz',
    # 'gamma2=0.03': '2024-08-04_22-19-32,nshots=10,gamma1=1e-05,gamma2=0.03.npz',
    # 'gamma2=0.01': '2024-08-04_22-37-26,nshots=10,gamma1=1e-05,gamma2=0.01.npz',

    # 'gamma2=0.003': '2024-08-06_07-06-45,nshots=10,gamma1=1e-05,gamma2=0.003.npz',
    # 'gamma2=0.002': '2024-08-06_06-47-37,nshots=10,gamma1=1e-05,gamma2=0.002.npz',
    # 'gamma2=0.001': '2024-08-04_23-48-46,nshots=10,gamma1=1e-05,gamma2=0.001.npz',
    # 'gamma2=0.0008': '2024-08-05_00-47-00,nshots=10,gamma1=1e-05,gamma2=0.0008.npz',
    # 'gamma2=0.0005': '2024-08-05_00-26-32,nshots=10,gamma1=1e-05,gamma2=0.0005.npz',
    # 'gamma2=0.0003': '2024-08-05_01-07-34,nshots=10,gamma1=1e-05,gamma2=0.0003.npz',
    # 'gamma2=0.0001': '2024-08-05_00-05-25,nshots=10,gamma1=1e-05,gamma2=0.0001.npz',
    # 'gamma2=None': '2024-08-04_21-38-15,nshots=10,gamma1=1e-05,gamma2=None.npz'


    # 'gamma2=0.002': '2024-08-06_07-53-26,nshots=10,gamma1=1e-05,gamma2=0.002.npz',
    # 'gamma2=None': '2024-08-06_08-43-50,nshots=10,gamma1=1e-05,gamma2=None.npz',

    # 'gamma2=0.002': '2024-08-06_09-14-35,nshots=5,gamma1=1e-05,gamma2=0.002.npz',
    # 'gamma2=None': '2024-08-06_09-04-27,nshots=5,gamma1=1e-05,gamma2=None.npz',

    # 'gamma2=0.002': '2024-08-06_10-26-14,nshots=24,gamma1=1e-05,gamma2=0.002.npz',
    # 'gamma2=None': '2024-08-06_11-47-59,nshots=24,gamma1=1e-05,gamma2=None.npz',

    # 'gamma2=0.002': '2024-08-06_13-02-53,nshots=24,gamma1=1e-05,gamma2=0.002.npz',
    # 'gamma2=None': '2024-08-06_14-52-04,nshots=24,gamma1=1e-05,gamma2=None.npz',

    # 'gamma2=0.002': '2024-08-06_16-03-32,nshots=10,gamma1=1e-05,gamma2=0.002,niters=10000,sigma=5.npz',
    # 'gamma2=None': '2024-08-06_15-47-01,nshots=10,gamma1=1e-05,gamma2=None,niters=10000,sigma=5.npz',

    # 'gamma2=0.002,sigma=8': '2024-08-07_13-29-39,nshots=24,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=8.npz',
    # 'gamma2=None,sigma=8': '2024-08-07_06-28-28,nshots=24,gamma1=1e-05,gamma2=None,niters=30000,sigma=8.npz',

    # 'gamma2=0.002,sigma=5': '2024-08-07_11-43-05,nshots=24,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=5.npz',
    # 'gamma2=None,sigma=5': '2024-08-07_04-45-25,nshots=24,gamma1=1e-05,gamma2=None,niters=30000,sigma=5.npz',

    # 'gamma2=0.002,sigma=3': '2024-08-07_09-56-55,nshots=24,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=3.npz',
    # 'gamma2=None,sigma=3': '2024-08-07_03-01-08,nshots=24,gamma1=1e-05,gamma2=None,niters=30000,sigma=3.npz',

    # 'gamma1=1e-4': '2024-08-09_03-35-20,nshots=24,gamma1=0.0001,gamma2=0.003,niters=30000,sigma=1.npz',
    # 'gamma1=5e-5': '2024-08-09_07-05-57,nshots=24,gamma1=5e-05,gamma2=0.003,niters=30000,sigma=1.npz',
    # 'gamma1=1e-5': '2024-08-07_15-16-33,nshots=24,gamma1=1e-05,gamma2=0.003,niters=30000,sigma=1.npz',
    # 'gamma1=1e-6': '2024-08-09_05-20-44,nshots=24,gamma1=1e-06,gamma2=0.003,niters=30000,sigma=1.npz',

    # 'gamma2=0.003': '2024-08-09_12-21-41,nshots=24,gamma1=1e-05,gamma2=0.003,niters=60000,sigma=1.npz',
    # 'gamma2=0.002': '2024-08-09_14-07-42,nshots=24,gamma1=1e-05,gamma2=0.002,niters=60000,sigma=1.npz',
    # 'gamma2=0.001': '2024-08-09_15-54-53,nshots=24,gamma1=1e-05,gamma2=0.001,niters=60000,sigma=1.npz',
    # 'gamma2=None': '2024-08-09_10-37-30,nshots=24,gamma1=1e-05,gamma2=None,niters=60000,sigma=1.npz',

    # 'gamma2=100': '2024-08-10_20-17-53,nshots=24,gamma1=1e-05,gamma2=100,niters=30000,sigma=1.npz',
    # 'gamma2=50': '2024-08-10_18-34-04,nshots=24,gamma1=1e-05,gamma2=50,niters=30000,sigma=1.npz',
    # 'gamma2=10': '2024-08-10_16-50-49,nshots=24,gamma1=1e-05,gamma2=10,niters=30000,sigma=1.npz',
    # 'gamma2=5': '2024-08-10_15-05-38,nshots=24,gamma1=1e-05,gamma2=5,niters=30000,sigma=1.npz',
    # 'gamma2=1': '2024-08-10_08-56-04,nshots=24,gamma1=1e-05,gamma2=1,niters=30000,sigma=1.npz',
    # 'gamma2=0.8': '2024-08-10_10-41-36,nshots=24,gamma1=1e-05,gamma2=0.8,niters=30000,sigma=1.npz',
    # 'gamma2=0.5': '2024-08-10_05-28-00,nshots=24,gamma1=1e-05,gamma2=0.5,niters=30000,sigma=1.npz',
    # 'gamma2=0.2': '2024-08-10_03-44-26,nshots=24,gamma1=1e-05,gamma2=0.2,niters=30000,sigma=1.npz',
    # 'gamma2=0.1': '2024-08-10_01-59-31,nshots=24,gamma1=1e-05,gamma2=0.1,niters=30000,sigma=1.npz',
    # 'gamma2=0.08': '2024-08-10_07-11-37,nshots=24,gamma1=1e-05,gamma2=0.08,niters=30000,sigma=1.npz',
    # 'gamma2=0.05':  '2024-08-09_18-58-35,nshots=24,gamma1=1e-05,gamma2=0.05,niters=30000,sigma=1.npz',
    # 'gamma2=0.04':  '2024-08-09_20-42-23,nshots=24,gamma1=1e-05,gamma2=0.04,niters=30000,sigma=1.npz',
    # 'gamma2=0.03':  '2024-08-09_22-26-18,nshots=24,gamma1=1e-05,gamma2=0.03,niters=30000,sigma=1.npz',
    # 'gamma2=0.025': '2024-08-10_00-10-30,nshots=24,gamma1=1e-05,gamma2=0.025,niters=30000,sigma=1.npz',
    # 'gamma2=0.01':  '2024-08-09_01-50-11,nshots=24,gamma1=1e-05,gamma2=0.01,niters=30000,sigma=1.npz',
    # 'gamma2=0.008': '2024-08-09_00-05-22,nshots=24,gamma1=1e-05,gamma2=0.008,niters=30000,sigma=1.npz',
    # 'gamma2=0.005': '2024-08-08_22-20-04,nshots=24,gamma1=1e-05,gamma2=0.005,niters=30000,sigma=1.npz',
    # 'gamma2=0.003': '2024-08-07_15-16-33,nshots=24,gamma1=1e-05,gamma2=0.003,niters=30000,sigma=1.npz',
    # 'gamma2=0.002': '2024-08-07_08-12-27,nshots=24,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=1.npz',
    # 'gamma2=0.001': '2024-08-07_17-03-29,nshots=24,gamma1=1e-05,gamma2=0.001,niters=30000,sigma=1.npz',
    # 'gamma2=None':  '2024-08-07_01-19-09,nshots=24,gamma1=1e-05,gamma2=None,niters=30000,sigma=1.npz',

    # 'gamma2=0.003': '2024-08-06_21-48-41,nshots=24,gamma1=1e-05,gamma2=0.003,niters=30000,sigma=0.npz',
    # 'gamma2=0.002': '2024-08-06_20-03-13,nshots=24,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=0.npz',
    # 'gamma2=0.001': '2024-08-06_23-35-13,nshots=24,gamma1=1e-05,gamma2=0.001,niters=30000,sigma=0.npz',
    # 'gamma2=None': '2024-08-06_18-17-57,nshots=24,gamma1=1e-05,gamma2=None,niters=30000,sigma=0.npz',

    # 'gamma2=0.003': '2024-08-08_06-14-00,nshots=10,gamma1=1e-05,gamma2=0.003,niters=30000,sigma=0.npz',
    # 'gamma2=0.002': '2024-08-08_05-23-29,nshots=10,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=0.npz',
    # 'gamma2=0.001': '2024-08-08_07-04-55,nshots=10,gamma1=1e-05,gamma2=0.001,niters=30000,sigma=0.npz',
    # 'gamma2=None': '2024-08-08_04-32-59,nshots=10,gamma1=1e-05,gamma2=None,niters=30000,sigma=0.npz',

    # 'gamma2=0.003': '2024-08-08_14-46-54,nshots=10,gamma1=1e-05,gamma2=0.003,niters=30000,sigma=1.npz',
    # 'gamma2=0.002': '2024-08-08_11-15-49,nshots=10,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=1.npz',
    # 'gamma2=0.001': '2024-08-08_15-40-04,nshots=10,gamma1=1e-05,gamma2=0.001,niters=30000,sigma=1.npz',
    # 'gamma2=None': '2024-08-08_07-54-45,nshots=10,gamma1=1e-05,gamma2=None,niters=30000,sigma=1.npz',

    # 'gamma2=1': '2024-08-11_04-22-54,nshots=24,gamma1=1e-05,gamma2=1,niters=100000,sigma=1.npz',
    # 'gamma2=None': '2024-08-11_00-17-56,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=1.npz',

    'gamma2=100': '2024-08-13_17-47-10,nshots=24,gamma1=1e-05,gamma2=100,niters=100000,sigma=1.npz',
    'gamma2=10': '2024-08-13_06-10-57,nshots=24,gamma1=1e-05,gamma2=10,niters=100000,sigma=1.npz',
    'gamma2=1': '2024-08-12_11-25-36,nshots=24,gamma1=1e-05,gamma2=1,niters=100000,sigma=1.npz',
    'gamma2=0.1': '2024-08-13_11-58-54,nshots=24,gamma1=1e-05,gamma2=0.1,niters=100000,sigma=1.npz',
    'gamma2=0.01': '2024-08-13_23-35-23,nshots=24,gamma1=1e-05,gamma2=0.01,niters=100000,sigma=1.npz',
    'gamma2=None': '2024-08-12_05-36-59,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=1.npz',
    # 'gamma2=1,filter': '2024-08-13_00-05-55,nshots=24,gamma1=1e-05,gamma2=1,niters=100000,sigma=1.npz',
    # 'gamma2=None,filter': '2024-08-12_18-16-59,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=1.npz'
}

vm_diffs = list(map(lambda x: load(x, 'arr_2'), files.values()))
objectives = list(map(lambda x: load(x, 'arr_3'), files.values()))
psnrs = list(map(lambda x: load(x, 'arr_4'), files.values()))

for i, label in enumerate(files.keys()):
    plt.plot(objectives[i], label=label)
plt.title("objetive(residual observed waveform)")
plt.xlabel("iterations")
plt.legend()
plt.show()

for i, label in enumerate(files.keys()):
    plt.plot(vm_diffs[i], label=label)
plt.title("velocity model difference l2 norm")
plt.xlabel("iterations")
plt.legend()
plt.show()

for i, label in enumerate(files.keys()):
    plt.plot(psnrs[i], label=label)
plt.title("velocity model psnr")
plt.xlabel("iterations")
plt.legend()
plt.show()

max_iters = min(list(map(lambda x: len(x), vm_diffs)))

vm_diffs_diff = list(map(lambda x: x[:max_iters] - vm_diffs[-1][:max_iters], vm_diffs[:-1]))
objectives_diff = list(map(lambda x: x[:max_iters] - objectives[-1][:max_iters], objectives[:-1]))
psnrs_diff = list(map(lambda x: x[:max_iters] - psnrs[-1][:max_iters], psnrs[:-1]))

for i, label in enumerate(list(files.keys())[:-1]):
    plt.plot(objectives_diff[i], label=label)
plt.title("objetive(residual observed waveform) diff")
plt.xlabel("iterations")
min_objectives_diff = min(0, np.min(np.array(objectives_diff))) * 1.2
plt.ylim(min_objectives_diff, -min_objectives_diff)
plt.legend()
plt.show()

for i, label in enumerate(list(files.keys())[:-1]):
    plt.plot(vm_diffs_diff[i], label=label)
plt.title("velocity model difference l2 norm diff")
plt.xlabel("iterations")
min_vm_diffs_diff = min(0, np.min(np.array(vm_diffs_diff))) * 1.2
plt.ylim(min_vm_diffs_diff, -min_vm_diffs_diff)
plt.legend()
plt.show()

for i, label in enumerate(list(files.keys())[:-1]):
    plt.plot(psnrs_diff[i], label=label)
plt.title("velocity model psnr diff")
plt.xlabel("iterations")
max_psnr_diff = max(0, np.max(np.array(psnrs_diff))) * 1.2
plt.ylim(-max_psnr_diff, max_psnr_diff)
plt.legend()
plt.show()
