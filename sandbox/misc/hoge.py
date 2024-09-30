import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lib.misc import output_path


def load(filename: str, label: str) -> npt.NDArray:
    path = output_path.joinpath(filename)
    data = np.load(path)
    return data[label]

files = {
    'proposed method': '2024-09-29_15-25-45,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=2000,sigma=0,.npz',
    'normal FWI': '2024-09-29_15-34-28,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=2000,sigma=0,.npz',
}

vm_diffs = list(map(lambda x: load(x, 'arr_2'), files.values()))
objectives = list(map(lambda x: load(x, 'arr_3'), files.values()))
psnrs = list(map(lambda x: load(x, 'arr_4'), files.values()))
ssims = list(map(lambda x: load(x, 'arr_5'), files.values()))

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

for i, label in enumerate(files.keys()):
    plt.plot(ssims[i], label=label)
plt.title("velocity model ssim")
plt.xlabel("iterations")
plt.legend()
plt.show()

max_iters = min(list(map(lambda x: len(x), vm_diffs)))

vm_diffs_diff = list(map(lambda x: x[:max_iters] - vm_diffs[-1][:max_iters], vm_diffs[:-1]))
objectives_diff = list(map(lambda x: x[:max_iters] - objectives[-1][:max_iters], objectives[:-1]))
psnrs_diff = list(map(lambda x: x[:max_iters] - psnrs[-1][:max_iters], psnrs[:-1]))
ssims_diff = list(map(lambda x: x[:max_iters] - ssims[-1][:max_iters], ssims[:-1]))

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

for i, label in enumerate(list(files.keys())[:-1]):
    plt.plot(ssims_diff[i], label=label)
plt.title("velocity model ssim diff")
plt.xlabel("iterations")
max_ssim_diff = max(0, np.max(np.array(ssims_diff))) * 1.2
plt.ylim(-max_ssim_diff, max_ssim_diff)
plt.legend()
plt.show()
