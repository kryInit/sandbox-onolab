import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lib.dataset import load_seismic_datasets__salt_model
from lib.misc import datasets_root_path, output_path
from lib.signal_processing.misc import calc_psnr, smoothing_with_gaussian_filter, zoom_and_crop

seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")
seismic_data = load_seismic_datasets__salt_model(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0

raw_true_velocity_model = seismic_data[300]
true_velocity_model = zoom_and_crop(raw_true_velocity_model, (50, 100))
initial_velocity_model = zoom_and_crop(smoothing_with_gaussian_filter(seismic_data[300], 1, 80), (50, 100))


def load(filename: str, label: str) -> npt.NDArray:
    path = output_path.joinpath(filename)
    data = np.load(path)
    return data[label]


files = {
    "alpha:700": "2024-11-05_02-22-21,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=700,.npz",
    "alpha:650": "2024-11-05_02-08-49,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=650,.npz",
    "alpha:600": "2024-11-05_01-55-17,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=600,.npz",
    "alpha:550": "2024-11-05_01-41-45,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=550,.npz",
    "alpha:500": "2024-11-05_01-28-14,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=500,.npz",
    "alpha:450": "2024-11-05_01-14-44,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=450,.npz",
    "alpha:400": "2024-11-05_01-01-14,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=400,.npz",
    "alpha:350": "2024-11-05_00-47-46,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=350,.npz",
    "alpha:300": "2024-11-05_00-34-18,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=300,.npz",
    "alpha:250": "2024-11-05_00-20-50,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=250,.npz",
    "alpha:200": "2024-11-05_00-07-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=200,.npz",
    "alpha:150": "2024-11-04_23-54-02,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=150,.npz",
    "alpha:100": "2024-11-04_23-40-41,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=100,.npz",
    "standard FWI": "2024-11-04_23-27-19,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=1,alpha=None,.npz",
    # 'alpha:700': '2024-10-11_18-42-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=700,.npz',
    # 'alpha:650': '2024-10-11_18-28-28,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=650,.npz',
    # 'alpha:600': '2024-10-11_18-14-34,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=600,.npz',
    # 'alpha:550': '2024-10-11_18-00-38,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=550,.npz',
    # 'alpha:500': '2024-10-11_14-44-06,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=500,.npz',
    # 'alpha:450': '2024-10-11_14-30-09,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=450,.npz',
    # 'alpha:400': '2024-10-11_14-16-12,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=400,.npz',
    # 'alpha:350': '2024-10-11_14-02-16,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=350,.npz',
    # 'alpha:300': '2024-10-11_13-48-21,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=300,.npz',
    # 'alpha:250': '2024-10-11_13-34-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=250,.npz',
    # 'alpha:200': '2024-10-11_15-44-41,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=200,.npz',
    # 'alpha:150': '2024-10-11_15-58-37,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=150,.npz',
    # 'alpha:100': '2024-10-11_16-12-35,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=100,.npz',
    # 'alpha:50':  '2024-10-11_16-16-52,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=1501,sigma=0,alpha=50,.npz',
    # 'standard FWI': '2024-10-11_19-32-20,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=0,alpha=None,.npz',
    # 'standard FWI': '2024-10-01_04-25-33,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=0,.npz',
    # 'alpha:340': '2024-10-01_06-31-11,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,.npz',
    # 'proposed method':  '2024-10-03_00-59-48,pds_with_L12norm,nshots=30,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,.npz',
    # 'proposed method0': '2024-10-03_15-02-22,pds_with_L12norm,nshots=30,gamma1=0.0001,gamma2=100,niters=3282,sigma=0,alpha=3500,.npz',
    # 'proposed method1': '2024-10-03_14-18-26,pds_with_L12norm,nshots=30,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=3000,.npz',
    # 'proposed method2': '2024-10-03_13-11-56,pds_with_L12norm,nshots=30,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=2500,.npz',
    # 'proposed method3': '2024-10-03_12-05-53,pds_with_L12norm,nshots=30,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=2000,.npz',
    # "standard FWI'": '2024-10-05_12-54-09,gradient,nshots=30,gamma1=8e-05,gamma2=None,niters=3012,sigma=0,alpha=0,.npz',
    # 'standard FWI': '2024-10-03_01-41-53,gradient,nshots=30,gamma1=0.0001,gamma2=None,niters=3194,sigma=0,.npz',
}

vel_models = list(map(lambda x: load(x, "arr_0"), files.values()))
vm_diffs = list(map(lambda x: load(x, "arr_2"), files.values()))
objectives = list(map(lambda x: load(x, "arr_3"), files.values()))
psnrs = list(map(lambda x: load(x, "arr_4"), files.values()))
ssims = list(map(lambda x: load(x, "arr_5"), files.values()))

# for i, label in enumerate(files.keys()):
#     plt.plot(objectives[i], label=label)
# plt.title("objetive(residual observed waveform)")
# plt.xlabel("iterations")
# plt.legend()
# plt.show()
#
# for i, label in enumerate(files.keys()):
#     plt.plot(vm_diffs[i], label=label)
# plt.title("velocity model difference l2 norm")
# plt.xlabel("iterations")
# plt.legend()
# plt.show()
#
# for i, label in enumerate(files.keys()):
#     plt.plot(psnrs[i], label=label)
# plt.title("velocity model psnr")
# plt.xlabel("iterations")
# plt.legend()
# plt.show()
#
# for i, label in enumerate(files.keys()):
#     plt.plot(ssims[i], label=label)
# # plt.title("velocity model ssim")
# plt.xlabel("iterations")
# plt.ylabel("SSIM")
# plt.legend()
# plt.show()
#
# max_iters = min(list(map(lambda x: len(x), vm_diffs)))
#
# vm_diffs_diff = list(map(lambda x: x[:max_iters] - vm_diffs[-1][:max_iters], vm_diffs[:-1]))
# objectives_diff = list(map(lambda x: x[:max_iters] - objectives[-1][:max_iters], objectives[:-1]))
# psnrs_diff = list(map(lambda x: x[:max_iters] - psnrs[-1][:max_iters], psnrs[:-1]))
# ssims_diff = list(map(lambda x: x[:max_iters] - ssims[-1][:max_iters], ssims[:-1]))
#
# for i, label in enumerate(list(files.keys())[:-1]):
#     plt.plot(objectives_diff[i], label=label)
# plt.title("objetive(residual observed waveform) diff")
# plt.xlabel("iterations")
# min_objectives_diff = min(0, np.min(np.array(objectives_diff))) * 1.2
# plt.ylim(min_objectives_diff, -min_objectives_diff)
# plt.legend()
# plt.show()
#
# for i, label in enumerate(list(files.keys())[:-1]):
#     plt.plot(vm_diffs_diff[i], label=label)
# plt.title("velocity model difference l2 norm diff")
# plt.xlabel("iterations")
# min_vm_diffs_diff = min(0, np.min(np.array(vm_diffs_diff))) * 1.2
# plt.ylim(min_vm_diffs_diff, -min_vm_diffs_diff)
# plt.legend()
# plt.show()
#
# for i, label in enumerate(list(files.keys())[:-1]):
#     plt.plot(psnrs_diff[i], label=label)
# plt.title("velocity model psnr diff")
# plt.xlabel("iterations")
# max_psnr_diff = max(0, np.max(np.array(psnrs_diff))) * 1.2
# plt.ylim(-max_psnr_diff, max_psnr_diff)
# plt.legend()
# plt.show()
#
# for i, label in enumerate(list(files.keys())[:-1]):
#     plt.plot(ssims_diff[i], label=label)
# plt.title("velocity model ssim diff")
# plt.xlabel("iterations")
# max_ssim_diff = max(0, np.max(np.array(ssims_diff))) * 1.2
# plt.ylim(-max_ssim_diff, max_ssim_diff)
# plt.legend()
# plt.show()

scale = 0.5

fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
alpha = list(map(lambda x: int(x.split("alpha:")[1]), list(files.keys())[:-1]))
ssims_last = list(map(lambda x: x[-1], ssims[:-1]))
# plt.plot([100, 700], [ssims[-1][-1], ssims[-1][-1]], color='black')
plt.axhline(y=ssims[-1][-1], color="black", linestyle="--", label="Standard FWI method")
plt.plot(alpha, ssims_last, label="Proposed method", color="#00a381")
# plt.title("velocity model ssim")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
plt.xlabel("α", fontsize=15)
plt.ylabel("SSIM", fontsize=15)
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
plt.plot(ssims[-1], label="Standard FWI method", color="#7a4171")
plt.plot(ssims[7], label="Proposed method", color="#00a381")
# plt.title("velocity model ssim")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
plt.xlabel("iterations", fontsize=15)
plt.ylabel("SSIM", fontsize=15)

plt.legend()
plt.show()

# plt.figure(figsize=(16*scale, 10*scale))
# alpha = list(map(lambda x: int(x.split('alpha:')[1]), list(files.keys())[:-1]))
# ssims_last = list(map(lambda x: x[-1], psnrs[:-1]))
# # plt.plot([100, 700], [ssims[-1][-1], ssims[-1][-1]], color='black')
# plt.axhline(y=psnrs[-1][-1], color='black', linestyle='--', label="standard FWI")
# plt.plot(alpha, ssims_last, label="proposed method", color='#00a381')
# # plt.title("velocity model ssim")
# plt.xlabel("alpha")
# plt.ylabel("PSNR")
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(16*scale, 10*scale))
# plt.plot(psnrs[-1], label='standard FWI', color='#7a4171')
# plt.plot(psnrs[7], label='proposed method', color='#00a381')
# # plt.title("velocity model ssim")
# plt.xlabel("iterations")
# plt.ylabel("PSNR")
# plt.legend()
# plt.show()

# depth = 25
# plt.figure(figsize=(16*scale, 10*scale))
# plt.plot(vel_models[-1][depth+40][40:-40], label='standard FWI')
# # plt.plot(vel_models[3][depth+40][40:-40], label='alpha=550')
# plt.plot(vel_models[7][depth+40][40:-40], label='alpha=350')
# # plt.plot(vel_models[11][depth+40][40:-40], label='alpha=150')
# plt.plot(true_velocity_model[depth], label='true', color='black')
# # plt.plot(initial_velocity_model[depth], label='initial')
# plt.title("velocity model")
# plt.xlabel("x")
# plt.ylabel("velocity")
# plt.legend()
# plt.show()
