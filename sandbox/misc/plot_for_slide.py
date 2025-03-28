import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy._typing import NDArray

import lib.signal_processing.diff_operator as diff_op
from lib.dataset import load_seismic_datasets__salt_model
from lib.dataset.load_BP2004_model import load_BP2004_model
from lib.misc import datasets_root_path, output_path
from lib.signal_processing.misc import calc_psnr, smoothing_with_gaussian_filter, zoom_and_crop
from lib.signal_processing.norm import L12_norm
from lib.visualize import show_minimum_velocity_model, show_velocity_model

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
    # 'gdm,0': '2025-03-02_19-25-41,gd_nesterov,nshots=69,gamma1=1e-05,gamma2=None,niters=2000,sigma=0,alpha=1200,.npz',
    # 'gdm,1': '2025-03-02_21-17-28,gd_nesterov,nshots=69,gamma1=1e-05,gamma2=None,niters=2000,sigma=1,alpha=1200,.npz',
    # 'gdm,5': '2025-03-02_22-36-18,gd_nesterov,nshots=69,gamma1=1e-05,gamma2=None,niters=2000,sigma=5,alpha=1200,.npz',
    #
    # 'pds,0': '2025-03-02_20-02-55,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=0,alpha=1200,.npz',
    # 'pds,1': '2025-03-02_20-40-26,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=1,alpha=1200,.npz',
    # 'pds,5': '2025-03-02_21-57-35,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=1200,.npz',
    # 'pds,0': '2025-03-03_02-24-42,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=0,alpha=1200,.npz'
    # 'pds,2600': '2025-03-03_18-59-59,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=2600,.npz',
    # 'pds,2400': '2025-03-03_18-23-03,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=2400,.npz',
    # 'pds,2200': '2025-03-03_17-46-04,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=2200,.npz',
    # 'pds,2000': '2025-03-03_17-09-10,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=2000,.npz',
    # 'pds,1800': '2025-03-03_16-32-16,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=1800,.npz',
    # 'pds,1600': '2025-03-03_15-55-13,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=1600,.npz',
    # 'pds,1400': '2025-03-03_15-18-21,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=1400,.npz',
    # 'pds,1200': '2025-03-03_14-41-31,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=1200,.npz',
    # 'pds,1000': '2025-03-03_14-04-27,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=1000,.npz',
    # 'pds,800': '2025-03-03_13-26-52,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=800,.npz',
    # 'pds,600': '2025-03-03_12-49-41,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=600,.npz',
    # 'pds,400': '2025-03-03_12-12-51,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=400,.npz',
    # 'pds,200': '2025-03-03_11-36-01,pds_nesterov,nshots=69,gamma1=1e-05,gamma2=100,niters=2000,sigma=5,alpha=200,.npz',
    # # 'pds,noisy,2000': '2025-03-16_04-32-14,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=2000,.npz',
    # # 'pds,noisy,1900': '2025-03-19_18-31-59,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1900,.npz',
    # 'pds,noisy,1800': '2025-03-16_02-55-30,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1800,.npz',
    # # 'pds,noisy,1700': '2025-03-19_16-55-01,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1700,.npz',
    # # 'pds,noisy,1600': '2025-03-16_01-18-34,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1600,.npz',
    # # 'pds,noisy,1500': '2025-03-19_15-18-23,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1500,.npz',
    # 'pds,noisy,1400': '2025-03-15_23-41-41,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1400,.npz',
    # # 'pds,noisy,1300': '2025-03-15_22-04-36,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1300,.npz',
    # # 'pds,noisy,1200': '2025-03-15_20-27-46,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1200,.npz',
    # # 'pds,noisy,1100': '2025-03-15_18-51-04,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1100,.npz',
    # 'pds,noisy,1000': '2025-03-15_17-14-16,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=1000,.npz',
    # # 'pds,noisy,900': '2025-03-18_23-57-43,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=900,.npz',
    # # 'pds,noisy,800': '2025-03-15_15-37-20,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=800,.npz',
    # 'gd,noisy': '2025-03-15_14-00-09,gd_nesterov,nshots=69,gamma1=0.0001,gamma2=None,niters=397,sigma=1,alpha=0,.npz',
    "pds,30000": "2025-03-21_15-01-44,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=30000,.npz",
    "pds,9000": "2025-03-20_22-41-22,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=9000,.npz",
    "pds,8600": "2025-03-20_21-05-05,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=8600,.npz",
    "pds,8200": "2025-03-20_19-29-12,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=8200,.npz",
    "pds,7800": "2025-03-20_17-52-51,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=7800,.npz",
    "pds,7400": "2025-03-20_16-16-47,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=7400,.npz",
    "pds,7000": "2025-03-20_14-40-20,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=7000,.npz",
    "pds,6600": "2025-03-20_13-03-36,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=6600,.npz",
    "pds,6200": "2025-03-20_11-26-37,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=6200,.npz",
    "pds,5800": "2025-03-20_09-49-26,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=5800,.npz",
    "pds,5400": "2025-03-20_08-12-06,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=5400,.npz",
    "pds,5000": "2025-03-20_06-34-45,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=5000,.npz",
    "pds,4600": "2025-03-20_04-57-28,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=4600,.npz",
    "pds,4200": "2025-03-20_03-20-11,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=4200,.npz",
    "pds,3800": "2025-03-20_01-42-52,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=3800,.npz",
    "pds,3400": "2025-03-19_02-22-50,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=3400,.npz",
    # 'pds,3200': '2025-03-19_13-41-27,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=3200,.npz',
    "pds,3000": "2025-03-19_03-59-43,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=3000,.npz",
    # 'pds,2800': '2025-03-19_12-04-35,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=2800,.npz',
    "pds,2600": "2025-03-19_05-36-38,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=2600,.npz",
    # 'pds,2400': '2025-03-19_10-27-50,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=2400,.npz',
    "pds,2200": "2025-03-19_07-13-46,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=2200,.npz",
    # 'pds,2000': '2025-03-07_18-01-49,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=2000,.npz',
    # 'pds,1900': '2025-03-18_22-20-49,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1900,.npz',
    "pds,1800": "2025-03-07_16-26-39,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1800,.npz",
    # 'pds,1700': '2025-03-18_20-43-44,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1700,.npz',
    # 'pds,1600': '2025-03-07_14-51-19,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1600,.npz',
    # 'pds,1500': '2025-03-18_19-06-23,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1500,.npz',
    "pds,1400": "2025-03-07_13-16-15,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1400,.npz",
    # 'pds,1300': '2025-03-07_11-41-23,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1300,.npz',
    # 'pds,1200': '2025-03-06_16-47-28,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1200,.npz',
    # 'pds,1100': '2025-03-07_10-07-20,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1100,.npz',
    "pds,1000": "2025-03-07_08-32-43,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1000,.npz",
    # 'pds,900': '2025-03-18_15-51-45,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=900,.npz',
    "pds,800": "2025-03-07_06-58-42,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=800,.npz",
    # 'pds,600': '2025-03-19_08-50-47,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=600,.npz',
    "gd": "2025-03-06_06-47-02,gradient,nshots=69,gamma1=0.0001,gamma2=None,niters=10000,sigma=0,alpha=500,.npz",
    # 'pds': '2025-03-06_18-21-38,pds,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1200,.npz',
    # 'pds_nesterov': '2025-03-06_16-47-28,pds_nesterov,nshots=69,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=1200,.npz',
    # 'pds0': '2025-03-07_02-39-08,pds_nesterov,nshots=69,gamma1=0.0006,gamma2=100,niters=5000,sigma=0,alpha=1200,.npz',
    # 'pds1': '2025-03-07_01-05-05,pds_nesterov,nshots=69,gamma1=0.0004,gamma2=100,niters=5000,sigma=0,alpha=1200,.npz',
    # 'pds2': '2025-03-06_23-31-30,pds_nesterov,nshots=69,gamma1=0.0002,gamma2=100,niters=5000,sigma=0,alpha=1200,.npz',
}

vel_models = list(map(lambda x: load(x, "arr_0"), files.values()))
vel_model_dict = dict(zip(files.keys(), vel_models))
vm_diffs = list(map(lambda x: load(x, "arr_2"), files.values()))
vm_diff_dict = dict(zip(files.keys(), vm_diffs))
objectives = list(map(lambda x: load(x, "arr_3"), files.values()))
objective_dict = dict(zip(files.keys(), objectives))
psnrs = list(map(lambda x: load(x, "arr_4"), files.values()))
psnr_dict = dict(zip(files.keys(), psnrs))
ssims = list(map(lambda x: load(x, "arr_5"), files.values()))
ssim_dict = dict(zip(files.keys(), ssims))
var_diffs = list(map(lambda x: load(x, "arr_6"), files.values()))
var_diff_dict = dict(zip(files.keys(), var_diffs))

scale = 0.5

# plt.plot(var_diffs[0], label="PDS,0")
# plt.legend()
# plt.ylabel("|| m_i - m_{i-1} ||^2_2")
# plt.xlabel("iterations")
# plt.show()


def show_velocity_model_image(data: NDArray, vmin: float, vmax: float, cmap: str = "coolwarm"):
    raw_figsize = (data.shape[1], data.shape[0])
    scale = 10.0 / max(raw_figsize[0], raw_figsize[1])
    image_aspect = (raw_figsize[0] * scale, raw_figsize[1] * scale)

    fig, ax = plt.subplots(figsize=image_aspect)
    ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    plt.gca().set_frame_on(False)

    plt.show()


def show_colorbar(data: NDArray, vmin: float, vmax: float, cmap: str = "coolwarm"):
    raw_figsize = (data.shape[1], data.shape[0])
    scale = 10.0 / max(raw_figsize[0], raw_figsize[1])
    image_aspect = (raw_figsize[0] * scale, raw_figsize[1] * scale)

    fig, ax = plt.subplots(figsize=image_aspect)
    cax = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax)
    # ax.axis('off')
    # plt.subplots_adjust(0, 0, 1, 1)
    # plt.gca().set_frame_on(False)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    plt.show()
    plt.savefig("colorbar.png", bbox_inches="tight", pad_inches=0)


dsize = 40
# show_minimum_velocity_model(load(files['gdm,0'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['gdm,1'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['gdm,5'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds,0'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds,1'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds,5'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')

# show_minimum_velocity_model(load(files['gd'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds0'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds1'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds2'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds,1400'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['pds,3400'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')

# seismic_data_path = datasets_root_path.joinpath("BP2004/vel_z6.25m_x12.5m_exact.segy")
# seismic_data = load_BP2004_model(seismic_data_path) / 1000.0
# true_velocity_model = zoom_and_crop(seismic_data[:, 1800:-1500], (152, 276))[::2]
# show_minimum_velocity_model(np.repeat(true_velocity_model, 2, axis=0), 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(np.repeat(load(files['pds,3400'], 'arr_0')[dsize:-dsize, dsize:-dsize], 2, axis=0), 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(np.repeat(load(files['gd'], 'arr_0')[dsize:-dsize, dsize:-dsize], 2, axis=0), 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(np.repeat(load(files['pds,800'], 'arr_0')[dsize:-dsize, dsize:-dsize], 2, axis=0), 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(np.repeat(load(files['pds,1400'], 'arr_0')[dsize:-dsize, dsize:-dsize], 2, axis=0), 1.5, 4.5, cmap='coolwarm')
show_minimum_velocity_model(np.repeat(load(files["pds,30000"], "arr_0")[dsize:-dsize, dsize:-dsize], 2, axis=0), 1.5, 4.5, cmap="coolwarm")
# show_minimum_velocity_model(np.repeat(load(files['pds,1'], 'arr_0')[dsize:-dsize, dsize:-dsize], 2, axis=0), 1.5, 4.5, cmap='coolwarm')

# tv = L12_norm(diff_op.D(load(files['gd'], 'arr_0')))
# print("TV: ", tv)

# fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
# alpha = list(map(lambda x: int(x.split("pds,")[1]), list(files.keys())[:-1]))
# ssims_last = list(map(lambda x: x[-1], ssims))[:-1]
# # ssims_last = list(map(lambda x: L12_norm(diff_op.D(load(x, 'arr_0'))), list(files.values())[:-1]))
# # ssims_last_noisy = list(map(lambda x: (x[-1] / 5000) ** 0.5, vm_diffs[14:-1]))
# plt.axhline(y=ssims[-1][-1], color="#460e44", linestyle="--", linewidth=2, label="Standard FWI method")
# # plt.axhline(y=(vm_diffs[-1][-1] / 5000) ** 0.5, color="#00a381", linestyle="--", linewidth=2, label="Standard FWI method(noisy)")
# plt.plot(alpha, ssims_last, label="Proposed method", color="#460e44", linewidth=3)
# # plt.plot(alpha, ssims_last_noisy, label="Proposed method(noisy)", color="#00a381", linewidth=3)
# plt.legend(fontsize=20)
# plt.ylabel("RMSE", fontsize=24)
# plt.xlabel("α", fontsize=24)
# # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
#
# # ax.set_ylim(0.30, 0.68)
#
# # yticks = [0.45, 0.50, 0.55, 0.60, 0.65]  # 任意の値
# # ax.set_yticks(yticks)
# # ax.set_yticklabels([f"{tick:.2f}" for tick in ytcks], fontsize=20)
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
#
# plt.subplots_adjust(left=0.14, right=0.995, bottom=0.15, top=0.995)
# plt.show()

# fig.patch.set_alpha(0)
# ax.set_facecolor((0, 0, 0, 0))
# ax.spines["top"].set_visile(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_linewidth(1.5)
# ax.spines["bottom"].set_linewidth(1.5)
# plt.savefig("alpha-ssim.png", bbox_inches="tight", pad_inches=0)


# fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
# alpha = list(map(lambda x: int(x.split("alpha:")[1]), list(files.keys())[:13]))
# ssims_last = list(map(lambda x: x[-1], ssims[:13]))
# ssims_last_noisy = list(map(lambda x: x[-1], ssims[14:-1]))
# plt.axhline(y=ssims[13][-1], color="#460e44", linestyle="--", linewidth=2, label="Standard FWI method")
# plt.axhline(y=ssims[-1][-1], color="#00a381", linestyle="--", linewidth=2, label="Standard FWI method(noisy)")
# plt.plot(alpha, ssims_last, label="Proposed method", color="#460e44", linewidth=3)
# plt.plot(alpha, ssims_last_noisy, label="Proposed method(noisy)", color="#00a381", linewidth=3)
# plt.legend(fontsize=20)
# plt.ylabel("SSIM", fontsize=24)
# plt.xlabel("α", fontsize=24)
# # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
#
# ax.set_ylim(0.41, 0.68)
#
# # yticks = [0.45, 0.50, 0.55, 0.60, 0.65]  # 任意の値
# # ax.set_yticks(yticks)
# # ax.set_yticklabels([f"{tick:.2f}" for tick in ytcks], fontsize=20)
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
#
# plt.subplots_adjust(left=0.14, right=0.995, bottom=0.15, top=0.995)
# plt.show()

# start_iter = 0
# end_iter = 5000
# ratio = 1
#
# fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
# # plt.plot(np.linspace(start_iter, end_iter, (end_iter - start_iter + ratio - 1) // ratio), psnr_dict[f'gd'][start_iter:end_iter:ratio], label=f"GD")
# # plt.plot(np.linspace(start_iter, end_iter, (end_iter - start_iter + ratio - 1) // ratio), ssim_dict[f'pds'][start_iter:end_iter:ratio], label=f"PDS")
#
# for key in ssim_dict.keys():
#     if key == 'gd,noisy':
#         plt.plot(np.linspace(start_iter, 396, (396 - start_iter + ratio - 1) // ratio), objective_dict['gd,noisy'][start_iter:end_iter:ratio], label='pd,noisy')
#     else:
#         plt.plot(np.linspace(start_iter, end_iter, (end_iter - start_iter + ratio - 1) // ratio), objective_dict[key][start_iter:end_iter:ratio], label=key)
#
#
# # ax.set_ylim(0.35, 0.70)
# # ax.set_xlim(0, 5000)
#
# # ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
# # ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
#
# plt.legend()
# plt.yscale('log')
# plt.ylabel("ssim")
# plt.xlabel("iterations")
# # plt.subplots_adjust(left=0.14, right=0.995, bottom=0.15, top=0.995)
# plt.show()

# fig.patch.set_alpha(0)
# ax.set_facecolor((0, 0, 0, 0))
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_linewidth(1.5)
# ax.spines["bottom"].set_linewidth(1.5)
# plt.savefig("ssim.png", bbox_inches="tight", pad_inches=0)

# fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
# # plt.plot(np.linspace(1000, 5000, 4000)[::10], objectives[13][1000::10], label="Standard FWI method", linestyle="--", color="#460e44", linewidth=2)
# plt.plot(np.linspace(1000, 5000, 4000)[::10], objectives[-1][1000::10], label="Standard FWI method(noisy)", linestyle="--", color="#00a381", linewidth=3)
# # plt.plot(np.linspace(1000, 5000, 4000)[::10], objectives[7][1000::10], label="Proposed method", color="#460e44", linewidth=2)
# plt.plot(np.linspace(1000, 5000, 4000)[::10], objectives[21][1000::10], label="Proposed method(noisy)", color="#00a381", linewidth=3)
#
# # ax.set_ylim(0.41, 0.75)
# # ax.set_xlim(0, 5000)
#
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
#
# plt.legend(fontsize=20)
# plt.ylabel("obective", fontsize=24)
# plt.xlabel("iterations", fontsize=24)
# plt.subplots_adjust(left=0.20, right=0.995, bottom=0.15, top=0.995)
# plt.show()
