import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy._typing import NDArray

from lib.dataset import load_seismic_datasets__salt_model
from lib.misc import datasets_root_path, output_path
from lib.signal_processing.misc import calc_psnr, smoothing_with_gaussian_filter, zoom_and_crop
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
    'alpha:700': '2025-02-16_12-56-47,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=700,.npz',
    'alpha:650': '2025-02-16_12-43-30,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=650,.npz',
    'alpha:600': '2025-02-16_12-29-45,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=600,.npz',
    'alpha:550': '2025-02-16_12-16-10,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=550,.npz',
    'alpha:500': '2025-02-16_12-02-37,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=500,.npz',
    'alpha:450': '2025-02-16_11-49-17,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=450,.npz',
    'alpha:400': '2025-02-16_11-35-53,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=400,.npz',
    # 'alpha:350': '2025-02-16_11-21-57,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=350,.npz',
    'alpha:350': '2025-02-22_03-22-47,pds_with_L12norm,nshots=69,gamma1=1e-05,gamma2=100,niters=20000,sigma=0,alpha=1000,.npz',
    'alpha:300': '2025-02-16_11-08-09,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=300,.npz',
    'alpha:250': '2025-02-16_10-54-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=250,.npz',
    'alpha:200': '2025-02-16_10-41-05,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=200,.npz',
    'alpha:150': '2025-02-16_10-27-45,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=150,.npz',
    'alpha:100': '2025-02-16_10-13-50,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=100,.npz',
    # 'standard FWI': '2025-02-16_10-00-02,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=0,alpha=550,.npz',
    'standard FWI': '2025-02-22_05-52-16,gradient,nshots=69,gamma1=1e-05,gamma2=None,niters=7977,sigma=0,alpha=0,.npz',

    'alpha:700_noisy': '2025-02-16_15-51-18,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=700,.npz',
    'alpha:650_noisy': '2025-02-16_15-37-52,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=650,.npz',
    'alpha:600_noisy': '2025-02-16_15-24-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=600,.npz',
    'alpha:550_noisy': '2025-02-16_15-11-09,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=550,.npz',
    'alpha:500_noisy': '2025-02-16_14-57-53,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=500,.npz',
    'alpha:450_noisy': '2025-02-16_14-44-26,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=450,.npz',
    'alpha:400_noisy': '2025-02-16_14-31-03,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=400,.npz',

    # 'alpha:350_noisy': '2025-02-16_14-17-38,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=350,.npz',
    'alpha:350_noisy':  '2025-02-17_05-46-46,pds_with_L12norm,nshots=69,gamma1=1e-05,gamma2=100,niters=20000,sigma=1,alpha=1000,.npz',
                      # '2025-02-16_23-00-25,pds_with_L12norm,nshots=69,gamma1=5e-05,gamma2=100,niters=1836,sigma=1,alpha=1000,.npz',

    'alpha:300_noisy': '2025-02-16_14-04-23,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=300,.npz',
    'alpha:250_noisy': '2025-02-16_13-51-06,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=250,.npz',
    'alpha:200_noisy': '2025-02-16_13-37-37,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=200,.npz',
    'alpha:150_noisy': '2025-02-16_13-23-52,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=150,.npz',
    'alpha:100_noisy': '2025-02-16_13-10-08,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=100,.npz',
    # 'standard FWI_noisy': '2025-02-16_09-45-18,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=1,alpha=550,.npz',
    'standard FWI_noisy': '2025-02-17_08-12-05,gradient,nshots=69,gamma1=1e-05,gamma2=None,niters=7757,sigma=1,alpha=0,.npz',
                          # '2025-02-16_23-17-59,gradient,nshots=69,gamma1=5e-05,gamma2=None,niters=936,sigma=1,alpha=0,.npz',

    # "alpha:700": "2024-10-11_18-42-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=700,.npz",
    # "alpha:650": "2024-10-11_18-28-28,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=650,.npz",
    # "alpha:600": "2024-10-11_18-14-34,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=600,.npz",
    # "alpha:550": "2024-10-11_18-00-38,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=550,.npz",
    # "alpha:500": "2024-10-11_14-44-06,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=500,.npz",
    # "alpha:450": "2024-10-11_14-30-09,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=450,.npz",
    # "alpha:400": "2024-10-11_14-16-12,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=400,.npz",
    # "alpha:350": "2024-10-11_14-02-16,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=350,.npz",
    # "alpha:300": "2024-10-11_13-48-21,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=300,.npz",
    # "alpha:250": "2024-10-11_13-34-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=250,.npz",
    # "alpha:200": "2024-10-11_15-44-41,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=200,.npz",
    # "alpha:150": "2024-10-11_15-58-37,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=150,.npz",
    # "alpha:100": "2024-10-11_16-12-35,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=0,alpha=100,.npz",
    # "standard FWI": "2024-10-11_19-32-20,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=0,alpha=None,.npz",
    # "alpha:700_noisy": "2024-11-05_02-22-21,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=700,.npz",
    # "alpha:650_noisy": "2024-11-05_02-08-49,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=650,.npz",
    # "alpha:600_noisy": "2024-11-05_01-55-17,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=600,.npz",
    # "alpha:550_noisy": "2024-11-05_01-41-45,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=550,.npz",
    # "alpha:500_noisy": "2024-11-05_01-28-14,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=500,.npz",
    # "alpha:450_noisy": "2024-11-05_01-14-44,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=450,.npz",
    # "alpha:400_noisy": "2024-11-05_01-01-14,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=400,.npz",
    # "alpha:350_noisy": "2024-11-05_00-47-46,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=350,.npz",
    # "alpha:300_noisy": "2024-11-05_00-34-18,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=300,.npz",
    # "alpha:250_noisy": "2024-11-05_00-20-50,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=250,.npz",
    # "alpha:200_noisy": "2024-11-05_00-07-25,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=200,.npz",
    # "alpha:150_noisy": "2024-11-04_23-54-02,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=150,.npz",
    # "alpha:100_noisy": "2024-11-04_23-40-41,pds_with_L12norm,nshots=20,gamma1=0.0001,gamma2=100,niters=5000,sigma=1,alpha=100,.npz",
    # "standard FWI_noisy": "2024-11-04_23-27-19,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=1,alpha=None,.npz",
}

vel_models = list(map(lambda x: load(x, "arr_0"), files.values()))
vm_diffs = list(map(lambda x: load(x, "arr_2"), files.values()))
objectives = list(map(lambda x: load(x, "arr_3"), files.values()))
psnrs = list(map(lambda x: load(x, "arr_4"), files.values()))
ssims = list(map(lambda x: load(x, "arr_5"), files.values()))

scale = 0.5


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
# show_minimum_velocity_model(raw_true_velocity_model, 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(true_velocity_model, 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(initial_velocity_model, 1.5, 4.5, cmap='coolwarm')

# show_minimum_velocity_model(load(files['alpha:150'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
show_minimum_velocity_model(load(files['alpha:350'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['alpha:550'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
show_minimum_velocity_model(load(files['standard FWI'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['alpha:150_noisy'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
show_minimum_velocity_model(load(files['alpha:350_noisy'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['alpha:550_noisy'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
show_minimum_velocity_model(load(files['standard FWI_noisy'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')
# show_minimum_velocity_model(load(files['alpha:700_noisy'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5, cmap='coolwarm')

# show_colorbar(true_velocity_model, 1.5, 4.5)

# show_velocity_model_image(load(files['alpha:300'], 'arr_0')[dsize:-dsize, dsize:-dsize], vmax=4.5, vmin=1.5, cmap='coolwarm')
# show_velocity_model_image(load(files['alpha:700'], 'arr_0')[dsize:-dsize, dsize:-dsize], vmax=4.5, vmin=1.5, cmap='coolwarm')

import math

# fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
# alpha = list(map(lambda x: int(x.split("alpha:")[1]), list(files.keys())[:13]))
# ssims_last = list(map(lambda x: (x[-1] / 5000) ** 0.5, vm_diffs[:13]))
# ssims_last_noisy = list(map(lambda x: (x[-1] / 5000) ** 0.5, vm_diffs[14:-1]))
# plt.axhline(y=(vm_diffs[13][-1] / 5000) ** 0.5, color="#460e44", linestyle="--", linewidth=2, label="Standard FWI method")
# plt.axhline(y=(vm_diffs[-1][-1] / 5000) ** 0.5, color="#00a381", linestyle="--", linewidth=2, label="Standard FWI method(noisy)")
# plt.plot(alpha, ssims_last, label="Proposed method", color="#460e44", linewidth=3)
# plt.plot(alpha, ssims_last_noisy, label="Proposed method(noisy)", color="#00a381", linewidth=3)
# plt.legend(fontsize=20)
# plt.ylabel("RMSE", fontsize=24)
# plt.xlabel("α", fontsize=24)
# # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
#
# ax.set_ylim(0.30, 0.68)
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

max_iter = 7700
ratio = 10

fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
plt.plot(np.linspace(0, max_iter, max_iter)[::ratio], ssims[13][:max_iter:ratio], label="Standard FWI method", linestyle="--", color="#460e44", linewidth=2)
plt.plot(np.linspace(0, max_iter, max_iter)[::ratio], ssims[-1][:max_iter:ratio], label="Standard FWI method(noisy)", linestyle="--", color="#00a381", linewidth=3)
plt.plot(np.linspace(0, max_iter, max_iter)[::ratio], ssims[7][:max_iter: ratio], label="Proposed method", color="#460e44", linewidth=2)
plt.plot(np.linspace(0, max_iter, max_iter)[::ratio], ssims[21][:max_iter:ratio], label="Proposed method(noisy)", color="#00a381", linewidth=3)

# ax.set_ylim(0.35, 0.70)
# ax.set_xlim(0, 5000)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)

plt.legend(fontsize=20)
plt.ylabel("SSIM", fontsize=24)
plt.xlabel("iterations", fontsize=24)
plt.subplots_adjust(left=0.14, right=0.995, bottom=0.15, top=0.995)
plt.show()

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
