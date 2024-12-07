import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy._typing import NDArray

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
    # 'standard FWI': '2024-10-11_19-32-20,gradient,nshots=20,gamma1=0.0001,gamma2=None,niters=5000,sigma=0,alpha=None,.npz',
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
# show_velocity_model(raw_true_velocity_model, 1.5, 4.5)
# show_velocity_model(true_velocity_model, 1.5, 4.5)
# show_velocity_model(initial_velocity_model, 1.5, 4.5)

# show_velocity_model(load(files['alpha:150'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5)
# show_velocity_model(load(files['alpha:350'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5)
# show_velocity_model(load(files['alpha:550'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5)
# show_velocity_model(load(files['standard FWI'], 'arr_0')[dsize:-dsize, dsize:-dsize], 1.5, 4.5)

# show_colorbar(true_velocity_model, 1.5, 4.5)


fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
alpha = list(map(lambda x: int(x.split("alpha:")[1]), list(files.keys())[:-1]))
ssims_last = list(map(lambda x: x[-1], ssims[:-1]))
plt.axhline(y=ssims[-1][-1], color="black", linestyle="--", linewidth=3, label="Standard FWI method")
plt.plot(alpha, ssims_last, label="Proposed method", color="#00a381", linewidth=3)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)


yticks = [0.50, 0.55, 0.60, 0.65]  # 任意の値
ax.set_yticks(yticks)
ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=12)

fig.patch.set_alpha(0)
ax.set_facecolor((0, 0, 0, 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
plt.savefig("alpha-ssim.png", bbox_inches="tight", pad_inches=0)


fig, ax = plt.subplots(figsize=(16 * scale, 10 * scale))
plt.plot(ssims[-1], label="Standard FWI method", color="#7a4171", linewidth=3)
plt.plot(ssims[7], label="Proposed method", color="#00a381", linewidth=3)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
fig.patch.set_alpha(0)
ax.set_facecolor((0, 0, 0, 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
plt.savefig("ssim.png", bbox_inches="tight", pad_inches=0)
