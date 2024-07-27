import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lib.misc import datasets_root_path, output_path

seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")

# Dimensions
nx, ny, nz = 676, 676, 210

with open(seismic_data_path, "r") as file:
    vel = np.fromfile(file, dtype=np.dtype("float32").newbyteorder(">"))
    vel = vel.reshape(nx, ny, nz, order="F")

    # Cast type
    vel = np.asarray(vel, dtype=float)

    # THE SEG/EAGE salt-model uses positive z downwards;
    # here we want positive upwards. Hence:
    vel = np.flip(vel, 2)

    seismic_data = np.transpose(vel, (2, 1, 0))
    print(seismic_data.shape)

    img_path = output_path.joinpath("salt.gif")
    with imageio.get_writer(img_path, mode="I", duration=0.02) as writer:
        for i in tqdm(range(seismic_data.shape[1])):
            fig, ax = plt.subplots()
            cax = ax.imshow(seismic_data[:, i], cmap="seismic")
            plt.axis("off")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            plt.close(fig)
