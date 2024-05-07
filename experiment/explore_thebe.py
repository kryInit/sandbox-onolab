import numpy as np
from lib.misc import project_root_path, datasets_root_path
import matplotlib.pyplot as plt
import pyvista as pv


def show_images():
    # n = 128
    seismic_data_path = datasets_root_path.joinpath('thebe/seistrain1.npz')
    fault_data_path = datasets_root_path.joinpath('thebe/faulttrain1.npy')
    seismic_data = np.load(seismic_data_path)['arr_0']
    fault_data = np.load(fault_data_path)
    print(seismic_data.shape, fault_data.shape)

    def save_images():
        for i in range(128):
            plt.imsave(project_root_path.joinpath(f'tmp/0-{i}.jpg'), seismic_data[i], cmap='gray')

    def show_head():
        i = 50
        fig, axes = plt.subplots(1, 2, figsize=(6, 5))

        print(np.min(seismic_data[i]), np.max(seismic_data[i]))

        axes[0].imshow(np.abs(seismic_data[i]), cmap='gray')
        axes[1].imshow(fault_data[i], cmap='gray')

        plt.tight_layout()
        plt.show()

    def show_by_plt():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.zeros((n, n))
        y, z = np.meshgrid(range(n), range(n))
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.gray(seismic_data[:, :, 0]), cmap='gray',
                        shade=False)

        y = np.zeros((n, n)) + n - 1
        x, z = np.meshgrid(range(n), range(n))
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.gray(seismic_data[:, n - 1, :]), cmap='gray',
                        shade=False)

        z = np.zeros((n, n))
        x, y = np.meshgrid(range(n), range(n))
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.gray(seismic_data[0, :, :]), cmap='gray',
                        shade=False, zorder=-1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def show_by_pyvista():
        coords = np.linspace(-n // 2, n // 2, n, endpoint=False)

        grid = pv.StructuredGrid()
        z, x, y = np.meshgrid(coords, coords, coords, indexing='ij')
        grid.points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        grid.dimensions = [n, n, n]
        grid.point_data['values'] = seismic_data.flatten(order='F')

        slice_x = grid.slice('x', origin=(0, 0, 0))
        slice_y = grid.slice('y', origin=(0, 0, 0))
        slice_z = grid.slice('z', origin=(0, 0, 0))

        p = pv.Plotter()
        p.add_mesh(slice_x, cmap='gray', clim=[seismic_data.min(), seismic_data.max()])
        p.add_mesh(slice_y, cmap='gray', clim=[seismic_data.min(), seismic_data.max()])
        p.add_mesh(slice_z, cmap='gray', clim=[seismic_data.min(), seismic_data.max()])
        p.show()

    # save_images()
    show_head()
    # show_by_plt()
    # show_by_pyvista()


if __name__ == '__main__':
    show_images()

