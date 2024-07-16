# referred to https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html

from typing import List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from devito import Eq, Function, Inc, Operator, TimeFunction, norm, set_log_level, solve
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, zoom

from lib.misc import datasets_root_path
from lib.model import Vec2D
from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver

# devitoのlogの抑制
set_log_level("WARNING")

WaveformData = NDArray[np.float64]
VelocityModel = NDArray[np.float64]
DampingModel = NDArray[np.float64]

WaveField = NDArray[np.float64]
WaveFieldHistory = NDArray[np.float64]


class Params(NamedTuple):
    real_cell_size: Vec2D[int]
    cell_meter_size: Vec2D[float]

    # damping
    damping_cell_thickness: int

    # time
    start_time: float
    unit_time: float
    simulation_times: int

    # input source
    source_peek_time: float
    source_frequency: float

    # shorts
    n_shots: int
    n_receivers: int


class VelocityModelGradientCalculator:
    def __init__(self, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver):
        self.true_model = true_model
        self.geometry = geometry
        self.solver = solver
        self.grad_operator = self._create_grad_op(true_model, geometry, solver)
        pass

    @classmethod
    def _create_grad_op(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver):
        grad = Function(name="grad", grid=true_model.grid)
        u = TimeFunction(name="u", grid=true_model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
        v = TimeFunction(name="v", grid=true_model.grid, save=None, time_order=2, space_order=solver.space_order)

        eqns = [Eq(v.backward, solve(true_model.m * v.dt2 - v.laplace + true_model.damp * v.dt.T, v.backward))]
        rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * true_model.grid.stepping_dim.spacing**2 / true_model.m)
        gradient_update = Inc(grad, -u.dt2 * v * true_model.m**1.5)

        return Operator(eqns + rec_term + [gradient_update], subs=true_model.spacing_map, name="Gradient")

    def _calc_true_observed_waveforms(self, params: Params, source_locations: NDArray[np.float64]) -> List[WaveformData]:
        observed_waveform = Receiver(name="d_obs", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        true_observed_waveforms = []
        for i in range(params.n_shots):
            self.geometry.src_positions[0, :] = source_locations[i, :]

            # 真のモデル: mode.vp を用いて、観測データ波形(observed_waveform)を計算
            self.solver.forward(vp=self.true_model.vp, rec=observed_waveform)

            true_observed_waveforms.append(observed_waveform.data[:].copy())
        return true_observed_waveforms

    def calc_grad(self, params: Params, current_velocity_model: Function, source_locations: NDArray[np.float64]) -> Tuple[float, Function]:
        grad = Function(name="grad", grid=self.true_model.grid)
        residual = Receiver(name="residual", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        calculated_waveform = Receiver(name="d_syn", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        objective = 0.0

        observed_waveforms = self._calc_true_observed_waveforms(params, source_locations)

        for i in range(params.n_shots):
            self.geometry.src_positions[0, :] = source_locations[i, :]

            # 現在のモデル: vp_in を用いて、計算データ波形(calculated_waveform)を計算
            _, calculated_wave_field, _ = self.solver.forward(vp=current_velocity_model, save=True, rec=calculated_waveform)

            # 観測データと計算データの残差を計算
            residual.data[:] = calculated_waveform.data[:] - observed_waveforms[i]

            objective += 0.5 * norm(residual) ** 2

            self.grad_operator.apply(rec=residual, grad=grad, u=calculated_wave_field, dt=self.solver.dt, vp=current_velocity_model)

        return objective, grad


def plot_velocity_model(data: NDArray, vmin: Union[float, None] = None, vmax: Union[float, None] = None, title: str = "velocity model", cmap: str = "jet"):
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X [km]")
    plt.ylabel("Depth [km]")
    plt.show()


def psnr(signal0, signal1, max_value):
    mse = np.mean((signal0.astype(float) - signal1.astype(float)) ** 2)
    return 10 * np.log10((max_value**2) / mse)


def main():
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

    assert seismic_data.shape == (nz, ny, nx)

    params = Params(
        real_cell_size=Vec2D(101, 51),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        n_shots=26,
        n_receivers=101,
    )

    shape = (params.real_cell_size.x, params.real_cell_size.y)
    spacing = (params.cell_meter_size.x, params.cell_meter_size.y)

    dsize = params.damping_cell_thickness
    th = 400

    raw_target = seismic_data[:, th].T / 1000
    target = zoom(raw_target, 51.0 / nz, order=3)[:101, :51]

    initial_vp = target.copy()
    for _ in range(10):
        initial_vp = gaussian_filter(initial_vp, sigma=1)

    initial_vp = np.zeros(target.shape) * 2.5

    true_model = SeismicModel(space_order=2, vp=target, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=params.damping_cell_thickness, bcs="damp", fs=False)
    current_model = SeismicModel(space_order=2, vp=initial_vp, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=params.damping_cell_thickness, bcs="damp", fs=False)

    print(f"initial psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize].T, current_model.vp.data[dsize:-dsize, dsize:-dsize].T, 5)}")

    # plot_velocity_model(seismic_data[:, th] / 1000, vmin=1.5, vmax=5)
    # plot_velocity(true_model)
    plot_velocity_model(true_model.vp.data[dsize:-dsize, dsize:-dsize].T, vmin=1.5, vmax=5)
    # plot_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize].T, vmin=1.5, vmax=5)

    # src_coordinates = np.array([[30, true_model.domain_size[1] * 0.5]])
    # rec_coordinates = np.array([[30, x] for x in np.linspace(0, true_model.domain_size[1], num=params.n_receivers)])
    # source_locations = np.array([[980, x] for x in np.linspace(0.0, true_model.domain_size[1], num=params.n_shots)])

    src_coordinates = np.array([[30, true_model.domain_size[1] * 0.5]])
    rec_coordinates = np.array([[x, 30] for x in np.linspace(0, true_model.domain_size[0], num=params.n_receivers)])
    source_locations = np.array([[x, 30] for x in np.linspace(0.0, true_model.domain_size[0], num=params.n_shots)])

    geometry = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates, params.start_time, params.simulation_times, f0=params.source_frequency, src_type="Ricker")
    solver = AcousticWaveSolver(true_model, geometry, space_order=4)

    ssolver = VelocityModelGradientCalculator(true_model, geometry, solver)

    n_iters = 1000

    def prox_l1(signal, gamma):
        return np.sign(signal) * np.maximum(np.abs(signal) - gamma, 0)

    D = lambda z: np.stack([np.roll(z, -1, axis=0) - z, np.roll(z, -1, axis=1) - z], axis=2)
    Dt = lambda z: np.roll(z[:, :, 0], 1, axis=0) - z[:, :, 0] + np.roll(z[:, :, 1], 1, axis=1) - z[:, :, 1]

    history = np.zeros((n_iters, 1))
    history1 = np.zeros((n_iters, 1))

    # flag = 'normal'
    # flag = 'admm'
    flag = "pds"

    if flag == "normal":
        step_size = 0.0026
        for i in range(0, n_iters):
            residual_norm_sum, direction = ssolver.calc_grad(params, current_model.vp, source_locations)

            grad = -direction.data / params.n_shots

            # update velocity model with box constraint
            current_model.vp.data[:] = current_model.vp.data - step_size * grad

            history[i] = residual_norm_sum
            diff = current_model.vp.data - true_model.vp.data
            history1[i] = np.sum(diff * diff)
            print(f"Objective value is {residual_norm_sum} at iteration {i + 1}, {history1[i]}")
            # if i % 100 == 0 or (i != 0 and history[i-1] < history[i]):
            # if i % 100 == 0:
            print(f"showed: {i}")
            plot_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize].T, vmin=1.5, vmax=5)
            print(f"psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize].T, current_model.vp.data[dsize:-dsize, dsize:-dsize].T, 5)}")

    if flag == "admm":
        l1_norm_weight = 1
        rho = 0.001
        gamma = 0.001
        tau = 0.001
        eta = current_model.vp.data
        beta = np.zeros(eta.shape)
        for i in range(0, n_iters):
            residual_norm_sum, direction = ssolver.calc_grad(params, current_model.vp, source_locations)

            grad = -direction.data / params.n_shots

            # update velocity model with box constraint
            # step_size = 0.05 / mmax(direction)
            # current_model.vp.data[:] = np.clip(current_model.vp.data + step_size * direction.data, 2.0, 3.5)

            # current_model.vp.data[:] = np.clip(current_model.vp.data + 0.001 * direction.data, 2.0, 3.5)

            v = current_model.vp.data
            v[:] = v - rho * (grad - gamma * (eta - v - beta))
            eta = Dt(prox_l1(D(v + beta), l1_norm_weight / gamma))
            beta = beta + tau * (v - eta)

            history[i] = residual_norm_sum
            diff = current_model.vp.data - true_model.vp.data
            history1[i] = np.sum(diff * diff)
            print(f"Objective value is {residual_norm_sum} at iteration {i + 1}, {history1[i]}")
            # if i % 100 == 0 or (i != 0 and history[i-1] < history[i]):
            # if i % 100 == 0:
            print(f"showed: {i}")
            plot_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize].T, vmin=1.5, vmax=5)
            print(f"psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize].T, current_model.vp.data[dsize:-dsize, dsize:-dsize].T, 5)}")

    if flag == "pds":
        l1_norm_weight = 1
        gamma1 = 0.0001
        gamma2 = 100
        y = D(current_model.vp.data)
        for i in range(0, n_iters):
            residual_norm_sum, direction = ssolver.calc_grad(params, current_model.vp, source_locations)

            grad = -direction.data

            # update velocity model with box constraint
            # step_size = 0.05 / mmax(direction)
            # current_model.vp.data[:] = np.clip(current_model.vp.data + step_size * direction.data, 2.0, 3.5)

            # current_model.vp.data[:] = np.clip(current_model.vp.data + 0.001 * direction.data, 2.0, 3.5)

            prev_x = current_model.vp.data.copy()
            x = current_model.vp.data
            x[:] = x - gamma1 * (grad + Dt(y))
            y = y + gamma2 * D(2 * x - prev_x)
            y = y - gamma2 * prox_l1(y / gamma2, l1_norm_weight / gamma2)

            history[i] = residual_norm_sum
            diff = current_model.vp.data - true_model.vp.data
            history1[i] = np.sum(diff * diff)
            print(f"Objective value is {residual_norm_sum} at iteration {i + 1}, {history1[i]}")
            # if i % 100 == 0 or (i != 0 and history[i-1] < history[i]):
            # if i % 100 == 0:
            print(f"showed: {i}")
            plot_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize].T, vmin=1.5, vmax=5)
            print(f"psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize].T, current_model.vp.data[dsize:-dsize, dsize:-dsize].T, 5)}")

    # Plot inverted velocity model
    # plot_velocity(current_model)

    # Plot objective function decrease
    plt.figure()
    plt.loglog(history)
    plt.xlabel("Iteration number")
    plt.ylabel("Misift value observed signal")
    plt.title("Convergence")
    plt.show()

    plt.figure()
    plt.plot(history1)
    plt.xlabel("Iteration number")
    plt.ylabel("Misift value velocity model")
    plt.title("Convergence")
    plt.show()


if __name__ == "__main__":
    main()
