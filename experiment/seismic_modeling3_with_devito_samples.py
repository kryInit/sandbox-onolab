# referred to https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html

from typing import List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from devito import Eq, Function, Inc, Max, Min, Operator, TimeFunction, mmax, norm, set_log_level, solve
from numpy.typing import NDArray

from lib.model import Vec2D
from lib.seismic import AcquisitionGeometry, Receiver, SeismicModel, demo_model, plot_velocity
from lib.seismic.acoustic import AcousticWaveSolver

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


def main():
    params = Params(
        real_cell_size=Vec2D(101, 101),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        n_shots=9,
        n_receivers=101,
    )

    shape = (params.real_cell_size.x, params.real_cell_size.y)
    spacing = (params.cell_meter_size.x, params.cell_meter_size.y)

    true_model = demo_model("circle-isotropic", vp_circle=3.0, vp_background=2.5, shape=shape, spacing=spacing, nbl=params.damping_cell_thickness)
    current_model = demo_model("circle-isotropic", vp_circle=2.5, vp_background=2.5, shape=shape, spacing=spacing, nbl=params.damping_cell_thickness, grid=true_model.grid)

    src_coordinates = np.array([[20, true_model.domain_size[1] * 0.5]])
    rec_coordinates = np.array([[980, y] for y in np.linspace(0, true_model.domain_size[0], num=params.n_receivers)])
    source_locations = np.array([[30, y] for y in np.linspace(0.0, (params.real_cell_size.y - 1) * params.cell_meter_size.y, num=params.n_shots)])

    geometry = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates, params.start_time, params.simulation_times, f0=params.source_frequency, src_type="Ricker")
    solver = AcousticWaveSolver(true_model, geometry, space_order=4)

    ssolver = VelocityModelGradientCalculator(true_model, geometry, solver)

    n_iters = 1000

    l1_norm_weight = 5
    gamma1 = 0.001
    gamma2 = 50

    def prox_l1(signal, gamma):
        return np.sign(signal) * np.maximum(np.abs(signal) - gamma, 0)

    D = lambda z: np.stack([np.roll(z, -1, axis=0) - z, np.roll(z, -1, axis=1) - z], axis=2)
    Dt = lambda z: np.roll(z[:, :, 0], 1, axis=0) - z[:, :, 0] + np.roll(z[:, :, 1], 1, axis=1) - z[:, :, 1]

    history = np.zeros((n_iters, 1))
    history1 = np.zeros((n_iters, 1))

    y = D(current_model.vp.data)

    for i in range(0, n_iters):
        residual_norm_sum, direction = ssolver.calc_grad(params, current_model.vp, source_locations)

        grad = -direction.data

        # update velocity model with box constraint
        # step_size = 0.05 / mmax(direction)
        # current_model.vp.data[:] = np.clip(current_model.vp.data + step_size * direction.data, 2.0, 3.5)

        # current_model.vp.data[:] = np.clip(current_model.vp.data + 0.001 * direction.data, 2.0, 3.5)

        prev_x = current_model.vp.data.copy()
        current_model.vp.data[:] = current_model.vp.data[:] - gamma1 * (grad + Dt(y))
        y = y + gamma2 * D(2 * current_model.vp.data - prev_x)
        y = y - gamma2 * prox_l1(y / gamma2, l1_norm_weight / gamma2)

        history[i] = residual_norm_sum
        diff = current_model.vp.data - true_model.vp.data
        history1[i] = np.sum(diff * diff)
        print(f"Objective value is {residual_norm_sum} at iteration {i+1}, {history1[i]}")
        # if i % 100 == 0 or (i != 0 and history[i-1] < history[i]):
        if i % 100 == 0:
            print(f"showed: {i}")
            plot_velocity(current_model)

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
