from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from devito import Eq, Function, Inc, Operator, TimeFunction, norm, solve
from numpy.typing import NDArray

from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver


class FastParallelVelocityModelGradientCalculator:
    def __init__(
        self,
        n_jobs: int,
        true_velocity_model: npt.NDArray,
        initial_velocity_model: npt.NDArray,
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
        damping_cell_thickness: int,
        start_time: float,
        end_time: float,
        source_frequency: float,
        source_locations: npt.NDArray,
        receiver_locations: npt.NDArray,
    ):
        self.n_jobs = n_jobs
        self.source_locations = source_locations

        self.true_model = SeismicModel(space_order=2, vp=true_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=damping_cell_thickness, bcs="damp", fs=False)
        self.current_model = SeismicModel(space_order=2, vp=initial_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=damping_cell_thickness, bcs="damp", fs=False)

        self.geometry = AcquisitionGeometry(self.true_model, receiver_locations, np.array([[0, 0]]), start_time, end_time, f0=source_frequency, src_type="Ricker")
        self.simulator = AcousticWaveSolver(self.true_model, self.geometry, space_order=4)

        self.grad_operator = self._create_grad_op(self.true_model, self.geometry, self.simulator)

        self.observed_waveforms = self._calc_true_observed_waveforms(self.true_model, self.geometry, self.simulator, source_locations)

    @classmethod
    def _create_grad_op(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver):
        grad = Function(name="grad", grid=true_model.grid)
        u = TimeFunction(name="u", grid=true_model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
        v = TimeFunction(name="v", grid=true_model.grid, save=None, time_order=2, space_order=solver.space_order)

        eqns = [Eq(v.backward, solve(true_model.m * v.dt2 - v.laplace + true_model.damp * v.dt.T, v.backward))]
        rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * true_model.grid.stepping_dim.spacing**2 / true_model.m)
        gradient_update = Inc(grad, u.dt2 * v * true_model.m**1.5)

        return Operator(eqns + rec_term + [gradient_update], subs=true_model.spacing_map, name="Gradient")

    @classmethod
    def _calc_true_observed_waveforms(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, simulator: AcousticWaveSolver, source_locations: NDArray) -> List[NDArray]:
        observed_waveform = Receiver(name="d_obs", grid=true_model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
        true_observed_waveforms = []
        n_shots = source_locations.shape[0]
        for i in range(n_shots):
            geometry.src_positions[0][:] = source_locations[i]

            # 真のモデル: mode.vp を用いて、観測データ波形(observed_waveform)を計算
            simulator.forward(vp=true_model.vp, rec=observed_waveform)

            true_observed_waveforms.append(observed_waveform.data.copy())
        return true_observed_waveforms

    def calc_grad(self) -> Tuple[float, NDArray[np.float64]]:
        objective = 0.0
        grad_value = np.zeros(self.current_model.vp.data.shape)
        n_shots = self.source_locations.shape[0]
        for i in range(n_shots):
            self.geometry.src_positions[0][:] = self.source_locations[i]

            # 現在のモデル: vp_in を用いて、計算データ波形(calculated_waveform)を計算
            calculated_waveform = Receiver(name="d_syn", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
            _, calculated_wave_field, _ = self.simulator.forward(vp=self.current_model.vp, save=True, rec=calculated_waveform)

            # 観測データと計算データの残差を計算
            residual = Receiver(name="residual", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
            residual.data[:] = calculated_waveform.data - self.observed_waveforms[i]
            objective += 0.5 * norm(residual) ** 2

            # 勾配を計算
            grad = Function(name="grad", grid=self.true_model.grid)
            self.grad_operator.apply(rec=residual, grad=grad, u=calculated_wave_field, dt=self.simulator.dt, vp=self.current_model.vp)
            grad_value += grad.data

        return objective, grad_value
