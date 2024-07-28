from typing import List, Tuple

import joblib

import numpy as np
import numpy.typing as npt
from devito import set_log_level
from devito import Eq, Function, Inc, Operator, TimeFunction, norm, solve
from numpy.typing import NDArray

from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver


class FastParallelVelocityModelGradientCalculator:
    def __init__(
        self,
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
        self.source_locations = source_locations

        self.true_model = SeismicModel(space_order=2, vp=true_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=damping_cell_thickness, bcs="damp", fs=False)
        self.current_model = SeismicModel(space_order=2, vp=initial_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=damping_cell_thickness, bcs="damp", fs=False)

        self.geometry = AcquisitionGeometry(self.true_model, receiver_locations, np.array([[0, 0]]), start_time, end_time, f0=source_frequency, src_type="Ricker")
        self.simulator = AcousticWaveSolver(self.true_model, self.geometry, space_order=4)

        self.grad_operator = self._create_grad_op(self.true_model, self.geometry, self.simulator)

        self.observed_waveforms = self._calc_true_observed_waveforms(self.true_model, self.geometry, self.simulator, source_locations)

    @classmethod
    def _create_grad_op(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver) -> Operator:
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

    def calc_grad(self, current_velocity_model: npt.NDArray, n_jobs: int = -1) -> Tuple[float, NDArray[np.float64]]:

        objective = 0.0
        grad_value = np.zeros(self.current_model.vp.data.shape)
        n_shots = self.source_locations.shape[0]

        result = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(calc_grad_helper)(self.geometry, self.simulator, self.source_locations[i], self.current_model, current_velocity_model, self.observed_waveforms[i], self.grad_operator) for i in range(n_shots))

        for i in range(n_shots):
            objective += result[i][0]
            print(grad_value.shape, result[i][1].shape)
            grad_value += result[i][1]

        return objective, grad_value

def calc_grad_helper(
        geometry: AcquisitionGeometry,
        simulator: AcousticWaveSolver,
        source_location: npt.NDArray,
        current_model: SeismicModel,
        current_velocity_model: npt.NDArray,
        observed_waveform: npt.NDArray,
        grad_operator: Operator
) -> Tuple[float, npt.NDArray]:
    set_log_level("WARNING")
    geometry.src_positions[0][:] = source_location

    print(current_velocity_model.T.shape, current_model.shape, current_model.nbl, current_model.spacing)
    new_current_model = SeismicModel(space_order=2, vp=current_velocity_model, origin=(0, 0), shape=current_model.shape, dtype=np.float32, spacing=current_model.spacing, nbl=current_model.nbl, bcs="damp", fs=False)
    residual = Receiver(name="residual", grid=current_model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    current_model = new_current_model

    # 現在のモデル: vp_in を用いて、計算データ波形(calculated_waveform)を計算
    calculated_waveform = Receiver(name="d_syn", grid=current_model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    print(current_model.vp.shape)
    _, calculated_wave_field, _ = simulator.forward(vp=current_model.vp, save=True, rec=calculated_waveform)

    # 観測データと計算データの残差を計算
    residual.data[:] = calculated_waveform.data - observed_waveform

    # 雑なobjective計算
    objective = 0.5 * np.sum(np.abs(residual.data ** 2))

    # ちゃんとしたobjective計算
    # objective += 0.5 * norm(residual) ** 2

    # 勾配を計算
    grad = Function(name="grad", grid=current_model.grid)
    grad_operator.apply(rec=residual, grad=grad, u=calculated_wave_field, dt=simulator.dt, vp=current_model.vp)

    return objective, grad.data


class FastParallelVelocityModelGradientCalculatorHelper:
    def __init__(
        self,
        true_velocity_model: npt.NDArray,
        initial_velocity_model: npt.NDArray,
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
        damping_cell_thickness: int,
        start_time: float,
        end_time: float,
        source_frequency: float,
        source_location: npt.NDArray,
        receiver_locations: npt.NDArray,
    ):
        self.true_model = SeismicModel(space_order=2, vp=true_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=damping_cell_thickness, bcs="damp", fs=False)
        self.current_model = SeismicModel(space_order=2, vp=initial_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=damping_cell_thickness, bcs="damp", fs=False)

        self.geometry = AcquisitionGeometry(self.true_model, receiver_locations, np.array([[0, 0]]), start_time, end_time, f0=source_frequency, src_type="Ricker")
        self.geometry.src_positions[0][:] = source_location

        self.simulator = AcousticWaveSolver(self.true_model, self.geometry, space_order=4)

        self.grad_operator = self._create_grad_op(self.true_model, self.geometry, self.simulator)

        self.observed_waveform = self._calc_true_observed_waveforms(self.true_model, self.geometry, self.simulator)


    @classmethod
    def _create_grad_op(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver) -> Operator:
        grad = Function(name="grad", grid=true_model.grid)
        u = TimeFunction(name="u", grid=true_model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
        v = TimeFunction(name="v", grid=true_model.grid, save=None, time_order=2, space_order=solver.space_order)

        eqns = [Eq(v.backward, solve(true_model.m * v.dt2 - v.laplace + true_model.damp * v.dt.T, v.backward))]
        rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * true_model.grid.stepping_dim.spacing**2 / true_model.m)
        gradient_update = Inc(grad, u.dt2 * v * true_model.m**1.5)

        return Operator(eqns + rec_term + [gradient_update], subs=true_model.spacing_map, name="Gradient")

    @classmethod
    def _calc_true_observed_waveforms(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, simulator: AcousticWaveSolver) -> List[NDArray]:
        observed_waveform = Receiver(name="d_obs", grid=true_model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
        simulator.forward(vp=true_model.vp, rec=observed_waveform)
        return observed_waveform.data.copy()

    def calc_grad(self, current_velocity_model: npt.NDArray, n_jobs: int = -1) -> Tuple[float, NDArray[np.float64]]:
        grad = Function(name="grad", grid=self.true_model.grid)
        residual = Receiver(name="residual", grid=self.current_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        calculated_waveform = Receiver(name="d_syn", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)

        # 現在のモデル: vp_in を用いて、計算データ波形(calculated_waveform)を計算
        _, calculated_wave_field, _ = self.simulator.forward(vp=current_velocity_model, save=True, rec=calculated_waveform)

        # 観測データと計算データの残差を計算
        residual.data[:] = calculated_waveform.data - self.observed_waveform

        # 雑なobjective計算
        objective = 0.5 * np.sum(np.abs(residual.data ** 2))

        # ちゃんとしたobjective計算
        # objective = 0.5 * norm(residual) ** 2

        self.grad_operator.apply(rec=residual, grad=grad, u=calculated_wave_field, dt=self.simulator.dt, vp=current_velocity_model)

        return objective, -grad.data

