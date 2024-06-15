# referred to https://www.devitoproject.org/examples/seismic/tutorials/03_fwi.html
import sys
from pathlib import Path
from typing import List, NamedTuple, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
from devito import Eq, Function, Grid, Operator, SparseTimeFunction, TimeFunction, set_log_level, solve
from matplotlib.colors import Normalize, TwoSlopeNorm
from numpy.typing import NDArray
from tqdm import tqdm

from lib.misc import output_path
from lib.model import Vec2D

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
    damping_coefficient: float

    # time
    start_time: float
    unit_time: float
    simulation_times: int

    # input source
    source_peek_time: float
    source_frequency: float

    # shorts
    shots_real_cell_pos: List[Vec2D[int]]
    receivers_real_cell_pos: List[Vec2D[int]]

    @property
    def simulation_cell_size(self) -> Vec2D[int]:
        return self.real_cell_size + Vec2D(self.damping_cell_thickness * 2, self.damping_cell_thickness * 2)

    @property
    def simulation_meter_size(self) -> Vec2D[float]:
        return self.simulation_cell_size * self.cell_meter_size

    @property
    def end_time(self) -> float:
        return self.start_time + self.unit_time * self.simulation_times

    @property
    def shots_simulation_cell_pos(self) -> List[Vec2D[int]]:
        return list(map(self.real_to_simulation_pos, self.shots_real_cell_pos))

    @property
    def shots_simulation_meter_pos(self) -> List[Vec2D[float]]:
        return list(map(self.cell_to_meter_pos, self.shots_simulation_cell_pos))

    @property
    def receivers_simulation_cell_pos(self) -> List[Vec2D[int]]:
        return list(map(self.real_to_simulation_pos, self.receivers_real_cell_pos))

    @property
    def receivers_simulation_meter_pos(self) -> List[Vec2D[float]]:
        return list(map(self.cell_to_meter_pos, self.receivers_simulation_cell_pos))

    def real_to_simulation_pos(self, pos: Vec2D[int]) -> Vec2D[int]:
        return pos + Vec2D(self.damping_cell_thickness, self.damping_cell_thickness)

    def cell_to_meter_pos(self, pos: Vec2D[int]) -> Vec2D[float]:
        return Vec2D(pos.x * self.cell_meter_size.x, pos.y * self.cell_meter_size.y)


def generate_grid(params: Params, origin: Vec2D[float] = Vec2D(0, 0)) -> Grid:
    shape = params.simulation_cell_size
    extent = params.simulation_meter_size
    return Grid(shape=(shape.x, shape.y), extent=(extent.x, extent.y), origin=(origin.x, origin.y))


def generate_input_waveform_with_ricker_wavelet(params: Params) -> WaveformData:
    time_range = np.arange(params.start_time, params.end_time + 1)
    tmp = (np.pi * params.source_frequency * (time_range - params.source_peek_time)) ** 2
    src_values = (1.0 - 2.0 * tmp) * np.exp(-tmp)
    return src_values


def generate_damping_model(params: Params) -> DampingModel:
    damping_model = np.zeros(params.simulation_cell_size)
    for i in reversed(range(params.damping_cell_thickness)):
        val = (params.damping_cell_thickness - i) * params.damping_coefficient
        damping_model[:, i] = val
        damping_model[:, -(i + 1)] = val
        damping_model[i, :] = val
        damping_model[-(i + 1), :] = val

    return damping_model


def create_simple_velocity_model(params: Params) -> VelocityModel:
    velocity_model = np.ones(params.simulation_cell_size) * 2.5
    w, h = params.simulation_cell_size.x, params.simulation_cell_size.y
    a, b = w / 2, h / 2
    y, x = np.ogrid[-a : h - a, -b : w - b]
    r = 15
    velocity_model[x * x + y * y <= r * r] = 3.0
    return velocity_model


def show_velocity_model(velocity_model: VelocityModel):
    # 適当実装
    img = velocity_model.copy()
    tmp = np.max(np.abs(img))
    min_value = -tmp * 2
    img[40, 40:141] = min_value
    img[140, 40:141] = min_value
    img[40:141, 40] = min_value
    img[40:141, 140] = min_value
    plt.imshow(img, vmin=2.45, vmax=3.05, cmap="jet", extent=(0, 1.0, 1.0, 0))
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    plt.colorbar()
    plt.title("velocity model [km/s]")
    plt.xlabel("X [km]")
    plt.ylabel("Depth [km]")
    plt.show()


def show_wave_field(wave_field: WaveField, title: str):
    # 適当実装
    img = wave_field.copy()
    tmp = np.max(np.abs(img))
    min_value = -tmp * 2
    img[40, 40:141] = min_value
    img[140, 40:141] = min_value
    img[40:141, 40] = min_value
    img[40:141, 140] = min_value
    plt.imshow(img, vmin=-tmp, vmax=tmp, cmap="seismic", extent=(-0.4, 1.4, 1.4, -0.4))
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X [km]")
    plt.ylabel("Depth [km]")
    plt.show()


def safe_field_history_as_gif(field_history: WaveFieldHistory):
    # 適当実装
    field_log = field_history.copy()
    min_value = -np.max(np.abs(field_log))
    field_log[:, 40, 40:141] = min_value
    field_log[:, 140, 40:141] = min_value
    field_log[:, 40:141, 40] = min_value
    field_log[:, 40:141, 140] = min_value

    path = output_path.joinpath("tmp-fields.gif")
    tmp = float(np.max(np.abs(field_log)))
    norm = TwoSlopeNorm(vmin=-tmp, vcenter=0, vmax=tmp)
    save_field_history_as_gif_helper(field_log, path, norm, extent=(-0.4, 1.4, 1.4, -0.4), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], figsize=(5, 5))
    sys.exit(-1)


def save_field_history_as_gif_helper(
    field_history: WaveFieldHistory,
    path: Path,
    norm: Union[Normalize | None] = None,
    field_only: bool = False,
    figsize: Union[Tuple[int, int], None] = None,
    extent: Union[Tuple[float, float, float, float], None] = None,
    xticks: Union[List[float], None] = None,
    yticks: Union[List[float], None] = None,
    title: Union[str, None] = None,
):
    n = field_history.shape[0]
    with imageio.get_writer(path, mode="I", duration=0.02) as writer:
        for i in tqdm(range(n)):
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(field_history[i], cmap="seismic", interpolation="nearest", norm=norm, extent=extent)
            if field_only:
                plt.axis("tight")
                plt.axis("off")
                fig.tight_layout()
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            else:
                plt.colorbar(im, ax=ax)
                plt.xlabel("X [km]")
                plt.ylabel("Depth [km]")
                if xticks:
                    plt.xticks(xticks)
                if yticks:
                    plt.yticks(yticks)
                if title is not None:
                    plt.title(title)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            writer.append_data(image[..., :3])
            plt.close(fig)


class WaveformSimulator:
    def __init__(self, params: Params, source_waveform: WaveformData, damping_model: DampingModel):
        self.params = params
        self.grid = generate_grid(params)
        self.single_forward_source_waveform = SparseTimeFunction(
            name="single_src", grid=self.grid, time_order=2, space_order=1, nt=params.simulation_times + 1, npoint=1, initializer=source_waveform.reshape((len(source_waveform), 1))
        )
        self.multi_backward_source_waveform = SparseTimeFunction(
            name="multi_src",
            grid=self.grid,
            time_order=2,
            space_order=1,
            nt=params.simulation_times + 1,
            npoint=len(params.receivers_real_cell_pos),
            coordinates=np.array(list(map(lambda p: [p.y, p.x], params.receivers_simulation_meter_pos))),
        )
        self.damping_model_function = Function(name="damping_model", grid=self.grid, space_order=0, initializer=damping_model)
        self.velocity_model_function = Function(name="velocity_model", grid=self.grid, time_order=0, space_order=1)
        self.forward_observed_waveform_function = TimeFunction(name="forward_observed_waveform", grid=self.grid, time_order=2, space_order=2)
        self.backward_observed_waveform_function = TimeFunction(name="backward_observed_waveform", grid=self.grid, time_order=2, space_order=2)
        self.backward_observed_waveform_dt2_function = TimeFunction(name="backward_observed_waveform_dt2", grid=self.grid, time_order=2, space_order=2)
        self.forward_simulation_operator = self._create_forward_simulation_operator(
            self.forward_observed_waveform_function, self.single_forward_source_waveform, self.damping_model_function, self.velocity_model_function
        )
        self.backward_simulation_operator = self._create_backward_simulation_operator(
            self.backward_observed_waveform_function, self.backward_observed_waveform_dt2_function, self.multi_backward_source_waveform, self.damping_model_function, self.velocity_model_function
        )

    @classmethod
    def _create_forward_simulation_operator(cls, u: TimeFunction, src: SparseTimeFunction, damping: Function, velocity_model: Function) -> Operator:
        src_term = src.inject(field=u.forward, expr=src)
        stencil = Eq(u.forward, solve(u.dt2 / velocity_model**2 - u.laplace + damping * u.dt, u.forward))
        return Operator([stencil] + src_term)

    @classmethod
    def _create_backward_simulation_operator(cls, v: TimeFunction, vdt2: TimeFunction, src: SparseTimeFunction, damping: Function, velocity_model: Function) -> Operator:
        src_term = src.inject(field=v.forward, expr=src)
        vdt2_term = [Eq(vdt2.forward, v.dt2)]
        stencil = [Eq(v.forward, solve(v.dt2 / velocity_model**2 - v.laplace + damping * v.dt, v.forward))]
        return Operator(stencil + src_term + vdt2_term)

    def simulate_forward_waveform(self, shot_simulation_meter_pos: Vec2D[float], velocity_model: Union[VelocityModel, None] = None) -> WaveFieldHistory:
        # シミュレーション用変数を破壊的に変更
        self.single_forward_source_waveform.coordinates.data[0, :] = np.array([shot_simulation_meter_pos.y, shot_simulation_meter_pos.x])
        if velocity_model is not None:
            self.velocity_model_function.data[:, :] = velocity_model

        field_logs = np.zeros((self.params.simulation_times + 1, self.params.simulation_cell_size.y, self.params.simulation_cell_size.x))
        field_logs[0] = self.forward_observed_waveform_function.data[0].copy()

        for i in range(1, self.params.simulation_times - 1):
            t_from = self.params.start_time + i * self.params.unit_time
            t_to = self.params.start_time + (i + 1) * self.params.unit_time
            self.forward_simulation_operator.apply(time_m=t_from, time_M=t_to, dt=self.params.unit_time)

            field_logs[i + 1] = self.forward_observed_waveform_function.data[0].copy()

        return field_logs

    def simulate_backward_waveform(self, residual_observed_waveform: WaveFieldHistory, velocity_model: Union[VelocityModel, None] = None) -> WaveFieldHistory:
        # シミュレーション用変数を破壊的に変更
        self.multi_backward_source_waveform.data[:, :] = residual_observed_waveform[::-1]
        if velocity_model is not None:
            self.velocity_model_function.data[:, :] = velocity_model

        field_logs = np.zeros((self.params.simulation_times + 1, self.params.simulation_cell_size.y, self.params.simulation_cell_size.x))
        field_logs[0] = self.backward_observed_waveform_dt2_function.data[0].copy()

        for i in range(1, self.params.simulation_times - 1):
            t_from = self.params.start_time + i * self.params.unit_time
            t_to = self.params.start_time + (i + 1) * self.params.unit_time
            self.backward_simulation_operator.apply(time_m=t_from, time_M=t_to, dt=self.params.unit_time)

            field_logs[i + 1] = self.backward_observed_waveform_dt2_function.data[0].copy()

        return field_logs[::-1]


class Solver:
    def __init__(self, params: Params, source_waveform: WaveformData, damping_model: DampingModel):
        self.params = params
        self.simulator = WaveformSimulator(params, source_waveform, damping_model)

    def solve(self, true_velocity_model: VelocityModel, initial_velocity_model: VelocityModel, n_iter: int) -> VelocityModel:
        current_velocity_model = initial_velocity_model.copy()
        true_observed_waveforms = self._calc_true_observed_waveforms(true_velocity_model)

        for th in range(n_iter):
            grad, residual_norm_sum = self.calc_grad_with_all_shot(current_velocity_model, true_observed_waveforms)
            alpha = 0.05 / np.max(grad)
            current_velocity_model -= grad * alpha
            print(f"iter: {th}, residual_norm_sum: {residual_norm_sum}, grad norm: {np.linalg.norm(grad, ord=2)}, error_norm: {np.linalg.norm(current_velocity_model - true_velocity_model, ord=2)}")
            show_velocity_model(current_velocity_model)

    def _calc_true_observed_waveforms(self, true_velocity_model: VelocityModel) -> List[WaveformData]:
        true_observed_waveforms = []
        for shot_pos in self.params.shots_simulation_meter_pos:
            true_field_history = self.simulator.simulate_forward_waveform(shot_pos, true_velocity_model)
            true_observed_waveform = np.array([true_field_history[:, p.y, p.x] for p in self.params.receivers_simulation_cell_pos]).T
            true_observed_waveforms.append(true_observed_waveform)
        return true_observed_waveforms

    def calc_grad_with_all_shot(self, current_velocity_model: VelocityModel, true_observed_waveforms: List[WaveformData]) -> Tuple[NDArray[np.float64], float]:
        grad = np.zeros((self.params.simulation_cell_size.y, self.params.simulation_cell_size.x))
        residual_norm_sum = 0
        for shot_pos, true_observed_waveform in zip(self.params.shots_simulation_meter_pos, true_observed_waveforms):
            grad_diff, residual_norm = self.calc_grad_with_one_shot(shot_pos, current_velocity_model, true_observed_waveform)
            grad += grad_diff
            residual_norm_sum += residual_norm

        return grad, residual_norm_sum

    def calc_grad_with_one_shot(self, shot_pos: Vec2D[int], current_velocity_model: VelocityModel, true_observed_waveform: WaveformData) -> Tuple[NDArray[np.float64], float]:
        # 推定速度モデルを用いた forward simulation
        simulated_forward_field_history = self.simulator.simulate_forward_waveform(shot_pos, current_velocity_model)
        simulated_observed_waveform = np.array([simulated_forward_field_history[:, p.y, p.x] for p in self.params.receivers_simulation_cell_pos]).T

        # 観測データと計算データの残差を計算
        diff_observed_waveform = simulated_observed_waveform - true_observed_waveform

        # 残差データを用いてbackward simulation
        simulated_backward_field_dt2_history = self.simulator.simulate_backward_waveform(diff_observed_waveform, current_velocity_model)

        # u * v.dt2の積分をとり勾配を計算（内積で積分計算をサボっている
        grad = np.sum(simulated_forward_field_history * simulated_backward_field_dt2_history, axis=0)
        residual_norm = np.linalg.norm(diff_observed_waveform, ord=2) ** 2 / 2.0
        return grad, residual_norm


def main():
    # パラメータ及びシミュレーション用変数設定
    params = Params(
        real_cell_size=Vec2D(101, 101),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        damping_coefficient=0.0005,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        shots_real_cell_pos=[Vec2D(2, i * 10 + 5) for i in range(10)],
        receivers_real_cell_pos=[Vec2D(98, i) for i in range(101)],
    )

    source_waveform = generate_input_waveform_with_ricker_wavelet(params)
    true_velocity_model = create_simple_velocity_model(params)
    initial_velocity_model = np.ones((params.simulation_cell_size.y, params.simulation_cell_size.x)) * 2.5
    damping_model = generate_damping_model(params)

    solver = Solver(
        params=params,
        source_waveform=source_waveform,
        damping_model=damping_model,
    )
    estimated_velocity_model = solver.solve(true_velocity_model, initial_velocity_model, 10)


if __name__ == "__main__":
    main()
