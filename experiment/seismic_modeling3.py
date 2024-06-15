# referred to https://www.devitoproject.org/examples/seismic/tutorials/03_fwi.html
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple, Tuple, Union

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

WaveField = NDArray[np.float64]
WaveFieldHistory = WaveField


class Params(NamedTuple):
    field_cell_size: Vec2D[int]
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
    source_cell_pos_in_field: Vec2D[int]

    # shorts
    shots_pos_in_field: List[Vec2D[int]]
    # receivers_pos_in_field: List[Vec2D[int]]


def generate_grid(
    field_cell_size: Vec2D[int],
    damping_cell_thickness: int,
    cell_meter_size: Vec2D[float],
    origin: Vec2D[float] = Vec2D(0, 0),
) -> Grid:
    # damping層込みのセルサイズ
    field_cell_size_with_damping = field_cell_size + Vec2D(damping_cell_thickness * 2, damping_cell_thickness * 2)
    field_meter_size_with_damping = field_cell_size_with_damping * cell_meter_size

    shape = field_cell_size_with_damping + Vec2D(1, 1)  # セル数 + 1
    extent = field_meter_size_with_damping  # フィールドサイズ[m]

    return Grid(shape=(shape.x, shape.y), extent=(extent.x, extent.y), origin=(origin.x, origin.y))


def generate_source_devito_function(name: str, grid: Grid, src_coordinate: Vec2D[int], src_values: NDArray[np.float64]) -> SparseTimeFunction:
    src = SparseTimeFunction(name=name, grid=grid, time_order=2, space_order=1, nt=len(src_values) + 1, npoint=1)
    src.coordinates.data[0, 0] = src_coordinate.y
    src.coordinates.data[0, 1] = src_coordinate.x
    src.data[:-1, 0] = src_values
    return src


def generate_input_waveform_with_ricker_wavelet(start_time: float, unit_time: float, simulation_times: int, frequency: float, source_peek_time: float) -> WaveformData:
    time_range = np.linspace(start_time, start_time + unit_time * simulation_times, simulation_times * unit_time + 1)
    src_values = (1.0 - 2.0 * (np.pi * frequency * (time_range - source_peek_time)) ** 2) * np.exp(-((np.pi * frequency * (time_range - source_peek_time)) ** 2))
    return src_values


def create_input_source(
    grid: Grid,
    start_time: float,
    unit_time: float,
    simulation_times: int,
    frequency: float,
    source_peek_time: float,
    source_cell_pos_in_field: Vec2D[int],
    damping_cell_thickness: int,
    cell_meter_size: Vec2D[float],
) -> SparseTimeFunction:
    # use ricker wavelet
    time_range = np.linspace(start_time, start_time + unit_time * simulation_times, simulation_times * unit_time + 1)
    src_values = (1.0 - 2.0 * (np.pi * frequency * (time_range - source_peek_time)) ** 2) * np.exp(-((np.pi * frequency * (time_range - source_peek_time)) ** 2))
    src_coordinate = (source_cell_pos_in_field + Vec2D(damping_cell_thickness, damping_cell_thickness)) * cell_meter_size

    return generate_source_devito_function("src", grid, src_coordinate, src_values)


def generate_damping_model_devito_function(grid: Grid, damping_cell_thickness: int, damping_coefficient: float) -> Function:
    damping_model = Function(name="eta", grid=grid, space_order=0)

    for i in reversed(range(damping_cell_thickness)):
        val = (damping_cell_thickness - i) * damping_coefficient
        damping_model.data[:, i] = val
        damping_model.data[:, -(i + 1)] = val
        damping_model.data[i, :] = val
        damping_model.data[-(i + 1), :] = val

    return damping_model


def create_simple_velocity_model_devito_function(grid: Grid, field_cell_size_y: int, damping_cell_thickness: int) -> Function:
    v = Function(name="v", grid=grid, time_order=0, space_order=1)

    velocity_model = 2.5 * np.ones(grid.shape)
    # velocity_model[damping_cell_thickness + field_cell_size_y // 2: -damping_cell_thickness, :] = 3

    a, b = grid.shape[0] / 2, grid.shape[1] / 2
    y, x = np.ogrid[-a : grid.shape[0] - a, -b : grid.shape[1] - b]
    r = 15
    velocity_model[x * x + y * y <= r * r] = 3.0

    v.data[:, :] = velocity_model

    return v


def simulate_forward_waveform(src: SparseTimeFunction, damping: Function, velocity_model: Function, grid: Grid, start_time: float, unit_time: float, simulation_times: int) -> WaveFieldHistory:
    observed_waveform = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

    u = observed_waveform  # alias(長いので
    src_term = src.inject(field=u.forward, expr=src)
    stencil = Eq(u.forward, solve(u.dt2 / velocity_model**2 - u.laplace + damping * u.dt, u.forward))
    op = Operator([stencil] + src_term)

    shape = observed_waveform.shape[1:]
    field_logs = np.zeros((simulation_times, shape[0], shape[1]))
    field_logs[0] = observed_waveform.data[0]

    for i in range(1, simulation_times - 1):
        t_from = start_time + i * unit_time
        t_to = start_time + (i + 1) * unit_time
        op.apply(time_m=t_from, time_M=t_to, dt=unit_time)

        field_logs[i + 1] = observed_waveform.data[0].copy()

    return field_logs


def simulate_backward_waveform(src: SparseTimeFunction, damping: Function, velocity_model: Function, grid: Grid, start_time: float, unit_time: float, simulation_times: int) -> WaveFieldHistory:
    observed_waveform = TimeFunction(name="v", grid=grid, time_order=2, space_order=2)
    vdt2 = TimeFunction(name="v_dt2", grid=grid, time_order=2, space_order=2)

    v = observed_waveform  # alias(長いので
    src_term = src.inject(field=v.forward, expr=src)
    vdt2_term = [Eq(vdt2.forward, v.dt2)]
    stencil = [Eq(v.forward, solve(v.dt2 / velocity_model**2 - v.laplace + damping * v.dt, v.forward))]
    op = Operator(stencil + src_term + vdt2_term)

    shape = v.shape[1:]
    field_logs = np.zeros((simulation_times, shape[0], shape[1]))
    field_logs[0] = v.data[0]

    for i in range(1, simulation_times - 1):
        t_from = start_time + i * unit_time
        t_to = start_time + (i + 1) * unit_time
        op.apply(time_m=t_from, time_M=t_to, dt=unit_time)

        field_logs[i + 1] = vdt2.data[0].copy()

    return field_logs


def generate_initial_velocity_model(shape: Vec2D[int]) -> NDArray[np.float64]:
    return np.ones((shape.y, shape.x)) * 2.5

def save_field_history_as_gif(
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


def main():
    # パラメータ及びシミュレーション用変数設定
    params = Params(
        field_cell_size=Vec2D(100, 100),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        damping_coefficient=0.0005,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        source_cell_pos_in_field=Vec2D(2, 50),
        shots_pos_in_field=[Vec2D(42, i * 10 + 45) for i in range(10)],
    )

    grid = generate_grid(params.field_cell_size, params.damping_cell_thickness, params.cell_meter_size)

    # setup source
    single_forward_source_waveform = SparseTimeFunction(name="src", grid=grid, time_order=2, space_order=1, nt=params.simulation_times * params.unit_time + 1, npoint=1)
    single_forward_source_waveform.data[:, 0] = generate_input_waveform_with_ricker_wavelet(
        params.start_time, params.unit_time, params.simulation_times, params.source_frequency, params.source_peek_time
    )

    # setup backward source
    multi_backward_source_waveform = SparseTimeFunction(name="src", grid=grid, time_order=2, space_order=1, nt=params.simulation_times * params.unit_time + 1, npoint=101)
    multi_backward_source_waveform.coordinates.data[:, 0] = np.linspace(40, 140, num=101) * 10
    multi_backward_source_waveform.coordinates.data[:, 1] = 138 * 10

    damping_model = generate_damping_model_devito_function(grid, params.damping_cell_thickness, params.damping_coefficient)
    true_velocity_model = create_simple_velocity_model_devito_function(grid, params.field_cell_size.y, params.damping_cell_thickness)

    initial_velocity_model = generate_initial_velocity_model(params.field_cell_size + Vec2D(params.damping_cell_thickness * 2, params.damping_cell_thickness * 2) + Vec2D(1, 1))
    velocity_model_function = Function(name="velocity_model", grid=grid, time_order=0, space_order=1)

    current_velocity_model = initial_velocity_model

    print(f"error_norm: {np.linalg.norm(current_velocity_model - true_velocity_model.data, ord=2)}")

    # img = true_velocity_model.data
    # raw_min_value = np.min(img)
    # tmp = np.max(np.abs(img))
    # min_value = 2.75
    # plt.imshow(img[40:141, 40:141], vmin=2.45, vmax=3.05, cmap='jet', extent=(0, 1.0, 1.0, 0))
    # plt.xticks([0, 0.5, 1])
    # plt.yticks([0, 0.5, 1])
    # plt.colorbar()
    # plt.title("velocity model [km/s]")
    # plt.xlabel("X [km]")
    # plt.ylabel("Depth [km]")
    # plt.show()
    # sys.exit(-1)

    # plt.imshow(true_velocity_model.data, cmap='jet', vmin=2.45, vmax=3.05)
    # plt.colorbar()
    # plt.title("true velocity model")
    # plt.show()
    # sys.exit(-1)

    n_iter = 50
    for _ in range(n_iter):
        # 速度モデルの値を更新
        velocity_model_function.data[:, :] = current_velocity_model

        grad = np.zeros(params.field_cell_size + Vec2D(params.damping_cell_thickness * 2, params.damping_cell_thickness * 2) + Vec2D(1, 1))
        residual_norm_sum = 0
        for idx, shot_pos in enumerate(params.shots_pos_in_field):
            # setup source coordinate
            single_forward_source_waveform.coordinates.data[0, 0] = shot_pos.y * 10
            single_forward_source_waveform.coordinates.data[0, 1] = shot_pos.x * 10

            # 真の速度モデルを用いた forward simulation
            true_field_log = simulate_forward_waveform(single_forward_source_waveform, damping_model, true_velocity_model, grid, params.start_time, params.unit_time, params.simulation_times)
            true_observed_waveform = true_field_log[:, 40:-40, 138]

            # 推定速度モデルを用いた forward simulation
            field_log_during_forward_simulation = simulate_forward_waveform(
                single_forward_source_waveform, damping_model, velocity_model_function, grid, params.start_time, params.unit_time, params.simulation_times
            )
            simulated_observed_waveform = field_log_during_forward_simulation[:, 40:-40, 138]

            # 観測データと計算データの残差を計算
            diff_observed_waveform = simulated_observed_waveform - true_observed_waveform

            # multi_backward_source_waveform.data[:-1, :] = diff_observed_waveform
            multi_backward_source_waveform.data[:-1, :] = diff_observed_waveform[::-1]

            # 残差データを用いてbackward simulation
            field_log_during_backward_simulation = simulate_backward_waveform(
                multi_backward_source_waveform, damping_model, velocity_model_function, grid, params.start_time, params.unit_time, params.simulation_times
            )
            dt2_field_log_during_backward_simulation = field_log_during_backward_simulation[::-1]

            # img = np.array(np.sum(dt2_field_log_during_backward_simulation * field_log_during_forward_simulation, axis=0))
            # tmp = np.max(np.abs(img))
            # min_value = -tmp * 2
            # img[40, 40:141] = min_value
            # img[140, 40:141] = min_value
            # img[40:141, 40] = min_value
            # img[40:141, 140] = min_value
            # plt.imshow(img, vmin=-tmp, vmax=tmp, cmap='seismic', extent=(-0.4, 1.4, 1.4, -0.4))
            # plt.xticks([0, 0.5, 1])
            # plt.yticks([0, 0.5, 1])
            # plt.colorbar()
            # plt.title("gradient of velocity model")
            # plt.xlabel("X [km]")
            # plt.ylabel("Depth [km]")
            # plt.show()
            # sys.exit(-1)

            # plt.imshow(np.array(true_observed_waveform.data)[:, 40:-40])
            # plt.colorbar(label="Amplitude")
            # plt.title("wave field")
            # plt.show()
            # sys.exit(-1)

            # field_log = true_field_log
            # min_value = -np.max(np.abs(field_log))
            # field_log[:, 40, 40:141] = min_value
            # field_log[:, 140, 40:141] = min_value
            # field_log[:, 40:141, 40] = min_value
            # field_log[:, 40:141, 140] = min_value
            #
            # path = output_path.joinpath("tmp-fields.gif")
            # tmp = float(np.max(np.abs(field_log)))
            # norm = TwoSlopeNorm(vmin=-tmp, vcenter=0, vmax=tmp)
            # save_field_history_as_gif(field_log, path, norm, extent=(-0.4, 1.4, 1.4, -0.4), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], figsize=(5, 5), field_only=True)
            # sys.exit(-1)

            grad_diff = np.sum(field_log_during_forward_simulation * dt2_field_log_during_backward_simulation, axis=0)
            grad += grad_diff

            residual_norm_sum += np.linalg.norm(diff_observed_waveform, ord=2) ** 2 / 2.0
            print(f"    residual: {np.linalg.norm(diff_observed_waveform, ord=2)}")
            print("    ", np.max(np.abs(field_log_during_forward_simulation)), np.max(np.abs(dt2_field_log_during_backward_simulation)))

        # img = grad
        # tmp = np.max(np.abs(img))
        # min_value = -tmp * 2
        # img[40, 40:141] = min_value
        # img[140, 40:141] = min_value
        # img[40:141, 40] = min_value
        # img[40:141, 140] = min_value
        # plt.imshow(img, vmin=-tmp, vmax=tmp, cmap='seismic', extent=(-0.4, 1.4, 1.4, -0.4))
        # plt.xticks([0, 0.5, 1])
        # plt.yticks([0, 0.5, 1])
        # plt.colorbar()
        # plt.title("gradient sum of velocity model by 9 shots")
        # plt.xlabel("X [km]")
        # plt.ylabel("Depth [km]")
        # plt.show()
        # sys.exit(-1)

        alpha = 0.05 / np.max(grad)
        current_velocity_model -= grad * alpha
        print(f"residual_norm_sum: {residual_norm_sum}")
        print(f"grad norm: {np.linalg.norm(grad, ord=2)}")
        print(f"error_norm: {np.linalg.norm(current_velocity_model - true_velocity_model.data, ord=2)}")
        # plt.imshow(current_velocity_model, cmap="jet", vmin=2.45, vmax=2.75)
        # plt.colorbar(label="Amplitude")
        # plt.title("wave field")
        # plt.show()
        img = current_velocity_model.copy()
        tmp = np.max(np.abs(img))
        min_value = -tmp * 2
        img[40, 40:141] = min_value
        img[140, 40:141] = min_value
        img[40:141, 40] = min_value
        img[40:141, 140] = min_value
        plt.imshow(img, vmin=2.45, vmax=2.75, cmap='jet', extent=(-0.4, 1.4, 1.4, -0.4))
        plt.xticks([0, 0.5, 1])
        plt.yticks([0, 0.5, 1])
        plt.colorbar()
        plt.title("current velocity model")
        plt.xlabel("X [km]")
        plt.ylabel("Depth [km]")
        plt.show()



if __name__ == "__main__":
    main()
