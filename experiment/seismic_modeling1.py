# referred to https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html

from pathlib import Path
from typing import NamedTuple, Tuple, Union

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


Field = NDArray[np.float64]
FieldLog = Field


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


def generate_input_source_devito_function(
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
    src = SparseTimeFunction(name="src", grid=grid, time_order=2, space_order=1, nt=simulation_times + 2, npoint=1)
    src_coordinate = (source_cell_pos_in_field + Vec2D(damping_cell_thickness, damping_cell_thickness)) * cell_meter_size
    src.coordinates.data[0, 0] = src_coordinate.x
    src.coordinates.data[0, 1] = src_coordinate.y

    time_range = np.linspace(start_time, start_time + unit_time * simulation_times, simulation_times + 1)

    # use ricker wavelet
    src_values = (1.0 - 2.0 * (np.pi * frequency * (time_range - source_peek_time)) ** 2) * np.exp(-((np.pi * frequency * (time_range - source_peek_time)) ** 2))
    src.data[:-1, 0] = src_values

    return src


def generate_damping_model_devito_function(grid: Grid, damping_cell_thickness: int, damping_coefficient: float) -> Function:
    damping_model = Function(name="eta", grid=grid, space_order=0)

    for i in range(damping_cell_thickness):
        val = (damping_cell_thickness - i) * damping_coefficient
        damping_model.data[:, i] = val
        damping_model.data[:, -(i + 1)] = val
        damping_model.data[i, :] = val
        damping_model.data[-(i + 1), :] = val

    return damping_model


def create_simple_velocity_model_devito_function(grid: Grid, field_cell_size_y, damping_cell_thickness: int) -> Function:
    v = Function(name="v", grid=grid, time_order=0, space_order=1)

    # 上半分を1.5[km/s], 下半分を2.5[km/s]とする
    velocity_model = 1.5 * np.ones(grid.shape)
    velocity_model[damping_cell_thickness + field_cell_size_y // 2 : -damping_cell_thickness, :] = 2.5
    v.data[:, :] = velocity_model

    return v


def generate_operator(u: TimeFunction, src: SparseTimeFunction, damping: Function, v: Function) -> Operator:
    damping_term = damping * u.dt
    src_term = src.inject(field=u.forward, expr=src.dt2 * v**2)

    pde = u.dt2 / v**2 - u.laplace + damping_term
    stencil = Eq(u.forward, solve(pde, u.forward))

    return Operator([stencil] + src_term)


def simulation(start_time: float, unit_time: float, simulation_times: int, op: Operator, observed_waveform: TimeFunction) -> FieldLog:
    shape = observed_waveform.shape[1:]
    field_logs = np.zeros((simulation_times, shape[0], shape[1]))
    field_logs[0] = observed_waveform.data[0]

    for i in range(1, simulation_times - 1):
        t_from = start_time + i * unit_time
        t_to = start_time + (i + 1) * unit_time
        op.apply(time_m=t_from, time_M=t_to, dt=unit_time)

        field_logs[i + 1] = observed_waveform.data[0].copy()

    return field_logs


def show_field_snapshot(field: Field, grid_extent: Tuple[float, float], norm: Union[Normalize | None] = None):
    plt.imshow(field, cmap="seismic", extent=(0, grid_extent[0], 0, grid_extent[1]), norm=norm)
    plt.colorbar(label="Amplitude")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("wave field")
    plt.show()


def show_samping_waveform(
    field_logs: Field,
    depth_cell: int,
    damping_cell_thickness: int,
    grid_extent: Tuple[float, float],
    cell_meter_size: Vec2D[float],
):
    samples = field_logs[:, damping_cell_thickness + depth_cell, damping_cell_thickness:-damping_cell_thickness]
    scale = float(np.max(samples)) / 10
    plt.imshow(
        samples,
        cmap="gray",
        vmin=-scale,
        vmax=scale,
        extent=(
            0,
            grid_extent[0] - damping_cell_thickness * cell_meter_size.x,
            0,
            grid_extent[1] - damping_cell_thickness * cell_meter_size.y,
        ),
    )
    plt.colorbar(label="Amplitude")
    plt.show()


def save_field_logs_as_gif(field_logs: FieldLog, path: Path, norm: Union[Normalize | None] = None):
    with imageio.get_writer(path, mode="I", duration=0.02) as writer:
        for i in tqdm(range(field_logs.shape[0])):
            fig, ax = plt.subplots()
            cax = ax.imshow(field_logs[i], cmap="seismic", interpolation="nearest", norm=norm)
            plt.axis("off")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            plt.close(fig)


def visualize(params: Params, grid: Grid, source_waveform: SparseTimeFunction, field_logs: FieldLog):
    # 表示時のcolor normalization用クラス
    norm = TwoSlopeNorm(vmin=float(np.min(field_logs)), vcenter=0, vmax=float(np.max(field_logs)))

    # show source waveform
    plt.plot(source_waveform.data), plt.title("source waveform"), plt.show()

    show_field_snapshot(field_logs[-1], grid.extent, norm)

    show_samping_waveform(field_logs, 2, params.damping_cell_thickness, grid.extent, params.cell_meter_size)

    img_path = output_path.joinpath("fields.gif")
    save_field_logs_as_gif(field_logs, img_path, norm)


def main():
    # パラメータ及びシミュレーション用変数設定
    params = Params(
        field_cell_size=Vec2D(100, 100),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=10,
        damping_coefficient=0.01,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        source_cell_pos_in_field=Vec2D(2, 50),
    )
    grid = generate_grid(params.field_cell_size, params.damping_cell_thickness, params.cell_meter_size)
    source_waveform = generate_input_source_devito_function(
        grid,
        params.start_time,
        params.unit_time,
        params.simulation_times,
        params.source_frequency,
        params.source_peek_time,
        params.source_cell_pos_in_field,
        params.damping_cell_thickness,
        params.cell_meter_size,
    )
    damping_model = generate_damping_model_devito_function(grid, params.damping_cell_thickness, params.damping_coefficient)
    velocity_model = create_simple_velocity_model_devito_function(grid, params.field_cell_size.y, params.damping_cell_thickness)

    # 計算対象
    observed_waveform = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)

    # create operator
    operator = generate_operator(observed_waveform, source_waveform, damping_model, velocity_model)

    # simulate
    field_logs = simulation(params.start_time, params.unit_time, params.simulation_times, operator, observed_waveform)

    # visualize
    visualize(params, grid, source_waveform, field_logs)


if __name__ == "__main__":
    main()
