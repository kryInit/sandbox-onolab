from multiprocessing import Queue, Process
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple, NamedTuple

import joblib

import numpy as np
import numpy.typing as npt
from devito import set_log_level
from devito import Eq, Function, Inc, Operator, TimeFunction, norm, solve
from numpy.typing import NDArray

from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver


class FastParallelVelocityModelProps(NamedTuple):
    true_velocity_model: npt.NDArray
    initial_velocity_model: npt.NDArray
    shape: Tuple[int, int]
    spacing: Tuple[float, float]
    damping_cell_thickness: int
    start_time: float
    end_time: float
    source_frequency: float
    source_locations: npt.NDArray
    receiver_locations: npt.NDArray


class FastParallelVelocityModelGradientCalculator:
    def __init__(self, props: FastParallelVelocityModelProps):
        self.n_shots = len(props.source_locations)

        dsize = props.damping_cell_thickness
        vm_shape = (props.shape[1] + dsize * 2, props.shape[0] + dsize * 2)

        self.velocity_model_shared_memory = SharedMemory(create=True, size=np.prod(vm_shape) * np.dtype(np.float32).itemsize)
        self.velocity_model = np.ndarray(vm_shape, dtype=np.float32, buffer=self.velocity_model_shared_memory.buf)

        self.residual_norm_shared_memories = []
        self.vm_grad_shared_memories = []
        self.residual_norms = []
        self.vm_grads = []
        for i in range(self.n_shots):
            residual_shared_memory = SharedMemory(create=True, size=1 * np.dtype(np.float32).itemsize)
            residual = np.ndarray(1, dtype=np.float32, buffer=residual_shared_memory.buf)
            self.residual_norm_shared_memories.append(residual_shared_memory)
            self.residual_norms.append(residual)

            grad_shared_memory = SharedMemory(create=True, size=np.prod(vm_shape) * np.dtype(np.float32).itemsize)
            grad = np.ndarray((vm_shape[1], vm_shape[0]), dtype=np.float32, buffer=grad_shared_memory.buf)
            self.vm_grad_shared_memories.append(grad_shared_memory)
            self.vm_grads.append(grad)

        self.input_queue = Queue()
        self.output_queue = Queue()

        print("process initializing...")
        self.processes = [Process(target=calc_grad_worker, args=(props, i, self.velocity_model_shared_memory.name, self.vm_grad_shared_memories[i].name, self.residual_norm_shared_memories[i].name, self.input_queue, self.output_queue)) for i in range(self.n_shots)]
        for p in self.processes:
            p.start()

        # 初期化が終わるまで待つ
        for _ in range(self.n_shots):
            self.output_queue.get()

        print("process initialized!")

    def __del__(self):
        for _ in range(self.n_shots):
            self.input_queue.put(None)

        for p in self.processes:
            p.join()

        self.velocity_model_shared_memory.close()
        self.velocity_model_shared_memory.unlink()

        for residual_norm in self.residual_norm_shared_memories:
            residual_norm.close()
            residual_norm.unlink()

        for grad_shared_memory in self.vm_grad_shared_memories:
            grad_shared_memory.close()
            grad_shared_memory.unlink()

    def calc_grad(self, current_velocity_model: npt.NDArray) -> Tuple[float, NDArray[np.float32]]:
        self.velocity_model[:] = current_velocity_model.T

        for i in range(self.n_shots):
            self.input_queue.put(0)

        for _ in range(self.n_shots):
            self.output_queue.get()

        objective = 0
        grad_value = np.zeros_like(current_velocity_model)
        for i in range(self.n_shots):
            objective += self.residual_norms[i][0]
            grad_value += self.vm_grads[i]

        return objective, grad_value


def calc_grad_worker(
        props: FastParallelVelocityModelProps,
        idx: int,
        velocity_model_shared_memory_name: str,
        vm_grad_shared_memory_name: str,
        residual_norm_shared_memory_name: str,
        input_queue: Queue,
        output_queue: Queue
):
    vm_shape = (props.shape[1] + props.damping_cell_thickness*2, props.shape[0] + props.damping_cell_thickness*2)

    velocity_model_shared_memory = SharedMemory(name=velocity_model_shared_memory_name)
    velocity_model = np.ndarray(vm_shape, dtype=np.float32, buffer=velocity_model_shared_memory.buf)

    vm_grad_shared_memory = SharedMemory(name=vm_grad_shared_memory_name)
    vm_grad = np.ndarray((vm_shape[1], vm_shape[0]), dtype=np.float32, buffer=vm_grad_shared_memory.buf)

    residual_norm_shared_memory = SharedMemory(name=residual_norm_shared_memory_name)
    residual_norm = np.ndarray(1, dtype=np.float32, buffer=residual_norm_shared_memory.buf)

    grad_calculator = FastParallelVelocityModelGradientCalculatorHelper(props, idx)

    # 初期化が終わったことを伝える
    output_queue.put(0)

    while True:
        code = input_queue.get()
        if code != 0:
            break

        residual_norm_sum, grad = grad_calculator.calc_grad(velocity_model)

        vm_grad[:] = grad
        residual_norm[0] = residual_norm_sum

        output_queue.put(0)

    velocity_model_shared_memory.close()
    vm_grad_shared_memory.close()


class FastParallelVelocityModelGradientCalculatorHelper:
    def __init__(self, props: FastParallelVelocityModelProps, idx: int):
        self.true_model = SeismicModel(space_order=2, vp=props.true_velocity_model.T, origin=(0, 0), shape=props.shape, dtype=np.float32, spacing=props.spacing, nbl=props.damping_cell_thickness, bcs="damp", fs=False)
        self.current_model = SeismicModel(space_order=2, vp=props.initial_velocity_model.T, origin=(0, 0), shape=props.shape, dtype=np.float32, spacing=props.spacing, nbl=props.damping_cell_thickness, bcs="damp", fs=False)

        self.geometry = AcquisitionGeometry(self.true_model, props.receiver_locations, np.array([[0, 0]]), props.start_time, props.end_time, f0=props.source_frequency, src_type="Ricker")
        self.geometry.src_positions[0][:] = props.source_locations[idx]

        self.simulator = AcousticWaveSolver(self.true_model, self.geometry, space_order=4)

        self.grad_operator = self._create_grad_op(self.true_model, self.geometry, self.simulator)

        self.observed_waveform = self._calc_true_observed_waveform(self.true_model, self.geometry, self.simulator)


    @classmethod
    def _create_grad_op(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver) -> Operator:
        grad = Function(name="grad", grid=true_model.grid)
        u = TimeFunction(name="u", grid=true_model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
        v = TimeFunction(name="v", grid=true_model.grid, save=None, time_order=2, space_order=solver.space_order)

        eqns = [Eq(v.backward, solve(true_model.m * v.dt2 - v.laplace + true_model.damp * v.dt.T, v.backward))]
        rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * true_model.grid.stepping_dim.spacing**2 / true_model.m)
        gradient_update = Inc(grad, -u.dt2 * v * true_model.m**1.5)

        return Operator(eqns + rec_term + [gradient_update], subs=true_model.spacing_map, name="Gradient")

    @classmethod
    def _calc_true_observed_waveform(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, simulator: AcousticWaveSolver) -> List[NDArray]:
        observed_waveform = Receiver(name="d_obs", grid=true_model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
        simulator.forward(vp=true_model.vp, rec=observed_waveform)
        return observed_waveform.data.copy()

    def calc_grad(self, current_velocity_model: npt.NDArray) -> Tuple[float, NDArray[np.float32]]:
        grad = Function(name="grad", grid=self.true_model.grid)
        residual = Receiver(name="residual", grid=self.current_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        calculated_waveform = Receiver(name="d_syn", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)

        # 現在のvelocity modelの値をupdate
        self.current_model.vp.data[:] = current_velocity_model.T

        # 現在のモデル: vp_in を用いて、計算データ波形(calculated_waveform)を計算
        _, calculated_wave_field, _ = self.simulator.forward(vp=self.current_model.vp, save=True, rec=calculated_waveform)

        # 観測データと計算データの残差を計算
        residual.data[:] = calculated_waveform.data - self.observed_waveform

        # 雑なobjective計算
        objective = 0.5 * np.sum(np.abs(residual.data ** 2))

        # ちゃんとしたobjective計算
        # objective = 0.5 * norm(residual) ** 2

        self.grad_operator.apply(rec=residual, grad=grad, u=calculated_wave_field, dt=self.simulator.dt, vp=self.current_model.vp)

        return objective, -grad.data

