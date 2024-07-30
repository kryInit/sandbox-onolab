import time
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

import numpy as np

global_shape = (10,)
shared_shape = (10,)


def worker(index: int, velocity_model_shared_memory_name: str, vm_grad_shared_memory_name: str, input_queue: Queue, output_queue: Queue):
    global_shm = SharedMemory(name=velocity_model_shared_memory_name)
    global_array = np.ndarray(global_shape, dtype=np.float64, buffer=global_shm.buf)

    shared_shm = SharedMemory(name=vm_grad_shared_memory_name)
    shared_array = np.ndarray(shared_shape, dtype=np.float64, buffer=shared_shm.buf)

    time.sleep(5)

    while True:
        code = input_queue.get()
        if code != 0:
            break

        shared_array[index] += global_array[index] + 1
        time.sleep(1)

        output_queue.put(0)

    global_shm.close()
    shared_shm.close()


def main():
    start_time = time.time()
    n_jobs = 10

    velocity_model_shared_memory = SharedMemory(create=True, size=np.prod(global_shape) * np.dtype(np.float64).itemsize)
    global_array = np.ndarray(global_shape, dtype=np.float64, buffer=velocity_model_shared_memory.buf)
    global_array.fill(0)

    vm_grad_shared_memory = []
    shared_arrays = []
    for i in range(n_jobs):
        m = SharedMemory(create=True, size=np.prod(shared_shape) * np.dtype(np.float64).itemsize)
        shared_array = np.ndarray(shared_shape, dtype=np.float64, buffer=m.buf)
        shared_array.fill(0)
        vm_grad_shared_memory.append(m)
        shared_arrays.append(shared_array)

    input_queue = Queue()
    output_queue = Queue()

    processes = [Process(target=worker, args=(i, velocity_model_shared_memory.name, vm_grad_shared_memory[i].name, input_queue, output_queue)) for i in range(n_jobs)]
    for p in processes:
        p.start()

    n_iters = 5
    for iteration in range(n_iters):
        print(f"\n--- Iteration {iteration + 1} ---")

        for i in range(n_jobs):
            input_queue.put(0)

        for _ in range(n_jobs):
            output_queue.get()

        for i in range(n_jobs):
            global_array += shared_arrays[i]

        print(f"global: {global_array}")
        for i in range(n_jobs):
            print(f"local: {shared_arrays[i]}")

    for _ in range(n_jobs):
        input_queue.put(None)

    for p in processes:
        p.join()

    print(f"final global: {global_array}")

    velocity_model_shared_memory.close()
    velocity_model_shared_memory.unlink()

    for m in vm_grad_shared_memory:
        m.close()
        m.unlink()

    print(f"elapsed: {time.time() - start_time}")


if __name__ == "__main__":
    main()
