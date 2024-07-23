from typing import Callable


def calc_pds_x_step[TX, TY](current_x: TX, y: TY, gamma1: float, calc_grad: Callable[[TX], TX], At: Callable[[TY], TX], calc_prox_g: [[TX, float], TX]) -> TX:
    tmp = current_x - gamma1 * (calc_grad(current_x) + At(y))
    return calc_prox_g(tmp, gamma1)


def calc_pds_y_step[TX, TY](current_x: TX, prev_x: TX, current_y: TY, gamma2: float, A: Callable[[TX], TY], calc_prox_h: [[TY, float], TY]) -> TX:
    tmp = current_y + gamma2 * A(2 * current_x - prev_x)
    return tmp - gamma2 * calc_prox_h(tmp / gamma2, 1.0 / gamma2)


def solve_by_pds[
    TX, TY
](
    initial_x: TX,
    gamma1: flaot,
    gamma2: float,
    calc_grad: Callable[[TX], TX],
    A: Callable[[TX], TY],
    At: Callable[[TY], TX],
    calc_prox_g: [[TX, float], TX],
    calc_prox_h: [[TY, float], TY],
    n_iter: int,
) -> TX:
    current_x = initial_x
    current_y = A(initial_x)

    for _ in range(n_iter):
        prev_x = current_x
        current_x = calc_pds_x_step(current_x, current_y, gamma1, calc_grad, At, calc_prox_g)
        current_y = calc_pds_y_step(current_x, prev_x, current_y, gamma2, A, calc_prox_h)
