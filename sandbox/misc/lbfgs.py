import numpy as np

from lib.signal_processing.l_bfgs import LBFGS
from lib.signal_processing.l_bfgs import lbfgs as lbfgs2

x = np.array([-2.5, 0.0])
lbfgs = LBFGS(step_size=1, m=5)


def objective(x):
    # 5x^2 - 6xy + 5y^2 - 10x + 6y
    # grad: 10x - 6y - 10, 10y - 6x + 6
    # ans:  x = 1, y = 0
    return 5.0 * x[0] ** 2.0 - 6.0 * x[0] * x[1] + 5.0 * x[1] ** 2 - 10.0 * x[0] + 6.0 * x[1]
    # return x[0] ** 2 + x[1] ** 2


def gradient(x):
    return np.array([10.0 * x[0] - 6.0 * x[1] - 10.0, 10.0 * x[1] - 6.0 * x[0] + 6.0])
    # return np.array([2 * x[0], 2 * x[1]])


xx, out = lbfgs2(x, objective, gradient, stepsize=1.0, maxiterate=20, memorysize=5, epsilon=10.0 ** (-8))
print(xx)

f_val = objective(x)
grad_val = gradient(x)

print(f"ステップ {-1}: x = {x}, f = {f_val}, |grad| = {np.linalg.norm(grad_val)}")

for i in range(10):
    x = lbfgs.step(x, f_val, grad_val)  # xを渡す
    f_val = objective(x)
    grad_val = gradient(x)
    print(f"ステップ {i}: x = {x}, f = {f_val}, |grad| = {np.linalg.norm(grad_val)}")
    if np.linalg.norm(grad_val) < 1e-5:  # 勾配のノルムが小さいときに収束
        print("収束しました")
        break
