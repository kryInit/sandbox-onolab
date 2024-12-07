import numpy as np


def searchdirection(s, y, g):
    q = -np.copy(g)
    num = len(s)
    a = np.zeros(num)

    if num == 0:
        return q
    # print(s, y)

    for i in np.arange(num)[::-1]:
        a[i] = np.dot(s[i], q) / np.dot(y[i], s[i])
        q -= a[i] * y[i]

    q = np.dot(s[-1], y[-1]) / np.dot(y[-1], y[-1]) * q

    for i in range(num):
        b = np.dot(y[i], q) / np.dot(y[i], s[i])
        q += (a[i] - b) * s[i]

    return q


class LBFGS:
    def __init__(self, m=10, step_size=0.1):
        """
        L-BFGSのクラスを初期化します。

        Parameters:
        - m: L-BFGSで保持する履歴の数
        - step_size: 固定のステップサイズ
        """
        self.m = m
        self.step_size = step_size  # 固定ステップサイズ
        self.s_list = []
        self.y_list = []
        self.x = None
        self.g_prev = None
        self.f_prev = None

    def step(self, x, f_val, grad_val):
        """
        1ステップのL-BFGS更新を行います。

        Parameters:
        - x: 現在の位置（更新前）
        - f_val: 現在の目的関数の値
        - grad_val: 現在の勾配の値

        Returns:
        - x_new: 更新後の位置
        """
        if self.x is None:
            self.x = np.copy(x)
            self.f_prev = f_val
            self.g_prev = grad_val

        else:
            # 履歴を更新
            y_k = grad_val - self.g_prev
            self.y_list.append(y_k)
            if len(self.s_list) == self.m:
                self.y_list.pop(0)

            # 次のステップのために更新
            self.g_prev = grad_val
            self.f_prev = f_val

        # 更新方向を計算
        p = self.calculate_update_direction(self.g_prev)

        # 更新
        x_new = self.x + (self.step_size * 0.01 if len(self.s_list) == 0 else self.step_size) * p

        # 履歴を更新
        s_k = x_new - self.x
        self.s_list.append(s_k)
        if len(self.s_list) == self.m:
            self.s_list.pop(0)
        self.x = x_new

        return x_new

    def calculate_update_direction(self, g):
        """
        更新方向を計算します（逆ヘッセ行列の近似を使用）。

        Parameters:
        - g: 現在の勾配

        Returns:
        - p: 更新方向
        """
        if len(self.s_list) == 0:
            return -g.copy()

        # return searchdirection(np.array(self.s_list.copy()), np.array(self.y_list.copy()), g.copy())

        q = g.copy()
        alpha = np.zeros(len(self.s_list))

        # 第1段階: 右から掛ける
        for i in range(len(self.s_list) - 1, -1, -1):
            s_dot_y = np.dot(self.s_list[i], self.y_list[i]) + 1e-8
            alpha[i] = np.dot(self.s_list[i], q) / s_dot_y
            q -= alpha[i] * self.y_list[i]

        # 初期推定 (H_0)
        if len(self.s_list) > 0:
            s_dot_y_last = np.dot(self.s_list[-1], self.y_list[-1])
            if s_dot_y_last != 0:
                H_0 = s_dot_y_last / np.dot(self.y_list[-1], self.y_list[-1])
            else:
                H_0 = 1.0
        else:
            H_0 = 1.0
        z = H_0 * q

        # 第2段階: 左から掛ける
        for i in range(len(self.s_list)):
            s_dot_y = np.dot(self.s_list[i], self.y_list[i]) + 1e-8
            beta = np.dot(self.y_list[i], z) / s_dot_y
            z += (alpha[i] - beta) * self.s_list[i]

        return -z  # 更新方向は負の勾配方向


def lbfgs(x, f, g, stepsize, maxiterate, memorysize, epsilon):

    outx = []
    s = np.empty((0, len(x)))
    y = np.empty((0, len(x)))
    xold = x.copy()
    gold = g(xold)
    J1 = f(xold)
    mmax = memorysize
    sp = stepsize
    print("f= ", J1)

    print(f"iter: {-1}, x: {xold}, f: {J1}, |g|: {np.linalg.norm(gold)}")

    outx.append(xold)
    for num in range(maxiterate):
        if np.linalg.norm(gold) < epsilon:
            print("g=", np.linalg.norm(gold))
            break

        d = searchdirection(s, y, gold)

        sp = stepsize * 0.01 if num == 0 else stepsize

        xnew = xold + sp * d
        gnew = g(xnew)
        J2 = f(xnew)

        J1 = J2
        si, yi = xnew - xold, gnew - gold
        if len(s) == mmax:
            s = np.roll(s, -1, axis=0)
            y = np.roll(y, -1, axis=0)
            s[-1] = si
            y[-1] = yi
        else:
            s = np.append(s, [si], axis=0)
            y = np.append(y, [yi], axis=0)

        xold, gold = xnew, gnew

        print(f"iter: {num}, x: {xold}, f: {J1}, |g|: {np.linalg.norm(gold)}, stepsize: {sp}")
        outx.append(xold)

    return xold, outx
