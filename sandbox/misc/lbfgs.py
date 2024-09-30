"""
Full waveform inversion (FWI) aims to reconstruct subsurface properties from observed seismic data.
Since FWI is an ill-posed inverse problem, appropriate regularizations or constraints are useful approaches to achieve accurate reconstruction.
The total variation (TV) -type regularization or constraint is widely known as a powerful prior that models the piecewise smoothness of subsurface properties.
However, the optimization problem of the TV-type regularized or constrained FWI is difficult to solve due to the non-linearity of the observation process and the non-smoothness of the TV-type regularization or constraint.
Conventional approaches to solve the problem rely on an inner loop and/or approximations, resulting in high computational cost and/or suboptimal value.
In this paper, we develop an efficient algorithm without an inner loop and approximations to solve the problem using a primal-dual splitting method.
Finally, we demonstrate the effectiveness of the proposed method through experiments using the SEG/EAGE Salt and Overthrust Models.
"""
import numpy as np


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
        self.x = None  # 初期位置を後で設定
        self.g_prev = None  # 前回の勾配を保持
        self.f_prev = None  # 前回の目的関数の値を保持

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

        # 更新方向を計算
        p = self.calculate_update_direction(self.g_prev)

        # 更新
        x_new = self.x + self.step_size * p

        # 履歴を更新
        s_k = x_new - self.x
        y_k = grad_val - self.g_prev

        if len(self.s_list) == self.m:
            self.s_list.pop(0)
            self.y_list.pop(0)

        self.s_list.append(s_k)
        self.y_list.append(y_k)

        # 次のステップのために更新
        self.x = x_new
        self.g_prev = grad_val
        self.f_prev = f_val

        return x_new

    def calculate_update_direction(self, g):
        """
        更新方向を計算します（逆ヘッセ行列の近似を使用）。

        Parameters:
        - g: 現在の勾配

        Returns:
        - p: 更新方向
        """
        q = g.copy()
        alpha = np.zeros(len(self.s_list))

        # 第1段階: 右から掛ける
        for i in range(len(self.s_list) - 1, -1, -1):
            s_dot_y = np.dot(self.s_list[i], self.y_list[i])
            if s_dot_y == 0:  # ゼロ割りを防ぐ
                continue
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
            s_dot_y = np.dot(self.s_list[i], self.y_list[i])
            if s_dot_y == 0:  # ゼロ割りを防ぐ
                continue
            beta = np.dot(self.y_list[i], z) / s_dot_y
            z += (alpha[i] - beta) * self.s_list[i]

        return -z  # 更新方向は負の勾配方向


# クラスの使用例
x = np.array([1.0, 1.0])  # 初期点
lbfgs = LBFGS(step_size=0.1)  # 固定ステップサイズ0.1でL-BFGSクラスを初期化


# 目的関数と勾配の初期計算
def objective(x):
    return x[0] ** 2 + x[1] ** 2


def gradient(x):
    return np.array([2 * x[0], 2 * x[1]])


f_val = objective(x)
grad_val = gradient(x)

# 外部で管理して1ステップずつ最適化
for i in range(100):
    x = lbfgs.step(x, f_val, grad_val)  # xを渡す
    f_val = objective(x)
    grad_val = gradient(x)
    print(f"ステップ {i}: x = {x}, f = {f_val}, |grad| = {np.linalg.norm(grad_val)}")
    if np.linalg.norm(grad_val) < 1e-5:  # 勾配のノルムが小さいときに収束
        print("収束しました")
        break
