import math
import numpy as np
import scipy.stats as sts

class DSV:
    def __init__(self, N, M, n, m, accuracy):
        dirichlet_distribution = np.random.dirichlet(np.ones(N * M))
        self.XY = self.vector_to_matrix(dirichlet_distribution, N, M)
        self.delta = self.get_delta(self.XY)

        self.X = self.get_component(N, n)
        self.Y = self.get_component(M, m)

        # Функция распределения ДСВ для Х
        self.f = self.F()
        # Условный закон распределения ДСВ Y при X
        self.f_x = self.F_x(N, M)

        self.dsv = self.generate_dsv(accuracy)

    def get_component(self, N, n):
        array = []
        while True:
            temp = np.random.randint(1, n)
            if temp not in array:
                array.append(temp)
                if len(array) == N:
                    return np.sort(np.array(array))

    def vector_to_matrix(self, vector, n, m):
        matrix = []
        for i in range(n):
            temp_vector = []
            for j in range(m):
                temp_vector.append(vector[(i * m) + j])
            matrix.append(temp_vector)
        return matrix

    def get_delta(self, XY):
        delta = []
        row = np.shape(XY)[0]
        for i in range(row):
            delta.append(sum(XY[i]))
        return delta

    def F(self):
        F = np.cumsum(self.delta)
        return np.array(F)

    def F_x(self, N, M):
        F_x = []
        for i in range(N):
            F_x.append([])
            temp_delta = np.cumsum(self.XY[i])
            for j in range(M):
                F_x[-1].append(temp_delta[j] / self.delta[i])
        return np.array(F_x)

    def generate_dsv(self, n):
        discrete_XY = []
        for i in range(n):
            x, y = np.random.uniform(size=2)
            x_index = np.searchsorted(self.f, x)
            y_index = np.searchsorted(self.f_x[x_index], y)
            discrete_XY.append([self.X[x_index], self.Y[y_index]])
        return discrete_XY

    def get_empirical_probability(self):
        empirical_matrix = np.zeros((self.X.size, self.Y.size))
        dsv_len = len(self.dsv)
        for x, y in self.dsv:
            x_index = np.where(self.X == x)
            y_index = np.where(self.Y == y)
            empirical_matrix[x_index, y_index] = self.dsv.count([x, y]) / dsv_len
        return empirical_matrix

    def get_mx_theoretical(self):
        delta_x = np.sum(self.XY, axis=1)
        mx_t = sum(delta_x * self.X)
        return mx_t

    def get_my_theoretical(self):
        delta_y = np.sum(self.XY, axis=0)
        my_t = sum(delta_y * self.Y)
        return my_t

    def get_mx_empirical(self):
        delta_x = 0
        for x, _ in self.dsv:
            delta_x += x
        dsv_len = len(self.dsv)
        mx_e = delta_x / dsv_len
        return mx_e

    def get_my_empirical(self):
        delta_y = 0
        for _, y in self.dsv:
            delta_y += y
        dsv_len = len(self.dsv)
        my_e = delta_y / dsv_len
        return my_e

    def get_dx_theoretical(self, mx_t = None):
        mx_t = self.get_mx_theoretical() if mx_t is None else mx_t
        delta_x = np.sum(self.XY, axis=1)
        mx2_t = sum(delta_x * (self.X ** 2))
        dx_t = mx2_t - mx_t ** 2
        return dx_t

    def get_dy_theoretical(self, my_t = None):
        my_t = self.get_my_theoretical() if my_t is None else my_t
        delta_y = np.sum(self.XY, axis=0)
        my2_t = sum(delta_y * (self.Y ** 2))
        dy_t = my2_t - my_t ** 2
        return dy_t

    def get_dx_empirical(self, mx_e = None):
        mx_e = self.get_mx_empirical() if mx_e is None else mx_e
        delta_x = 0
        for x, _ in self.dsv:
            delta_x += (x - mx_e) ** 2
        dsv_len = len(self.dsv)
        dx_e = delta_x / (dsv_len - 1)
        return dx_e

    def get_dy_empirical(self, my_e = None):
        my_e = self.get_my_empirical() if my_e is None else my_e
        delta_y = 0
        for _, y in self.dsv:
            delta_y += (y - my_e) ** 2
        dsv_len = len(self.dsv)
        dy_e = delta_y / (dsv_len - 1)
        return dy_e

    def get_interval_estimations_mx(self, mx_e = None, dx_e = None,):
        mx_e = self.get_mx_empirical() if mx_e is None else mx_e
        dx_e = self.get_dx_empirical(mx_e) if dx_e is None else dx_e

        dsv_len = len(self.dsv)
        tt = sts.t(dsv_len - 1)
        arr = tt.rvs(1000000)
        delta_x = sts.mstats.mquantiles(arr, prob=0.95) * math.sqrt(dx_e / (dsv_len - 1))
        interval_estimations_x = mx_e - delta_x, mx_e + delta_x
        return interval_estimations_x

    def get_interval_estimations_my(self, my_e = None, dy_e = None):
        my_e = self.get_my_empirical() if my_e is None else my_e
        dy_e = self.get_dy_empirical(my_e) if dy_e is None else dy_e

        dsv_len = len(self.dsv)
        tt = sts.t(dsv_len - 1)
        arr = tt.rvs(1000000)
        delta = sts.mstats.mquantiles(arr, prob=0.95) * math.sqrt(dy_e / (dsv_len - 1))
        interval_estimations_y = my_e - delta, my_e + delta
        return interval_estimations_y

    def get_interval_estimations_dx(self, dx_e = None):
        dx_e = self.get_dx_empirical() if dx_e is None else dx_e

        dsv_len = len(self.dsv)
        tt = sts.chi2(dsv_len - 1)
        arr = tt.rvs(1000000)
        delta = sts.mstats.mquantiles(arr, prob=[0.01, 0.99])  # # 0.99
        interval_estimate_x = (dsv_len * dx_e / delta[1], dsv_len * dx_e / delta[0])
        return interval_estimate_x

    def get_interval_estimations_dy(self, dy_e=None):
        dy_e = self.get_dy_empirical() if dy_e is None else dy_e

        dsv_len = len(self.dsv)
        tt = sts.chi2(dsv_len - 1)
        arr = tt.rvs(1000000)
        delta = sts.mstats.mquantiles(arr, prob=[0.01, 0.99])  # # 0.99
        interval_estimate_y = (dsv_len * dy_e / delta[1], dsv_len * dy_e / delta[0])
        return interval_estimate_y

    def get_correlation_coefficient(self, mx_e = None, my_e = None):
        mx_e = self.get_mx_empirical() if mx_e is None else mx_e
        my_e = self.get_my_empirical() if my_e is None else my_e
        dividend = 0
        for x, y in self.dsv:
            dividend += (x - mx_e) * (y - my_e)

        temp = []
        for x, y in self.dsv:
            temp.append([(x - mx_e) ** 2, (y - my_e)** 2])

        sum_x, sum_y = 0, 0
        for x, y in temp:
            sum_x += x
            sum_y += y
        divider = math.sqrt(sum_x * sum_y)

        correlation_coefficient = dividend / divider
        return correlation_coefficient

    def get_correlation(self, matrix, mx, my, dx, dy):
        divider = (mx * my) * - 1
        for i in range(len(self.X)):
            for j in range(len(self.Y)):
                divider += self.X[i] * self.Y[j] * matrix[i][j]

        correlation = divider / np.sqrt(dx * dy)
        return correlation

    def print_stat(self):
        print("X:\n", self.X)
        print("Y:\n", self.Y)
        print("XY:\n", self.XY)
        print("Delta:\n", self.delta)
        print("Функция распределения X:\n", self.f)
        print("Условный закон распредления Y при X:\n",self.f_x)
        print("ДСВ:\n", self.dsv)