from dsv import DSV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def draw_graphs(t_matrix, e_matrix, dsv, X, Y):
    # Basic Configuration
    fig, axes = plt.subplots(ncols=2, figsize=(14, 14))
    ax1, ax2 = axes

    # Heat maps.
    sns.heatmap(t_matrix, ax=ax1, xticklabels=Y, yticklabels=X, annot=True, vmin=0, vmax=1, square=True, cmap='coolwarm', linewidths=0.5, linecolor="black")
    ax1.set_title('Theoretical_matrix')

    sns.heatmap(e_matrix, ax=ax2, xticklabels=Y, yticklabels=X, annot=True, vmin=0, vmax=1, square=True, cmap='coolwarm', linewidths=0.5, linecolor="black")
    ax2.set_title('Empirical_matrix')

    plt.show()

    fig, axes = plt.subplots(ncols=2, figsize=(8, 8))
    ax1, ax2 = axes

    count = len(dsv)

    x_elements = [item[0] for item in dsv]
    values = [x_elements.count(x) / count for x in X]
    ax1.bar(X, values, width=0.1)
    ax1.set_title('Empirical_matrix _ X')

    y_elements = [item[1] for item in dsv]
    values = [y_elements.count(y) / count for y in Y]
    ax2.bar(Y, values, width=0.1)
    ax2.set_title('Empirical_matrix _ Y')
    plt.show()

def print_result(min, max, value, parametr):
    if min < value < max:
        print("Точечная вероятность", parametr, "попала в интервальный промежуток с вероятностью 0.99'")
    else:
        print("Точечная вероятность", parametr, "не попала в интервальный промежуток с вероятностью 0.99'")

if __name__ == "__main__":
    dsv = DSV(8, 8, 10, 10, 1000)
    dsv.print_stat()

    empirical_matrix = dsv.get_empirical_probability()
    print('Эмперическая матрица распределения:\n', empirical_matrix)
    draw_graphs(dsv.XY, empirical_matrix, dsv.dsv, dsv.X, dsv.Y)

    print('Точечные оценки\n')
    mx_t = dsv.get_mx_theoretical()
    my_t = dsv.get_my_theoretical()
    mx_e = dsv.get_mx_empirical()
    my_e = dsv.get_my_empirical()
    print("Теоретическая M[x]:", mx_t)
    print("Теоретическая M[y]:", my_t)
    print("Эмпирическая M[x]:", mx_e)
    print("Эмпирическая M[x]:", my_e)

    dx_t = dsv.get_dx_theoretical(mx_t)
    dy_t = dsv.get_dy_theoretical(my_t)
    dx_e = dsv.get_dx_empirical(mx_e)
    dy_e = dsv.get_dy_empirical(my_e)
    print("Теоретическая D[x]:", dx_t)
    print("Теоретическая D[y]:", dy_t)
    print("Эмпирическая D[x]:", dx_e)
    print("Эмпирическая D[x]:", dy_e)

    print('Интервальные оценки\n')

    interval_estimations_mx = dsv.get_interval_estimations_mx(mx_e, dx_e)
    interval_estimations_my = dsv.get_interval_estimations_my(my_e, dy_e)
    print("interval_estimations_mx:", interval_estimations_mx)
    print("interval_estimations_my:", interval_estimations_my)
    interval_estimations_dx = dsv.get_interval_estimations_dx(dx_e)
    interval_estimations_dy = dsv.get_interval_estimations_dy(dy_e)
    print("interval_estimate_dx:", interval_estimations_dx)
    print("interval_estimate_dy:", interval_estimations_dy)

    correlation_coefficient = dsv.get_correlation_coefficient(mx_e, my_e)
    print("Коэффициент кореляции Пирсона:", correlation_coefficient)
    correlation_coefficient_t = dsv.get_correlation(dsv.XY, mx_t, my_t, dx_t, dy_t)
    correlation_coefficient_e = dsv.get_correlation(empirical_matrix, mx_e, my_e, dx_e, dy_e)
    print("Теоретический коэффициент кореляции:\n", correlation_coefficient_t)
    print("Эмперический коэффициент кореляции::\n", correlation_coefficient_e)

    print_result(interval_estimations_mx[0], interval_estimations_mx[1],  mx_e, "mx")
    print_result(interval_estimations_my[0], interval_estimations_my[1],  my_e, "my")
    print_result(interval_estimations_dx[0], interval_estimations_dx[1], dx_e, "dx")
    print_result(interval_estimations_dy[0], interval_estimations_dy[1], dy_e, "dy")