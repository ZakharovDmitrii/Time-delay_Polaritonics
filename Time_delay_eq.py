import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.special import hankel1
import time
import random
from scipy.fft import fft, fftfreq

time0 = time.time() # время начала выполнения программы

# Константы
d_list = np.linspace(10,20,11) # диапазон для d

Gam_A = 0.05
R = 5 * (10**(-3)) # 10^(-3) добавляется так как величина дается в микроЭВ, а все энергетические константы берем в милиЭВ
P = 100
J0 = 1.1
k0 = 1.7 + 0.014j
W = 1.22 - 0.5j
alpha = 0.1 * (10**(-3))
g = 0.1 * (10**(-3))
v = 1.9
beta = -1
exp_b = np.exp(beta*1j)


def J(d):   # Функция-коэффициент при слагаемом с запаздыванием
    return (J0 * hankel1(0,k0*d) * np.conjugate(hankel1(0,k0*d))).real # дополнительно берется реальная часть, так как мнимая 10**(-18)


for d in d_list: # перебираем возможные d

    def equation(Y, t): # функция определяютщая вид системы уравнений
        c1, c1i, c2, c2i, n1, n2 = Y(t) # вектор, который ходим найти (с1.real, c1.imag, c2.real, c2.imag, n1, n2)
        c1_d, c1i_d, c2_d, c2i_d, n1_d, n2_d = Y(t - d / v) # вектор соответсвующий запаздыванию
        return [c1i * (W.real + g * n1 + alpha * (c1 ** 2 + c1i ** 2)) + c1 * (W.imag + n1 * R / 2) + J(d) * (exp_b.real * c2i_d + exp_b.imag * c2_d), # уравнение на с1.real
                -c1 * (W.real + g * n1 + alpha * (c1 ** 2 + c1i ** 2)) + c1i * (W.imag + n1 * R / 2) - J(d) * (exp_b.real * c2_d - exp_b.imag * c2i_d), # уравнение на с1.imag
                c2i * (W.real + g * n2 + alpha * (c2 ** 2 + c2i ** 2)) + c2 * (W.imag + n2 * R / 2) + J(d) * (exp_b.real * c1i_d + exp_b.imag * c1_d), # уравнение на с2.real
                -c2 * (W.real + g * n2 + alpha * (c2 ** 2 + c2i ** 2)) + c2i * (W.imag + n2 * R / 2) - J(d) * (exp_b.real * c1_d - exp_b.imag * c1i_d), # уравнение на с2.imag
                - (Gam_A + R * (c1 ** 2 + c1i ** 2)) * n1 + P, # уравнение на n1
                - (Gam_A + R * (c2 ** 2 + c2i ** 2)) * n2 + P  # уравнение на n2
                ]


    initial_list = np.zeros(6, dtype=float) # определяем вектор начальных условий
    for i in range(len(initial_list)):
        initial_list[i] = random.uniform(0, 1) # компоненты вектора - слуйчаные числа от 0 до 1


    def initial_history_func(t):  # вводим функцию начальных условией, нужна, чтобы решить систему
        return [initial_list[0], initial_list[1], initial_list[2], initial_list[3], initial_list[4], initial_list[5]]

    ts = np.linspace(0, 100, 1000) # массив значений аргумента

    ys = ddeint(equation, initial_history_func, ts) # решение системы уравнений в виде массива компонент вектора Y в разные моменты времени - матрица (1000,6)

    plt.rcParams['font.size'] = 8    # график для с1.real и c1.imag
    fig, axs = plt.subplots()
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
    axs.plot(ts, ys[:,0], color='red', linewidth=1, label = 'Real c1')
    axs.plot(ts, ys[:,1], color='orange', linewidth=1, label = 'Im c1')
    axs.legend()
    name = 'c1 для d=' + str(d) + '.pdf'
    fig.savefig(name, dpi = 500) # сохраняет в формате pdf в ту же папку, где находится программа

    plt.rcParams['font.size'] = 8 # график для с2.real и c2.imag
    fig, axs = plt.subplots()
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
    axs.plot(ts, ys[:,2], color='green', linewidth=1, label = 'Real c2')
    axs.plot(ts, ys[:,3], color='olive', linewidth=1, label = 'Im c2')
    axs.legend()
    name = 'c2 для d=' + str(d) + '.pdf'
    fig.savefig(name, dpi = 500)

    plt.rcParams['font.size'] = 8 # график для n1 и n2
    fig, axs = plt.subplots()
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
    axs.plot(ts, ys[:,4], color='purple', linewidth=1, label = 'n1')
    axs.plot(ts, ys[:,5], color='blue', linewidth=1, label = 'n2')
    axs.legend()
    name = 'n для d='+ str(d)  +'.pdf'
    fig.savefig(name, dpi = 500)

    c1_list = np.zeros_like(ys[:,0], dtype='complex') # массив с1.real + i * c1.imag
    c2_list = np.zeros_like(ys[:,1], dtype='complex') # массив с2.real + i * c2.imag

    for i in range(len(c1_list)):
        c1_list[i] = ys[:,0][i] + ys[:,1][i]*1j
        c2_list[i] = ys[:, 2][i] + ys[:, 3][i]*1j

    # Пробное фурье

    N = 1000 # число точек
    dt = np.diff(ts)[0]  # интервал по времени из массива в строке 53
    f1 = 20 / (N * dt) # пробные частоты для фурье
    f2 = 10 / (N * dt)
    y = np.sin(2*np.pi*f1*ts) + 0.3*np.sin(2*np.pi*f2*ts) # функция сигнала
    f = fftfreq(len(ts), np.diff(ts)[0]) # диапозон частот
    y_FFT = fft(y)  # Фурье сигнала

    fig, axs = plt.subplots()
    plt.plot(f, np.abs(y_FFT)) # график Фурье
    plt.axvline(x=f1, color='black', linewidth=0.8 ,linestyle='--' ) # частоты, которые хотим получить
    plt.axvline(x=f2, color='black', linewidth=0.8 , linestyle='--')
    plt.xlabel('$f_n$ [$s^{-1}$]', fontsize=20)
    plt.ylabel('|$\hat{x}_n$|', fontsize=20)
    plt.xlim(-0.5, 0.5)
    fig.savefig('four_test.pdf', dpi=500)


    c1_four = fft(c1_list) # Фурье для с1
    c2_four = fft(c2_list) # Фурье для с2

    c1_max = max(c1_four) # пик Фурье для с1
    c1_max_index = np.argmax(c1_four) # индекс в списке пика для с1
    c2_max = max(c2_four) # пик Фурье для с2
    c2_max_index = np.argmax(c2_four) # индекс в списке пика для с2

    f = fftfreq(len(ts), np.diff(ts)[0]) # диапазон частот
    freq_max_c1 = f[c1_max_index]
    freq_max_c2 = f[c2_max_index]
    fig, axs = plt.subplots()
    plt.plot(f, np.abs(c1_four))
    plt.plot(f, np.abs(c2_four))
    plt.xlim(-1, 1)
    plt.grid()
    fig.savefig('c1с2_four для d' + str(d) + '.pdf', dpi=500) # сохраняет график Фурье для с1 и с2


    print('d=' + str(d), time.time() - time0) # выводит время счета для одного d (решение системы на с и Фурье)
    print(freq_max_c1,freq_max_c2)