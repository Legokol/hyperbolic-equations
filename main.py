import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Не самый общий метод: здесь f полагается постоянной, а порядок системы 2x2 фиксирован
def cir_method(A, f, u0, T, L, tau, h):
    a, s = np.linalg.eig(A)
    s = s.T
    s_inv = np.linalg.inv(s)
    w1 = np.zeros((int(T / tau + 1), int(L / h + 1)))
    w2 = np.zeros(w1.shape)

    u = np.zeros(w1.shape)
    v = np.zeros(w1.shape)

    for i in range(w1.shape[1]):  # заполняем начальные условия
        u[0][i] = u0(h * i)[0]
        v[0][i] = u0(h * i)[1]

        w1[0][i] = s[0][0] * u0(h * i)[0] + s[0][1] * u0(h * i)[1]
        w2[0][i] = s[1][0] * u0(h * i)[0] + s[1][1] * u0(h * i)[1]
    for i in range(w1.shape[0] - 1):
        if i + 1 >= w1.shape[1] - i - 2:  # решаем только в треугольнике, где решение существует и единственно
            break
        for j in range(i + 1, w1.shape[1] - 1 - i):
            w1[i + 1][j] = tau * (s[0][0] * f[0] + s[0][1] * f[1]) + w1[i][j] - tau / h / 2 * (
                    (a[0] + np.abs(a[0])) * (w1[i][j] - w1[i][j - 1]) + (a[0] - np.abs(a[0])) * (
                    w1[i][j + 1] - w1[i][j]))
            w2[i + 1][j] = tau * (s[1][0] * f[0] + s[1][1] * f[1]) + w2[i][j] - tau / h / 2 * (
                    (a[1] + np.abs(a[1])) * (w2[i][j] - w2[i][j - 1]) + (a[1] - np.abs(a[1])) * (
                    w2[i][j + 1] - w2[i][j]))

            u[i + 1][j] = s_inv[0][0] * w1[i + 1][j] + s_inv[0][1] * w2[i + 1][j]
            v[i + 1][j] = s_inv[1][0] * w1[i + 1][j] + s_inv[1][1] * w2[i + 1][j]

    return u, v


def u0(x):
    return np.array([np.sin(x), np.cos(x)])


A = np.array([[0, 1], [1, 0]])

T = 10
L = np.pi
h = 0.01
tau = h

x = np.linspace(0, L, int(L / h + 1))
t = np.linspace(0, L / 2, int(L / h + 1) // 2)
u, v = cir_method(A, np.array([0, 0]), u0, T, L, tau, h)

fig, ax = plt.subplots()
ax.set_ylim([-.01, .01])
line, = ax.plot(x, v[0])

"""
def animate(i):
    line.set_ydata(t[i] + np.sin(x) * (np.cos(t[i]) + np.sin(t[i])))
    return line,

def animate(i):
    line.set_ydata(np.cos(x) * (np.cos(t[i]) - np.sin(t[i])))
    return line,

def animate(i):
    line.set_ydata(v[i])
    return line,

def animate(i):
    line.set_ydata(v[i] - (np.cos(x) * (np.cos(tau * i) - np.sin(tau * i))))
    return line,
"""

# anim = animation.FuncAnimation(fig, animate, interval=50, repeat=False, frames=t.size)
# anim.save('v_diff.gif')
plt.show()
