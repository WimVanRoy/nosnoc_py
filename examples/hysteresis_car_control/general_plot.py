import matplotlib.pyplot as plt
import numpy as np
import pickle


def interp0(x, xp, yp):
    """Zeroth order hold interpolation w/ same
    (base)   signature  as numpy.interp."""

    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while k < len(xp) and x0 > xp[k]:
            k += 1
        return yp[k-1]

    if isinstance(x,float):
        return func(x)
    elif isinstance(x, list):
        return [func(x) for x in x]
    elif isinstance(x, np.ndarray):
        return np.asarray([func(x) for x in x])
    else:
        raise TypeError('argument must be float, list, or ndarray')


def plot(x_list, t_grid, u_list, t_grid_u):
    """Plot."""
    q = [x[0] for x in x_list]
    v = [x[1] for x in x_list]
    aux = [x[-2] + x[-3] for x in x_list]
    t = [x[-1] for x in x_list]
    u = [u[0] for u in u_list]
    plt.figure()
    plt.plot(t, q)
    v_aux = [max(q[i+1] - q[i] / max(1e-9, t[i+1] - t[i]), 0) for i in range(len(t)-1)]

    plt.figure()
    plt.plot(t[:-1], v_aux)
    plt.plot(t, v)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t, v)
    plt.xlabel("$t$")
    plt.ylabel("$v(t)$")
    plt.subplot(2, 2, 2)
    plt.plot(t, interp0(t_grid, t_grid_u, u))
    plt.xlabel("$t$")
    plt.ylabel("$u(t)$")
    plt.subplot(2, 2, 3)
    plt.plot(t, aux)
    plt.xlabel("$t$")
    plt.ylabel("$w_1(t) + w_2(t)$")
    plt.subplot(2, 2, 4)
    plt.plot(v, aux)
    plt.xlabel("$v$")
    plt.ylabel("$w_1(t) + w_2(t)$")
    plt.show()


with open("data_3d.pickle", "rb") as f:
    results = pickle.load(f)

plot(
    results["x_traj"], results["t_grid"],
    results["u_list"], results["t_grid_u"]
)
