"""Create MINLP for the hysteresis car problem."""

import casadi as ca
import numpy as np
from crc_algo.opt.description import Description
import matplotlib.pyplot as plt


class DescriptionExt(Description):
    """Extended description."""

    def __init__(self):
        """Create extended description with auxillary functions."""
        Description.__init__(self)
        self.M = 1e6
        self.eps = 1e-9

    def equal_if_on(self, trigger, equality, ns=1):
        """Big M formulation."""
        self.leq((-self.M*(1-trigger)) * np.ones((ns, 1)), equality)
        self.leq(equality, self.M*(1-trigger) * np.ones((ns, 1)))

    def higher_if_on(self, trigger, equation):
        """Trigger = 1 if equation > -eps."""
        self.leq(-(1-trigger) * self.M, equation + self.eps)
        self.leq(equation + self.eps, self.M * trigger)


def create_problem(time_as_parameter=False, use_big_M=False):
    """Create problen."""
    # Parameters
    N_stages = 10
    N_control_intervals = 1
    N_finite_elements = 1
    n_s = 4
    tau = ca.collocation_points(n_s, "radau")
    C_irk, D_irk, B_irk = ca.collocation_coeff(tau)

    # Hystheresis parameters
    psi_on = [10, 20]
    psi_off = [5, 15]

    # Model parameters
    q_goal = 150
    v_goal = 0
    v_max = 30
    u_max = 5

    # fuel costs:
    C = [1, 1.8, 2.5]
    # ratios
    n = [1, 2, 3]

    # State variables:
    q = ca.SX.sym("q")  # position
    v = ca.SX.sym("v")  # velocity
    L = ca.SX.sym("L")  # Fuel usage
    X = ca.vertcat(q, v, L)
    X0 = [0, 0, 0]
    lbx = [0, 0, -ca.inf]
    ubx = [ca.inf, v_max, ca.inf]
    q_goal = 100
    v_goal = 0
    n_x = 3
    # Binaries to represent the problem:
    n_y = len(n)
    Y0 = np.zeros(n_y)
    Y0[0] = 1
    u = ca.SX.sym('u')  # drive
    n_u = 1
    U0 = np.zeros(n_u)

    lbu = np.array([-u_max])
    ubu = np.array([u_max])

    # x = q v L
    X = ca.vertcat(q, v, L)

    F_dyn = [
        ca.Function(f'f_dyn_{i}', [X, u], [ca.vertcat(
            v, n[i]*u, C[i]
        )]) for i in range(len(n))
    ]

    psi = v
    psi_fun = ca.Function('psi', [X], [psi])

    # Create problem:
    opti = DescriptionExt()
    # Time optimal control
    if not time_as_parameter:
        T_final = opti.sym("T_final", 1, lb=1e-2, ub=1e2, x0=15)
    else:
        T_final = opti.add_parameters("T_final", 1, values=15)

    h = T_final/(N_stages*N_control_intervals*N_finite_elements)
    # Cost: time only
    J = T_final

    Xk = opti.sym("Xk", n_x, lb=X0, ub=X0, x0=X0)
    for k in range(N_stages):
        Yk = opti.sym_bool("Yk", n_y)
        if k == 0:
            opti.eq(Yk[1], Y0[1])
            opti.eq(Yk[0], Y0[0])
        else:
            eps = 0.01
            opti.add_g(1 - eps, ca.sum1(Yk), 1 + eps)  # SOS1
            # Transition condition
            LknUp = opti.sym_bool("LknUp", n_y-1)
            LknDown = opti.sym_bool("LknDown", n_y-1)
            # Transition
            LkUp = opti.sym_bool("LkUp", n_y-1)
            LkDown = opti.sym_bool("LkDown", n_y-1)

            psi = psi_fun(Xk)
            for i in range(n_y-1):
                # Trigger
                # If psi > psi_on -> psi - psi_on >= 0 -> LknUp = 1
                opti.higher_if_on(LknUp[i], psi - psi_on[i])
                opti.higher_if_on(LknDown[i], psi_off[i] - psi)
                # Only if trigger is ok, go up
                opti.leq(LkUp[i], LknUp[i])
                opti.leq(LkDown[i], LknDown[i])
                # Only go up if active
                opti.leq(LkUp[i], Yk[i])
                opti.leq(LkDown[i], Yk[i+1])
                # Force going up if trigger = 1 and in right state!
                opti.leq(LknUp[i] + Yk[i] - 1, LkUp[i])
                opti.leq(LknDown[i] + Yk[i + 1] - 1, LkDown[i])

        for i in range(N_control_intervals):
            Uk = opti.sym("U", n_u, lb=lbu, ub=ubu, x0=U0)

            for j in range(N_finite_elements):
                # n_s collocation points
                Xc = ca.horzcat(*[
                    opti.sym("Xc", n_x, lb=lbx, ub=ubx, x0=X0)
                    for i in range(n_s)
                ])

                # Use collocation
                Z = ca.horzcat(Xk, Xc)
                Pidot = Z @ C_irk
                if use_big_M:
                    for i in range(n_s):
                        for ii in range(n_y):
                            opti.equal_if_on(
                                Yk[ii],
                                Pidot[:, i] - h * F_dyn[ii](Xc[i], Uk),
                                n_x
                            )
                else:
                    for i in range(n_s):
                        eq = 0
                        for ii in range(n_y):
                            eq += Yk[ii] * F_dyn[ii](Z[:, i], Uk)

                        opti.eq(Pidot[:, i], h * eq)

                Xk_end = Z @ D_irk

                # # J = J + L*B_irk*h;

                Xk = opti.sym("Xk", n_x, lb=lbx, ub=ubx, x0=X0)
                opti.eq(Xk, Xk_end)

    # Terminal constraints:
    slack1 = opti.sym('slack', 1, lb=0, ub=1)
    slack2 = opti.sym('slack', 1, lb=0, ub=1)
    opti.leq(Xk[0] - q_goal, slack1)
    opti.leq(q_goal - Xk[0], slack1)
    opti.leq(Xk[1] - v_goal, slack2)
    opti.leq(v_goal - Xk[1], slack2)
    J += Xk[2]
    opti.f = J + slack1 + slack2
    return opti


def run_bonmin():
    """Run bonmin."""
    opti = create_problem()
    opti.set_solver(ca.nlpsol, 'bonmin', is_discrete=True)
    opti.solve()
    return opti


def run_ipopt():
    """Run ipopt."""
    opti = create_problem()
    opti.set_solver(ca.nlpsol, 'ipopt', is_discrete=False)
    opti.solve()
    return opti


def run_gurobi():
    """Run gurobi."""
    opti = create_problem(time_as_parameter=True, use_big_M=True)
    opti.set_solver(ca.qpsol, "gurobi", is_discrete=True, options={"error_on_fail": False})
    T_max = 20
    T_min = 1e-2
    lb_k = [T_min]
    ub_k = [T_max]
    tolerance = 1e-2
    while ub_k[-1] - lb_k[-1] > tolerance:
        T_new = (ub_k[-1] + lb_k[-1]) / 2
        opti.set_parameters("T_final", T_new)
        opti.solve()
        if opti.is_solved():
            print(f"SUCCES {T_new=}")
            # Success
            ub_k.append(T_new)
        else:
            print(f"INF {T_new=}")
            lb_k.append(T_new)

    return opti


if __name__ == "__main__":
    opti = run_bonmin()
    print(opti.get("T_final"))
    Xk = np.array(opti.get("Xk"))
    plt.plot(Xk)
    plt.show()
    print(opti.get("Yk"))
