"""Create MINLP for the hysteresis car problem."""

import casadi as ca
import numpy as np


class OptiDiscrete(ca.Opti):
    """Opti discrete."""

    def __init__(self, *args, **kwargs):
        """Create an extended opti class handling discrete variables."""
        super(OptiDiscrete, self).__init__()
        self.discrete = []
        self.lbx = []
        self.ubx = []
        self.indices = {}
        self.M = 1e6
        self.eps = 1e-9

    def variable(self, name, *args, lb=-ca.inf, ub=ca.inf, discrete=False):
        """Create variables."""
        var = ca.Opti.variable(self, *args)
        size = var.shape[0] * var.shape[1]
        if name not in self.indices:
            self.indices[name] = []

        self.indices[name].extend([
            len(self.discrete) + i for i in range(size)
        ])
        self.discrete.extend([int(discrete)] * size)

        if not isinstance(lb, ca.DM):
            lb = ca.DM(lb)
        if not isinstance(ub, ca.DM):
            ub = ca.DM(ub)

        if lb.shape != var.shape:
            raise Exception(f"{lb.shape=} not equal to {var.shape=} for {var=}")
        elif ub.shape != var.shape:
            raise Exception(f"{ub.shape=} not equal to {var.shape=} for {var=}")

        # TODO: check size?
        self.lbx = ca.vertcat(self.lbx, lb)
        self.ubx = ca.vertcat(self.ubx, ub)
        return var

    def equal_if_on(self, trigger, equality, ns=1, deviation=0):
        """Big M formulation."""
        self.subject_to(
            equality >= (-M*(1-trigger)+deviation) * np.ones((ns, 1))
        )
        self.subject_to(equality <= M*(1-trigger) * np.ones((ns, 1)))

    def higher_if_on(self, trigger, equation):
        """Trigger = 1 if equation > -eps."""
        self.subject_to(-(1-trigger) * M <= equation + self.eps)
        self.subject_to(equation + self.eps <= M * trigger)


# Parameters
N_stages = 10
N_control_intervals = 1
N_finite_elements = 2
time_as_parameter = False
n_s = 2
tau = ca.collocation_points(n_s, "radau")
C_irk, D_irk, B_irk = ca.collocation_coeff(tau)

# Hystheresis parameters
psi_on = [10, 15]
psi_off = [5, 10]

# Model parameters
q_goal = 150
v_goal = 0
v_max = 30
u_max = 5
M = 1e6

# fuel costs:
C = [1, 1.8, 2.5]
# ratios
n = [1, 2, 3]

# State variables:
q = ca.SX.sym("q")  # position
v = ca.SX.sym("v")  # velocity
L = ca.SX.sym("L")  # Fuel usage
X = ca.vertcat(q, v, L)
X0 = np.array([0, 0, 0]).T
lbx = np.array([0, 0, -ca.inf]).T
ubx = np.array([ca.inf, v_max, ca.inf]).T
q_goal = 100
v_goal = 0
n_x = 3
# Binaries to represent the problem:
n_y = len(n)
Y0 = np.zeros(n_y)
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
opti = OptiDiscrete()
# Time optimal control
if not time_as_parameter:
    T_final = opti.variable("T_final", 1, lb=1e-2, ub=1e2)
else:
    T_final = opti.parameter()

h = T_final/(N_stages*N_control_intervals*N_finite_elements)
# Cost: time only
J = T_final


Xk = opti.variable("Xk", n_x, lb=X0, ub=X0)
opti.set_initial(Xk, X0)


for k in range(N_stages):
    Yk = opti.variable("Yk", n_y, lb=[0]*n_y, ub=[1]*n_y, discrete=True)
    opti.set_initial(Yk, Y0)
    if k == 0:
        opti.subject_to(Yk == Y0)
    else:
        opti.subject_to(ca.sum1(Yk) == 1)  # SOS1 constraint
        # Transition condition
        LknUp = opti.variable(
            "LknUp", n_y-1, lb=[0]*(n_y-1), ub=[1]*(n_y-1), discrete=True
        )
        LknDown = opti.variable(
            "LknDown", n_y-1, lb=[0]*(n_y-1), ub=[1]*(n_y-1), discrete=True
        )
        # Transition
        LkUp = opti.variable(
            "LkUp", n_y, lb=[0]*n_y, ub=[1]*n_y, discrete=True)
        LkDown = opti.variable(
            "LkDown", n_y, lb=[0]*n_y, ub=[1]*n_y, discrete=True)
        psi = psi_fun(Xk)
        for i in range(n_y-1):
            # Only if trigger is ok, go up
            opti.subject_to(LkUp[i] <= LknUp[i])
            # Only go up if active
            opti.subject_to(LkUp[i] <= Yk[i])
            # Force going up if trigger = 1 and in right state!
            opti.subject_to(LkUp[i] >= LknUp[i] + Yk[i] - 1)
            # If psi > psi_on -> psi - psi_on >= 0 -> LknUp = 1
            opti.higher_if_on(LknUp[i], psi - psi_on[i])

        for i in range(1, n_y):
            opti.subject_to(LkDown[i-1] <= LknDown[i-1])
            opti.subject_to(LkDown[i-1] <= Yk[i])
            opti.subject_to(LkDown[i-1] >= LknDown[i-1] + Yk[i] - 1)
            opti.higher_if_on(LknDown[i-1], psi_off[i-1] - psi)

    for i in range(N_control_intervals):
        Uk = opti.variable("U", n_u, lb=lbu, ub=ubu)
        opti.set_initial(Uk, U0)

        for j in range(N_finite_elements):
            # n_s collocation points
            Xc = ca.horzcat(*[
                opti.variable("Xc", n_x, lb=lbx, ub=ubx)
                for i in range(n_s)
            ])
            opti.set_initial(Xc, ca.repmat(X0, 1, n_s))
            # Dynamics!
            odes = [
                f_dyn(Xc, Uk)
                for f_dyn in F_dyn
            ]

            # Use collocation
            Z = ca.horzcat(Xk, Xc)
            Pidot = Z @ C_irk
            Xk_end = Z @ D_irk

            # J = J + L*B_irk*h;
            for ode_id, ode in enumerate(odes):
                opti.equal_if_on(Yk[ode_id], Pidot[:] - h * ode[:], n_x*n_s)

            Xk = opti.variable("Xk", n_x, lb=lbx, ub=ubx)
            opti.subject_to(Xk == Xk_end)


# Terminal constraints:
opti.subject_to(Xk[0] == q_goal)
opti.subject_to(Xk[1] == v_goal)
J += Xk[2]
opti.minimize(J)

nlp = {'x': opti.x, 'f': opti.f, 'g': opti.g}
solver = ca.nlpsol("solver", "ipopt", nlp, {
    # "discrete": opti.discrete
})
solution = solver(
    x0=opti.x0,
    lbx=opti.lbx, ubx=opti.ubx,
    lbg=opti.lbg, ubg=opti.ubg
)
stats = solver.stats()
print(stats)
