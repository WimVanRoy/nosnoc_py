import casadi as ca
import numpy as np
from nosnoc.rk_utils import generate_butcher_tableu_integral, IrkSchemes


class OptiDiscrete(ca.Opti):

    def __init__(self, *args, **kwargs):
        super(OptiDiscrete, self).__init__()
        self.discrete = []
        self.lbx = []
        self.ubx = []
        self.indices = {}
        self.M = 1e6

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

        # TODO: check size?
        self.lbx = ca.vertcat(self.lbx, lb)
        self.ubx = ca.vertcat(self.ubx, ub)
        return var

    def equal_if_on(self, trigger, equality, ns=1):
        """Big M formulation."""
        self.subject_to(equality >= -M*(1-trigger) * np.ones((ns, 1)))
        self.subject_to(equality <= M*(1-trigger) * np.ones((ns, 1)))


# Parameters
N_stages = 10
N_control_intervals = 1
N_finite_elements = 2
time_as_parameter = False
n_s = 2
tau = ca.collocation_points(n_s, "radau")

irk_scheme = IrkSchemes.RADAU_IIA
C_irk, D_irk, B_irk = ca.collocation_coeff(tau)

# Hystheresis parameters
v_levels = [5, 10, 15]

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
n_x = 3
# Binaries to represent the problem:
n_y = len(n)
Y0 = np.zeros(n_y)
u = ca.SX.sym('u')  # drive
n_u = 1
U0 = np.zeros(n_u)

lbu = np.array([-u_max])
ubu = np.array([u_max])
lbx = np.array([-ca.inf, 0, -ca.inf, -1, 0]).T
ubx = np.array([ca.inf, v_max, ca.inf, 2, ca.inf]).T

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
if time_as_parameter:
    T_final = opti.variable("T_final", 1, lb=1e-2, ub=1e2)
else:
    T_final = opti.parameter()

h = T_final/(N_stages*N_control_intervals*N_finite_elements)
# Cost: time only
J = T_final


Xk = opti.variable("Xk", n_x)
opti.subject_to(Xk == X0)
opti.set_initial(Xk, X0)


for k in range(N_stages):
    Yk = opti.variable("Yk", n_y, lb=0, ub=1, discrete=True)
    Lk = opti.variable("Lkm", n_y, lb=0, ub=1, discrete=True)
    opti.set_initial(Yk, Y0)
    if k == 0:
        opti.subject_to(Yk == Y0)
    else:
        opti.subject_to(ca.sum1(Yk) == 1)  # SOS1 constraint

    for i in range(N_control_intervals):
        Uk = opti.variable("U", n_u, lb=lbu, ub=ubu)
        opti.set_initial(Uk, U0)

        for j in range(N_finite_elements):
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
            # J = J + quad*B_irk*h;
            for ode_id, ode in enumerate(odes):
                opti.equal_if_on(Yk[ode_id], Pidot[:] - h * ode[:], n_x*n_s)

            Xk = opti.variable("Xk", n_x, lb=lbx, ub=ubx)
            opti.subject_to(Xk == Xk_end)


