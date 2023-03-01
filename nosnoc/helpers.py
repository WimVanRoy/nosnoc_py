from typing import Optional

import numpy as np
from .solver import NosnocSolver


class NosnocSimLooper:

    def __init__(self,
                 solver: NosnocSolver,
                 x0: np.ndarray,
                 Nsim: int,
                 p_values: Optional[np.ndarray] = None,
                 w_init: Optional[list] = None):
        """
        :param solver: NosnocSolver to be called in a loop
        :param x0: np.ndarray: initial state
        :param Nsim: int: number of simulation steps
        :param p_values: Optional np.ndarray of shape (Nsim, n_p_glob), parameter values p_glob are updated at each simulation step accordingly.
        :param w_init: Optional: a list of np.ndarray with w values to initialize the solver at each step.
        """
        # check that NosnocSolver solves a pure simulation problem.
        if not solver.problem.is_sim_problem():
            raise Exception("NosnocSimLooper can only be used with pure simulation problem")

        # p values
        self.p_values = p_values
        if self.p_values is not None:
            if self.p_values.shape != (Nsim, solver.problem.model.dims.n_p_glob):
                raise ValueError("p_values should have shape (Nsim, n_p_glob). "
                                 f"Expected ({Nsim}, {solver.problem.model.dims.n_p_glob}), got {self.p_values.shape}")

        # create
        self.solver: NosnocSolver = solver
        self.Nsim = Nsim

        self.xcurrent = x0
        self.X_sim = [x0]
        self.time_steps = np.array([])
        self.theta_sim = []
        self.lambda_sim = []
        self.alpha_sim = []
        self.w_sim = []
        self.w_all = []
        self.cost_vals = []
        self.w_init = w_init

        self.cpu_nlp = np.zeros((Nsim, solver.opts.max_iter_homotopy + 1))

    def run(self, stop_on_failure=False) -> None:
        """Run the simulation loop."""
        for i in range(self.Nsim):
            # set values
            self.solver.set("x0", self.xcurrent)
            if self.w_init is not None:
                self.solver.set("w", self.w_init[i])
            if self.p_values is not None:
                self.solver.set("p_global", self.p_values[i, :])
            # solve
            results = self.solver.solve()
            # collect
            self.X_sim += results["x_list"]
            self.xcurrent = self.X_sim[-1]
            self.cpu_nlp[i, :] = results["cpu_time_nlp"]
            self.time_steps = np.concatenate((self.time_steps, results["time_steps"]))
            self.theta_sim.append(results["theta_list"])
            self.lambda_sim.append(results["lambda_list"])
            self.alpha_sim.append(results["alpha_list"])
            self.w_sim += [results["w_sol"]]
            self.w_all += [results["w_all"]]
            self.cost_vals.append(results["cost_val"])

            if (stop_on_failure and results["status"] == "Infeasible_Problem_Detected"):
                return False

        return True

    def get_results(self) -> dict:
        self.t_grid = np.concatenate((np.array([0.0]), np.cumsum(self.time_steps)))
        results = {
            "X_sim": np.array(self.X_sim),
            "cpu_nlp": np.nan_to_num(self.cpu_nlp),
            "time_steps": self.time_steps,
            "t_grid": self.t_grid,
            "theta_sim": self.theta_sim,
            "lambda_sim": self.lambda_sim,
            "w_sim": self.w_sim,
            "w_all": self.w_all,
            "cost_vals": self.cost_vals
        }
        return results
