import cvxpy as cvx
import numpy as np

from controllers.controller_base import (PredictiveController, build_required_decor, ControllerBuildRequiredError,
                                         ControllerSolverError)
from controllers.components.objective_atoms import ObjectiveAtoms
from controllers.components.variables import EvoVariables
from controllers.components.mld_evolution_matrices import MldEvoMatrices
from utils.matrix_utils import matmul
from utils.helper_funcs import eq_all_val

from utils.func_utils import ParNotSet
from utils.versioning import versioned

import io
import contextlib


@versioned(versioned_sub_objects=('variables', 'mld_evo_matrices', 'solve_version'))
class MpcController(PredictiveController):
    _repr_components = ['N_p', 'N_tilde', 'std_obj_atoms', 'variables', 'mld_numeric_k']

    @build_required_decor
    def reset_components(self, x_k=None, omega_tilde_k=None):
        super(MpcController, self).reset_components(x_k=x_k, omega_tilde_k=omega_tilde_k)
        self._std_objective = 0
        self._other_objectives = []
        self._objective = 0

    @property
    def std_obj_atoms(self):
        return self._std_obj_atoms

    @property
    def objective(self):
        return self._objective

    @build_required_decor
    def set_std_obj_atoms(self, objective_atoms_struct=None, **kwargs):
        if self._std_obj_atoms:
            self._std_obj_atoms.set(objective_atoms_struct=objective_atoms_struct, **kwargs)
        else:
            self._std_obj_atoms = ObjectiveAtoms(self, objective_atoms_struct=objective_atoms_struct,
                                                 **kwargs)

    @build_required_decor
    def update_std_obj_atoms(self, objective_weights_struct=None, **kwargs):
        if self._std_obj_atoms:
            self._std_obj_atoms.update(objective_weights_struct, **kwargs)
        else:
            self._std_obj_atoms = ObjectiveAtoms(self, objective_atoms_struct=objective_weights_struct,
                                                 **kwargs)

    def gen_std_objective(self, objective_weights: ObjectiveAtoms, variables: EvoVariables):
        try:
            return objective_weights.gen_cost(variables)
        except AttributeError:
            if objective_weights is None:
                return 0
            else:
                raise TypeError(f"objective_weights must be of type {ObjectiveAtoms.__class__.__name__}")

    @build_required_decor
    def set_objective(self, std_objective=ParNotSet, other_objectives=ParNotSet):
        if std_objective is not ParNotSet:
            self._std_objective = std_objective if std_objective is not None else (
                self.gen_std_objective(self.std_obj_atoms, self.variables))
        if other_objectives is not ParNotSet:
            self._other_objectives = other_objectives if other_objectives is not None else []


        self._objective = self._std_objective + cvx.sum(self._other_objectives)

        self.set_build_required()

    @build_required_decor(set=False)
    def build(self, with_std_objective=True, with_std_constraints=True, sense=None, disable_soft_constraints=False):
        self._mld_evo_matrices.update()

        if with_std_objective:
            self.set_objective(std_objective=None)
        else:
            self.set_objective(std_objective=0)

        if with_std_constraints:
            std_evo_constraints = None
        else:
            std_evo_constraints = []

        self.set_constraints(std_evo_constaints=std_evo_constraints, disable_soft_constraints=disable_soft_constraints)
        assert isinstance(self._constraints, list)

        sense = 'minimize' if sense is None else sense
        if sense.lower().startswith('min'):
            self._problem = cvx.Problem(cvx.Minimize(self._objective), self._constraints)
        elif sense.lower().startswith('max'):
            self._problem = cvx.Problem(cvx.Maximize(self._objective), self._constraints)
        else:
            raise ValueError(f"Problem 'sense' must be either 'minimize' or 'maximize', got '{sense}'.")

        self.update_stored_version()