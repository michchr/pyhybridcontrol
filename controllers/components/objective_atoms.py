import re as regex
from abc import abstractmethod

import cvxpy as cvx
import numpy as np

from controllers.components.component_base import ComponentBase
from controllers.components.variables import EvoVariables, EvoVariableStruct
from controllers.controller_utils import process_base_args

from models.mld_model import MldInfo, MldModel
from utils.decorator_utils import process_method_args_decor
from utils.func_utils import ParNotSet
from utils.matrix_utils import atleast_2d_col, block_diag_dense, matmul
from utils.versioning import increments_version_decor

from structdict import named_struct_dict

_objective_weight_types = ['vector', 'matrix']


class ObjectiveWeightBase(ComponentBase):
    _field_names = ['weight_N_tilde', 'weight_N_p', 'weight_f']
    _field_names_set = frozenset(_field_names)
    _weight_type = 'base'

    _var_names_set = frozenset(MldInfo._var_names)
    __internal_names = ['_var_name']

    def __init__(self, var_name, weight_N_tilde=None, weight_N_p=None, weight_f=None, controller=None,
                 N_p=ParNotSet, N_tilde=ParNotSet, mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet):

        super(ObjectiveWeightBase, self).__init__(controller, N_p=N_p, N_tilde=N_tilde,
                                                  mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)

        if var_name not in self._var_names_set:
            raise ValueError(f"var_name must be in {set(self._var_names_set)} currently, '{var_name}'")
        else:
            self._var_name = var_name

        self.update(weight_N_tilde=weight_N_tilde, weight_N_p=weight_N_p, weight_f=weight_f)

    @property
    def weight_type(self):
        return self._weight_type

    @property
    def var_name(self):
        return self._var_name

    def is_zero(self):
        if self.weight_N_tilde is None or np.all(np.isclose(self.weight_N_tilde,0.0)):
            return True
        else:
            return False

    @increments_version_decor
    def update(self, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        var_dim = self.mld_info_k.get_var_dim(self._var_name)
        self._set_weight(var_dim, weight_N_tilde=weight_N_tilde, weight_N_p=weight_N_p, weight_f=weight_f)
        self._update_set_with(self.N_p, self.N_tilde)

    def set(self, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        self.clear()
        self.update(weight_N_tilde=weight_N_tilde, weight_N_p=weight_N_p, weight_f=weight_f)

    def _reset(self):
        super(ObjectiveWeightBase, self)._reset()
        self._base_dict_init(dict.fromkeys(self._field_names))

    @abstractmethod
    def _set_weight(self, var_dim, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        pass


class VectorWeight(ObjectiveWeightBase):
    _weight_type = 'vector'

    def _set_weight(self, var_dim, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        weight = self['weight_N_tilde']

        N_tilde = self.N_tilde
        N_p = self.N_p
        required_weight_shape = (N_tilde * var_dim, 1)

        if weight is None or weight.shape != required_weight_shape:
            weight = np.zeros(required_weight_shape)

        if weight_N_tilde is not None:
            weight[:] = self._process_vector_weight(var_dim=var_dim,
                                                    value=weight_N_tilde, length=N_tilde,
                                                    length_name='N_tilde', is_terminal=False)

        if weight_N_p is not None:
            if N_p > N_tilde:
                raise ValueError("Cannot set weight_N_p if N_tilde < N_p")
            else:
                weight[:N_p * var_dim, :1] = (
                    self._process_vector_weight(var_dim=var_dim, value=weight_N_p, length=N_p, length_name='N_p',
                                                is_terminal=False))

        if weight_f is not None:
            weight[-var_dim:, :1] = self._process_vector_weight(var_dim=var_dim,
                                                                value=weight_f, length=1,
                                                                length_name='1', is_terminal=True)

        if N_p > N_tilde:
            weight_N_p = None
        else:
            weight_N_p = weight[:N_p * var_dim, :1]

        weight_f = weight[-var_dim:, :1]

        self._base_dict_update(weight_N_tilde=weight,
                               weight_N_p=weight_N_p,
                               weight_f=weight_f)

    def _process_vector_weight(self, var_dim, value, length, length_name, is_terminal=False):
        value = atleast_2d_col(value)
        value_shape = value.shape
        if value_shape[1] != 1:
            raise ValueError(f"Column dim of {self._weight_type} weight for opt_var: '{self._var_name}', must be 1.")
        elif is_terminal:
            if value_shape[0] == var_dim:
                return value
            else:
                raise ValueError(
                    f"Row dim of {self._weight_type} terminal weight for opt_var: '{self._var_name}' must "
                    f"be in {{{var_dim}}}")
        elif value_shape[0] == (var_dim * length):
            return value
        elif value_shape[0] == var_dim:
            return np.tile(value, (length, 1))
        else:
            raise ValueError(
                f"Row dim of {self._weight_type} weight for opt_var: '{self._var_name}', must be in "
                f"{{{var_dim}, {var_dim}*{length_name}}}")


class MatrixWeight(ObjectiveWeightBase):
    _weight_type = 'matrix'

    def _set_weight(self, var_dim, weight_N_tilde=None, weight_N_p=None, weight_f=None):
        weight = self['weight_N_tilde']

        N_tilde = self.N_tilde
        N_p = self.N_p
        required_weight_shape = (N_tilde * var_dim, N_tilde * var_dim)

        if weight is None or weight.shape != required_weight_shape:
            weight = np.zeros(required_weight_shape)

        if weight_N_tilde is not None:
            weight[:] = self._process_matrix_weight(var_dim=var_dim,
                                                    value=weight_N_tilde, length=N_tilde,
                                                    length_name='N_tilde', is_terminal=False)
        if weight_N_p is not None:
            weight[:N_p * var_dim, :N_p * var_dim] = (
                self._process_matrix_weight(var_dim=var_dim, value=weight_N_p, length=N_p,
                                            length_name='N_p', is_terminal=False))

            if N_p > N_tilde:
                raise ValueError("Cannot set weight_N_p if N_tilde < N_p")
            else:
                weight[:N_p * var_dim, :N_p * var_dim] = (
                    self._process_matrix_weight(var_dim=var_dim, value=weight_N_p, length=N_p, length_name='N_p',
                                                is_terminal=False))

        if weight_f is not None:
            weight[-var_dim:, -var_dim:] = (
                self._process_matrix_weight(var_dim=var_dim, value=weight_f, length=1, length_name='1',
                                            is_terminal=True))

        if N_p > N_tilde:
            weight_N_p = None
        else:
            weight_N_p = weight[:N_p * var_dim, :N_p * var_dim]

        weight_f = weight[-var_dim:, -var_dim:]

        self._base_dict_update(weight_N_tilde=weight,
                               weight_N_p=weight_N_p,
                               weight_f=weight_f)

    def _process_matrix_weight(self, var_dim, value, length, length_name, is_terminal=False):
        value = atleast_2d_col(value)
        value_shape = value.shape
        if value_shape[0] != value_shape[1]:
            raise ValueError(
                f"{self._weight_type} weight for opt_var: '{self._var_name}', must be square. Currently has shape: "
                f"{value_shape}")
        elif is_terminal:
            if value_shape[0] == var_dim:
                return value
            else:
                raise ValueError(
                    f"Row dim of {self._weight_type} terminal weight for opt_var: '{self._var_name}' must "
                    f"be in {{{var_dim}}}")
        elif value_shape[0] == (var_dim * length):
            return value
        elif value_shape[0] == var_dim:
            return block_diag_dense([value] * length)
        else:
            raise ValueError(
                f"Row dim of {self._weight_type} weight for opt_var: '{self._var_name}', must be in "
                f"{{{var_dim}, {var_dim}*{length_name}}}")


_objective_atom_types = ['linear', 'quadratic', 'L1', 'L2', 'Linf']


class ObjectiveAtomBase(ComponentBase):
    _field_names = ['weight', 'is_rate_atom', 'cost', 'variable']
    _field_names_set = frozenset(_field_names)
    _atom_type = 'base'

    _var_names_set = frozenset(MldInfo._var_names)
    __internal_names = ['_var_name']

    def __init__(self, var_name, weight=None, is_rate_atom=False,
                 controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet):
        super(ObjectiveAtomBase, self).__init__(controller, N_p=N_p, N_tilde=N_tilde,
                                                mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)

        if var_name not in self._var_names_set:
            raise ValueError(f"var_name must be in {set(self._var_names_set)} currently, '{var_name}'")
        else:
            self._var_name = var_name

        self.is_rate_atom: bool = is_rate_atom
        self.weight: ObjectiveWeightBase = None
        self.cost = 0
        if weight is not None:
            self.update(weight=weight)

    @property
    def var_name(self):
        return self._var_name

    def update(self, weight=ParNotSet, is_rate_atom=None):
        if weight is not ParNotSet:
            if isinstance(weight, ObjectiveWeightBase):
                if weight.var_name == self._var_name:
                    self.weight = weight
                else:
                    raise ValueError(f"weight var_name:{weight.var_name} does not match self.var_name:{self.var_name}")
            elif weight is None:
                self.weight = None
            else:
                raise TypeError(
                    f"Invalid type:{type(weight)} for weight, must be None or "
                    f"subclass of {ObjectiveWeightBase.__name__}")

        self.is_rate_atom = is_rate_atom if is_rate_atom is not None else self.is_rate_atom

    @process_method_args_decor(process_base_args)
    def apply_atom(self, variable: EvoVariableStruct, N_p=None, N_tilde=None,
                   mld_numeric_k: MldModel = None, mld_numeric_tilde=None,
                   mld_info_k: MldInfo = None):

        mld_info_k = mld_info_k or self.mld_info_k
        var_dim = mld_info_k.get_var_dim(self._var_name)
        weight = self.weight.weight_N_tilde if self.weight is not None else self.weight

        if var_dim:
            if self.is_rate_atom:
                variable = self._gen_rate_variable_from_variable(variable=variable, var_dim=var_dim, N_p=N_p,
                                                                 N_tilde=N_tilde)

            if isinstance(self.weight, VectorWeight):
                cost = self.apply_atom_with_vector_weight(variable=variable, weight=weight, var_dim=var_dim)
            elif isinstance(self.weight, MatrixWeight):
                cost = self.apply_atom_with_matrix_weight(variable=variable, weight=weight, var_dim=var_dim)
            else:
                # todo ensure this is correct
                cost = self.apply_atom_with_unity_weight(variable=variable, weight=weight, var_dim=var_dim)
        else:
            cost = 0

        self.variable = variable
        self.cost = cost
        return cost

    @abstractmethod
    def apply_atom_with_unity_weight(self, variable, weight, var_dim=None):
        pass

    @abstractmethod
    def apply_atom_with_vector_weight(self, variable, weight, var_dim=None):
        pass

    @abstractmethod
    def apply_atom_with_matrix_weight(self, variable, weight, var_dim=None):
        pass

    @staticmethod
    def _gen_rate_variable_from_variable(variable: EvoVariableStruct, var_dim, N_p, N_tilde):
        d_variable = EvoVariableStruct()
        d_var_N_tilde = variable.var_N_tilde - cvx.vstack(
            [variable.var_k_neg1, variable.var_N_tilde[:-var_dim]])  # v(k)-v(k-1)
        EvoVariables._set_var_using_var_N_tilde(variable=d_variable, var_dim=var_dim, var_N_tilde=d_var_N_tilde,
                                                N_p=N_p, N_tilde=N_tilde)

        return d_variable


class LinearObjectiveAtom(ObjectiveAtomBase):
    _atom_type = 'linear'

    def apply_atom_with_unity_weight(self, variable, weight, var_dim=None):
        return cvx.sum(variable.var_N_tilde)

    def apply_atom_with_vector_weight(self, variable, weight, var_dim=None):
        return matmul(weight.T, variable.var_N_tilde[:weight.shape[0]])

    def apply_atom_with_matrix_weight(self, variable, weight, var_dim=None):
        return cvx.sum(matmul(weight, variable.var_N_tilde[:weight.shape[1]]))


class QuadraticObjectiveAtom(ObjectiveAtomBase):
    _atom_type = 'quadratic'

    def apply_atom_with_unity_weight(self, variable, weight, var_dim=None):
        return cvx.sum_squares(variable.var_N_tilde)

    def apply_atom_with_vector_weight(self, variable, weight, var_dim=None):
        return cvx.sum_squares(cvx.multiply(weight, variable.var_N_tilde[:weight.shape[0]]))

    def apply_atom_with_matrix_weight(self, variable, weight, var_dim=None):
        return cvx.quad_form(variable.var_N_tilde[:weight.shape[1]], weight)


class L1ObjectiveAtom(ObjectiveAtomBase):
    _atom_type = 'L1'

    def apply_atom_with_unity_weight(self, variable, weight, var_dim=None):
        return cvx.norm1(variable.var_N_tilde)

    def apply_atom_with_vector_weight(self, variable, weight, var_dim=None):
        return cvx.norm1(cvx.multiply(weight, variable.var_N_tilde[:weight.shape[0]]))

    def apply_atom_with_matrix_weight(self, variable, weight, var_dim=None):
        return cvx.norm1(matmul(weight, variable.var_N_tilde[:weight.shape[1]]))


class L22ObjectiveAtom(QuadraticObjectiveAtom):
    _atom_type = 'L22'


class LInfObjectiveAtom(ObjectiveAtomBase):
    _atom_type = 'Linf'

    def apply_atom_with_unity_weight(self, variable, weight, var_dim=None):
        return cvx.sum(cvx.norm_inf(variable.var_mat_N_tilde, axis=0))

    def apply_atom_with_vector_weight(self, variable, weight, var_dim=None):
        vec_weighted = cvx.multiply(weight, variable.var_N_tilde[:weight.shape[0]])
        return cvx.sum(cvx.norm1(cvx.reshape(vec_weighted, (var_dim, vec_weighted.size / var_dim)), axis=0))

    def apply_atom_with_matrix_weight(self, variable, weight, var_dim=None):
        mat_weighted = matmul(weight, variable.var_N_tilde[:weight.shape[1]])
        return cvx.sum(cvx.norm1(cvx.reshape(mat_weighted, (var_dim, mat_weighted.size / var_dim)), axis=0))


_ATOM_TYPES_MAP = {
    'Linear'   : LinearObjectiveAtom,
    'Quadratic': QuadraticObjectiveAtom,
    'L1'       : L1ObjectiveAtom,
    'L22'      : L22ObjectiveAtom,
    'Linf'     : LInfObjectiveAtom
}

_WEIGHT_TYPES_MAP = {
    'vector': VectorWeight,
    'matrix': MatrixWeight,
}

ObjectiveAtomStruct = named_struct_dict('ObjectiveAtomStruct', sorted_repr=True)


class ObjectiveAtoms(ComponentBase):
    _allowed_atom_types = _objective_atom_types
    _allowed_weight_types = _objective_weight_types
    _allowed_post_fix = {'N_p', "N_tilde", "f", ""}

    _field_names = MldInfo._var_names
    _field_names_set = frozenset(_field_names)
    _var_names = _field_names

    _regex_atom_type_pat = regex.compile(r'(Linear)|(Quadratic)|([L](1|(22)|(inf)))')
    _regex_atom_is_rate_pat = regex.compile(r'[dD][^e]')

    def _process_atom_names_args(self, f_kwargs=None, *,
                                 objective_atoms_struct=ParNotSet, _var_kwargs=ParNotSet,
                                 **kwargs):
        _var_kwargs = _var_kwargs or {}
        if objective_atoms_struct is None:
            objective_atoms_struct = _var_kwargs
        else:
            try:
                objective_atoms_struct.update(_var_kwargs)
            except AttributeError:
                raise TypeError("objective_atoms_struct must be of type subclass of dict or None")

        f_kwargs['objective_atoms_struct'] = objective_atoms_struct

    @process_method_args_decor(_process_atom_names_args)
    def __init__(self, controller=None, objective_atoms_struct=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k=ParNotSet, mld_numeric_tilde=ParNotSet, **kwargs):
        super(ObjectiveAtoms, self).__init__(controller, N_p=N_p, N_tilde=N_tilde,
                                             mld_numeric_k=mld_numeric_k, mld_numeric_tilde=mld_numeric_tilde)
        self.update(_disable_process_args=True, objective_atoms_struct=objective_atoms_struct, **kwargs)

    def _reset(self):
        super(ObjectiveAtoms, self)._reset()
        self._base_dict_init(self.base_dict.fromkeys(self._var_names))

    @increments_version_decor
    @process_method_args_decor(_process_atom_names_args)
    def update(self, objective_atoms_struct=None, **kwargs):
        N_p = self.N_p
        N_tilde = self.N_tilde
        mld_info_k = self.mld_info_k
        update_atoms = self.as_base_dict()
        for var_name_or_atom_str, weight in objective_atoms_struct.items():
            if var_name_or_atom_str in self._var_names:
                if isinstance(weight, ObjectiveAtomBase):
                    pass
                    # todo tidyup
                    # self._set_atom(update_atoms, var_name_or_atom_str, weight.weight_type,
                    #                weight['weight_N_tilde'], 'weight_N_tilde')
                else:
                    raise TypeError(
                        f"{var_name_or_atom_str} value in objective_weights_struct must be"
                        f" subclass of {ComponentBase.__name__}")
            else:
                self._set_atom_from_string(update_atoms=update_atoms, string_in=var_name_or_atom_str,
                                           value=weight, N_p=N_p, N_tilde=N_tilde, mld_info_k=mld_info_k)

        for var_name, var_atoms in update_atoms.items():
            if var_atoms is not None and not var_atoms:
                update_atoms[var_name]=None

        self._base_dict_update(update_atoms)
        self._update_set_with(N_p, N_tilde)

    @process_method_args_decor(_process_atom_names_args)
    def set(self, objective_atoms_struct=None, **kwargs):
        self.clear()
        self.update(_disable_process_args=True, objective_atoms_struct=objective_atoms_struct, **kwargs)

    def _set_atom_from_string(self, update_atoms, string_in, value, N_p, N_tilde, mld_info_k):
        atom_info = string_in.split('_')
        weight_type_name = 'vector' if "".join(atom_info[0:1]).islower() else 'matrix'

        atom_type_name = "".join(atom_info[1:2]).capitalize()
        if not self._regex_atom_type_pat.search(atom_type_name):
            atom_type_name = "Linear" if weight_type_name == 'vector' else "Quadratic"
            var_name = "".join(atom_info[1:2]).lower()
            post_fix = "_".join(atom_info[2:])
        else:
            var_name = "".join(atom_info[2:3]).lower()
            post_fix = "_".join(atom_info[3:])

        is_rate_atom = False
        if self._regex_atom_is_rate_pat.search(var_name):  # ^e to enable capture of delta variable
            var_name = var_name[1:]
            is_rate_atom = True

        atom_name = "_".join([atom_type_name, weight_type_name]) + ("_d" if is_rate_atom else "")

        if var_name in self._var_names and post_fix in self._allowed_post_fix:
            if value is None:
                return

            value = atleast_2d_col(value)
            var_dim = mld_info_k.get_var_dim(var_name)

            if post_fix:
                weight_length_name = "_".join(['weight', post_fix])
            elif value.shape[0] == var_dim or value.shape[0] == var_dim * N_tilde:
                weight_length_name = "weight_N_tilde"
            else:
                weight_length_name = "weight_N_p"

            atom_type: ObjectiveAtomBase = _ATOM_TYPES_MAP[atom_type_name]
            weight_type: ObjectiveWeightBase = _WEIGHT_TYPES_MAP[weight_type_name]

            self._set_atom(update_atoms, var_name,
                           atom_name, atom_type, is_rate_atom,
                           weight_type, weight_length_name, value)
        else:
            raise ValueError(
                f"weight_name: '{string_in}' is not valid. Must be of the form:\n"
                f"  \"lower/upper[_Linear|_Quadratic|_L1|_L22|_Linf]_[d]var_name[_N_tilde|_N_p|_f]\"")

    def _set_atom(self, update_atoms, var_name,
                  atom_name, atom_type: ObjectiveAtomBase, is_rate_atom,
                  weight_type: ObjectiveWeightBase, weight_length_name, value):

        var_atoms = update_atoms.get(var_name)
        if not var_atoms:
            update_atoms[var_name] = var_atoms = ObjectiveAtomStruct()

        var_atom = var_atoms.get(atom_name)
        weight_value = {weight_length_name: value}
        if not var_atom and not np.all(np.isclose(value, 0.0)):
            weight = weight_type.from_component(self, var_name=var_name, **weight_value)
            update_atoms[var_name][atom_name] = atom_type.from_component(self, var_name=var_name,
                                                                         weight=weight,
                                                                         is_rate_atom=is_rate_atom)
        elif isinstance(var_atom, ObjectiveAtomBase):
            if isinstance(var_atom.weight, ObjectiveWeightBase):
                var_atom.weight.update(**weight_value)
            else:
                var_atom.weight = weight_type.from_component(self, var_name=var_name,
                                                             **weight_value)
            if var_atom.weight.is_zero():
                del var_atoms[atom_name]


    def gen_cost(self, variables: EvoVariables):
        cost = 0
        atoms: ObjectiveAtomBase
        for var_name, atoms in self.items():
            if atoms is not None:
                for atom_name, atom in atoms.items():
                    if atom is not None:
                        opt_var = variables.get(var_name)
                        cost += atom.apply_atom(variable=opt_var)
        return cost
