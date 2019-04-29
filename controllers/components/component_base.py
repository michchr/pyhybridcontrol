from models.mld_model import MldModel, MldInfo
from structdict import NamedFixedStructDictMixin
from utils.func_utils import ParNotSet
from utils.versioning import versioned, VersionMixin
import numpy as np


from typing import TYPE_CHECKING as _TYPE_CHECKING
if _TYPE_CHECKING:
    from controllers.controller_base import ControllerBase


@versioned(versioned_sub_objects=('mld_numeric_k', 'mld_numeric_tilde'))
class ComponentBase(NamedFixedStructDictMixin, dict, VersionMixin):
    _field_names = ()
    _field_names_set = frozenset()

    __internal_names = ['_controller', '_N_p', '_N_tilde', '_mld_numeric_k', '_mld_numeric_tilde', 'set_with_N_p',
                        'set_with_N_tilde', '_stored_version']
    _internal_names_set = set(__internal_names)

    @classmethod
    def from_component(cls, control_component: 'ComponentBase', *args,
                       controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                       mld_numeric_k: MldModel = ParNotSet, mld_numeric_tilde=ParNotSet, **kwargs):
        new_comp = cls.__new__(cls)
        kwargs['controller'] = controller if controller is not None else control_component._controller
        kwargs['N_p'] = N_p if N_p is not ParNotSet else control_component._N_p
        kwargs['N_tilde'] = N_tilde if N_tilde is not ParNotSet else control_component._N_tilde
        kwargs['mld_numeric_k'] = mld_numeric_k if mld_numeric_k is not ParNotSet else control_component._mld_numeric_k
        kwargs['mld_numeric_tilde'] = mld_numeric_tilde if mld_numeric_tilde is not ParNotSet else mld_numeric_tilde
        new_comp.__init__(*args, **kwargs)
        return new_comp

    def __init__(self, controller=None, N_p=ParNotSet, N_tilde=ParNotSet,
                 mld_numeric_k: MldModel = ParNotSet, mld_numeric_tilde=ParNotSet):
        super(ComponentBase, self).__init__()

        self._controller: ControllerBase = controller
        self._N_p = N_p
        self._N_tilde = N_tilde
        self._mld_numeric_tilde = mld_numeric_tilde
        self._mld_numeric_k = mld_numeric_k

        if self._controller is None:
            if not isinstance(N_p, (int, np.integer)):
                raise ValueError("N_p must be an integer")
            if not isinstance(N_tilde, (int, np.integer)):
                raise ValueError("N_tilde must be an integer")
            if not isinstance(mld_numeric_k, MldModel) or mld_numeric_k.mld_type != MldModel.MldModelTypes.numeric:
                raise ValueError("mld_numeric_k must be an MldModel with mld_type=='numeric'")
        self._reset()

    def _reset(self):
        self.set_with_N_p = None
        self.set_with_N_tilde = None

    def _update_set_with(self, N_p, N_tilde):
        self.set_with_N_p = N_p
        self.set_with_N_tilde = N_tilde

    @property
    def N_p(self):
        N_p = self._N_p
        return (N_p if N_p is not ParNotSet else self._controller.N_p)

    @property
    def N_tilde(self):
        N_tilde = self._N_tilde
        return (N_tilde if N_tilde is not ParNotSet else self._controller.N_tilde)

    @property
    def mld_numeric_k(self) -> MldModel:
        mld_numeric_tilde = self._mld_numeric_tilde
        mld_numeric_k = mld_numeric_tilde[0] if mld_numeric_tilde not in (None, ParNotSet) else self._mld_numeric_k
        return (mld_numeric_k if mld_numeric_k is not ParNotSet else self._controller.mld_numeric_k)

    @property
    def mld_numeric_tilde(self):
        mld_numeric_tilde = self._mld_numeric_tilde
        return (mld_numeric_tilde if mld_numeric_tilde is not ParNotSet else self._controller.mld_numeric_tilde)

    @property
    def mld_info_k(self) -> MldInfo:
        return self.mld_numeric_k.mld_info

    @property
    def x_k(self):
        return self._controller.x_k

    @property
    def omega_tilde_k(self):
        return self._controller.omega_tilde_k

    def set_build_required(self):
        if self._controller:
            self._controller.set_build_required()
