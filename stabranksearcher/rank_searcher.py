import numpy as np
import netsquid as ns
from stabranksearcher.stab_basis_provider.stab_basis_provider import StabBasisProvider
from stabranksearcher.stab_basis_provider.brute_force import BruteForceStabBasisProvider
from stabranksearcher.stab_basis_provider.random import RandomStabBasisProvider
from stabranksearcher.stab_basis_provider.random_walk import RandomWalkStabBasisProvider, SimulatedAnnealingMoveDecider
from stabranksearcher.quantum_state_tools import ket_to_qstate


class StabRankSearcher:

    def __init__(self):
        self._stab_basis_provider = None

    def reset(self):
        pass

    @property
    def stab_basis_provider(self):
        return self._stab_basis_provider

    @stab_basis_provider.setter
    def stab_basis_provider(self, val):
        if not isinstance(val, StabBasisProvider):
            raise TypeError("{} not of type StabBasisProvider".format(val))
        self._stab_basis_provider = val

    def run(self):
        if self._stab_basis_provider is None:
            raise ValueError("Need to first set attribute stab_basis_provider")



class BruteForceStabRankSearcher(StabRankSearcher):

    def run(self, ket):
        self._stab_basis_provider = \
            BruteForceStabBasisProvider(number_of_qubits=ket.num_qubits)
        super().run()
        basis = self._stab_basis_provider.get_next_basis()
        while basis is not None:
            outcome = basis.does_qstate_live_in_subspace(ket)
            if outcome:
                return basis
            basis = self._stab_basis_provider.get_next_basis()
        return None


class NRandomStabRankSearcher(StabRankSearcher):

    STAB_BASIS_PROVIDER_CLS = RandomStabBasisProvider

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def run(self, target_qstate, stabrank=1, number_of_bases=1):
        if not isinstance(target_qstate, ns.qubits.qstate.QState):
            raise TypeError
        self._stab_basis_provider = \
            self.STAB_BASIS_PROVIDER_CLS(number_of_qubits=target_qstate.num_qubits,
                                         stabrank=stabrank)
        super().run()
        while self._counter < number_of_bases:
            self._counter += 1
            basis = self._stab_basis_provider.get_next_basis()
            if basis.does_qstate_live_in_subspace(target_qstate):
                return basis
        return None


class RandomWalkStabRankSearcher(StabRankSearcher):

    STAB_BASIS_PROVIDER_CLS = RandomWalkStabBasisProvider

    def __init__(self, beta_init, beta_final, number_of_betas):
        super().__init__()
        self._beta_init = beta_init
        self._beta_final = beta_final
        self._number_of_betas = number_of_betas
        self._beta_step = (self._beta_final - self._beta_init) / number_of_betas
        self.reset()

    def reset(self):
        self._total_counter = 0

    @property
    def counter(self):
        return self._total_counter

    def run(self, target_qstate, stabrank=1, number_of_bases=1):
        if not isinstance(target_qstate, ns.qubits.qstate.QState):
            raise TypeError
        self.stab_basis_provider = \
            self.STAB_BASIS_PROVIDER_CLS(target_qstate=target_qstate,
                                         stabrank=stabrank)
        super().run()
        beta = self._beta_init
        while beta < self._beta_final:
            counter = 0
            move_decider = SimulatedAnnealingMoveDecider(beta=beta)
            while counter < number_of_bases:
                basis = self.stab_basis_provider.get_next_basis(move_decider=move_decider)
                counter += 1
                if basis.does_qstate_live_in_subspace(target_qstate):
                    self._total_counter += counter
                    return basis
            self._total_counter += counter
            beta += self._beta_step
        return None
