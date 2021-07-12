import unittest
import numpy as np
import netsquid as ns
from netsquid.qubits.stabtools import StabRepr
import netsquid as ns
import netsquid.qubits.qubitapi as qapi
from stabranksearcher.basis import Basis, get_basis_copy
from stabranksearcher.stab_basis_provider.stab_basis_provider import StabBasisProvider
from stabranksearcher.rank_searcher import (
    BruteForceStabRankSearcher,
    NRandomStabRankSearcher,
    RandomWalkStabRankSearcher)
from stabranksearcher.quantum_state_tools import ket_to_qstate


## Test BruteForceStabRankSearcher

#ket = np.array([[1, 0]])
#krepr = ns.qubits.kettools.KetRepr(ket)
#searcher = BruteForceStabRankSearcher()
#basis = searcher.run(ket=krepr)
#print(basis.size)
#
#
#
#print('\n\n')
#ket = np.array([[1, 1j]])
#ket = np.array([[1, 1j]]) / np.linalg.norm(ket)
#krepr = ns.qubits.kettools.KetRepr(ket)
#searcher = BruteForceStabRankSearcher()
#basis = searcher.run(ket=krepr)
#print(basis.size)
#
#
#print('\n\n')
#ket = np.array([[1, 0.25]])
#ket = ket / np.linalg.norm(ket)
#krepr = ns.qubits.kettools.KetRepr(ket)
#searcher = BruteForceStabRankSearcher()
#basis = searcher.run(ket=krepr)
#print(basis.size)




#ket = np.array([[1, 0.25]])
#ket = ket / np.linalg.norm(ket)
##ket = np.array([[1, 1j]])
##ket = np.array([[1, 1j]]) / np.linalg.norm(ket)
#krepr = ns.qubits.kettools.KetRepr(ket)
#searcher = NRandomStabRankSearcher()
#basis = searcher.run(ket=krepr, stabrank=1, number_of_bases=10)
#print("Found basis:", basis)
#if basis is not None:
#    print("Found rank:", basis.size)
#print("Number of attempts:", searcher.counter)


class TestRandomWalkStabRankSearcher(unittest.TestCase):

    class ConstantStabBasisProvider(StabBasisProvider):

        def __init__(self, target_qstate, stabrank):
            if target_qstate.num_qubits != 1:
                raise NotImplementedError
            z_plus = StabRepr(check_matrix=[[0, 1]], phases=[1])
            qubits = qapi.create_qubits(num_qubits=1)
            qapi.assign_qstate(qubits, z_plus)
            qstate = qubits[0].qstate
            self._basis = Basis(qstates=[qstate])

        def get_next_basis(self, move_decider=None):
            return get_basis_copy(self._basis)

    class ConstantAfterFirstStabBasisProvider(StabBasisProvider):

        def __init__(self, target_qstate, stabrank):
            if target_qstate.num_qubits != 1:
                raise NotImplementedError

            # first basis
            z_minus = StabRepr(check_matrix=[[0, 1]], phases=[-1])
            qubits = qapi.create_qubits(num_qubits=1)
            qapi.assign_qstate(qubits, z_minus)
            qstate = qubits[0].qstate
            self._first_basis = Basis(qstates=[qstate])

            # second basis
            z_plus = StabRepr(check_matrix=[[0, 1]], phases=[1])
            qubits = qapi.create_qubits(num_qubits=1)
            qapi.assign_qstate(qubits, z_plus)
            qstate = qubits[0].qstate
            self._second_basis = Basis(qstates=[qstate])

            self._has_delivered_first_basis_already = False

        def get_next_basis(self, move_decider=None):
            if self._has_delivered_first_basis_already:
                return get_basis_copy(self._second_basis)
            else:
                self._has_delivered_first_basis_already = True
                return self._first_basis

    def test_run_constant_stab_basis_provider(self):

        # case: returned stabilizer is the correct one
        ket = np.array([[1, 0]])
        qstate = ket_to_qstate(ket=ket)
        searcher = RandomWalkStabRankSearcher(beta_init=0, beta_final=10, number_of_betas=1)
        searcher.STAB_BASIS_PROVIDER_CLS = TestRandomWalkStabRankSearcher.ConstantStabBasisProvider
        basis = searcher.run(target_qstate=qstate, stabrank=1, number_of_bases=10)
        self.assertTrue(basis is not None)
        self.assertEqual(searcher.counter, 1)

        # case: returned stabilizer is not the correct one
        ket = np.array([[0, 1]])
        qstate = ket_to_qstate(ket=ket)
        searcher = RandomWalkStabRankSearcher(beta_init=0, beta_final=10, number_of_betas=1)
        searcher.STAB_BASIS_PROVIDER_CLS = TestRandomWalkStabRankSearcher.ConstantStabBasisProvider
        basis = searcher.run(target_qstate=qstate, stabrank=1, number_of_bases=42)
        self.assertTrue(basis is None)
        self.assertEqual(searcher.counter, 42)

    def test_run_constant_after_first_stab_basis_provider(self):

        # case: first state is correct
        ket = np.array([[0, 1]])
        qstate = ket_to_qstate(ket=ket)
        searcher = RandomWalkStabRankSearcher(beta_init=0, beta_final=10, number_of_betas=1)
        searcher.STAB_BASIS_PROVIDER_CLS = \
            TestRandomWalkStabRankSearcher.ConstantAfterFirstStabBasisProvider
        basis = searcher.run(target_qstate=qstate, stabrank=1, number_of_bases=10)
        self.assertTrue(searcher.counter, 1)
        self.assertTrue(basis is not None)

        # case: second state is correct
        ket = np.array([[1, 0]])
        qstate = ket_to_qstate(ket=ket)
        searcher = RandomWalkStabRankSearcher(beta_init=0, beta_final=10, number_of_betas=1)
        searcher.STAB_BASIS_PROVIDER_CLS = \
            TestRandomWalkStabRankSearcher.ConstantAfterFirstStabBasisProvider
        basis = searcher.run(target_qstate=qstate, stabrank=1, number_of_bases=10)
        self.assertTrue(searcher.counter, 2)
        self.assertTrue(basis is not None)

        # case: state is never correct
        s = 1.0 / np.sqrt(2)
        ket = np.array([[s, s]])
        qstate = ket_to_qstate(ket=ket)
        searcher = RandomWalkStabRankSearcher(beta_init=0, beta_final=10, number_of_betas=1)
        searcher.STAB_BASIS_PROVIDER_CLS = \
            TestRandomWalkStabRankSearcher.ConstantAfterFirstStabBasisProvider
        basis = searcher.run(target_qstate=qstate, stabrank=1, number_of_bases=43)
        self.assertTrue(searcher.counter, 43)
        self.assertTrue(basis is None)


if __name__ == "__main__":
    unittest.main()
