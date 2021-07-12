import unittest
import numpy as np
from netsquid.qubits.stabtools import StabRepr
import netsquid.qubits.qubitapi as qapi
from stabranksearcher.stab_basis_provider.random_walk import (
        BasisWithTargetState,
        RandomWalkStabBasisProvider,
        MoveDecider)
from stabranksearcher.quantum_state_tools import ket_to_qstate


class TestBasisWithTargetState(unittest.TestCase):

    # TODO
    pass


class TestRandomWalkStabBasisProvider(unittest.TestCase):

    class TurnOnOffMoveDecider(MoveDecider):

        def __init__(self, on=True):
            self.on = on

        def should_move(self, current_score, tentative_next_score):
            return self.on

    class RandomWalkStabBasisProviderWithPlusStateAsInitialState(RandomWalkStabBasisProvider):

        @staticmethod
        def get_random_stabilizer_state(number_of_qubits=1):
            # mock method to always obtain the |+> state at the start
            srepr = StabRepr(check_matrix=[[1, 0]], phases=[1])
            qubits = qapi.create_qubits(num_qubits=number_of_qubits)
            qapi.assign_qstate(qubits, srepr)
            return qubits[0].qstate

    class BasisWithTargetStateSequential(BasisWithTargetState):

        QSTATES = [
            ket_to_qstate(np.array([[1, 0]])),
            ket_to_qstate(np.array([[0, 1]])),
            ket_to_qstate(np.array([[1, 1]])),
            ket_to_qstate(np.array([[1, -1]])),
            ket_to_qstate(np.array([[1, -1]])),
            ket_to_qstate(np.array([[1, -1]]))
            ]

        def __init__(self, qstates, target_qstate):
            super().__init__(qstates=qstates, target_qstate=target_qstate)
            self.counter = 0

        def randomly_modify(self):
            if self.counter < len(self.QSTATES):
                self._qstates[0] = self.QSTATES[self.counter]
                self.counter += 1
            else:
                raise Exception

    def test_get_next_basis(self):

        zero_ket = np.array([[1, 0]])
        zero_qstate = ket_to_qstate(zero_ket)
        provider = \
            TestRandomWalkStabBasisProvider.RandomWalkStabBasisProviderWithPlusStateAsInitialState(
                target_qstate=zero_qstate,
                stabrank=1)

        move_decider = self.TurnOnOffMoveDecider(on=False)

        # check that if the move decider is turned 'off', then we a move is never performed,
        # i.e. we always get the same state back (the |+> state)
        s = 1 / np.sqrt(2)
        plus_ket = np.array([[s, s]])
        plus_qstate = ket_to_qstate(plus_ket)

        for __ in range(10):
            basis = provider.get_next_basis(move_decider=move_decider)
            self.assertTrue(basis.does_qstate_live_in_subspace(qstate=plus_qstate))

        # check that if the move decider is turned 'on', then we a move is always performed
        # i.e. we always get the same state back (the |+> state)
        move_decider = self.TurnOnOffMoveDecider(on=True)

        # overwriting the basis that the StabBasisProvider currently holds
        provider._basis_with_target_state = self.BasisWithTargetStateSequential(
            qstates=basis.qstates,
            target_qstate=plus_qstate)

        # checking that we get the bases we expect
        for index in range(len(self.BasisWithTargetStateSequential.QSTATES)):
            basis = provider.get_next_basis(move_decider=move_decider)
            self.assertTrue(basis.does_qstate_live_in_subspace(
                                qstate=self.BasisWithTargetStateSequential.QSTATES[index]))

        with self.assertRaises(Exception):
            basis = provider.get_next_basis(move_decider=move_decider)


if __name__ == "__main__":
    unittest.main()
