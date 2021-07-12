import itertools
from netsquid.qubits.stabtools import StabRepr
import netsquid.qubits.qubitapi as qapi
from stabranksearcher.basis import Basis


class BruteForceStabBasisProvider:

    def __init__(self, number_of_qubits=1):
        self._number_of_qubits = number_of_qubits
        self._counter = 0
        self._stabilizer_bases = \
            BruteForceStabBasisProvider.get_all_stabilizer_bases(
                number_of_qubits=self._number_of_qubits)

    @staticmethod
    def get_all_stabilizer_bases_of_given_stabilizer_rank(
            number_of_qubits=1, stabrank=1):
        possible_stabilizer_states = \
            BruteForceStabBasisProvider.get_all_stabilizer_states(
                number_of_qubits=number_of_qubits)
        combinations = \
            itertools.combinations(possible_stabilizer_states, stabrank)
        # TODO make use of generator instead of list to save memory usage
        return [Basis(qstates=list(combination))
                for combination in combinations]

    @staticmethod
    def get_all_stabilizer_bases(number_of_qubits=1):
        # TODO make use of generator instead of list to save memory usage
        bases = []
        possible_stabilizer_ranks = range(1, 2 ** number_of_qubits + 1)
        for stabrank in possible_stabilizer_ranks:
            bases_of_fixed_rank = \
                BruteForceStabBasisProvider.get_all_stabilizer_bases_of_given_stabilizer_rank(
                    number_of_qubits=number_of_qubits,
                    stabrank=stabrank)
            bases += bases_of_fixed_rank
        return bases

    @staticmethod
    def get_all_stabilizer_states(number_of_qubits=1):
        """
        Returns
        -------
        list of QStates
        """
        if number_of_qubits == 1:
            X_PLUS = StabRepr(check_matrix=[[1, 0]], phases=[1])
            X_MINUS = StabRepr(check_matrix=[[1, 0]], phases=[-1])
            Z_PLUS = StabRepr(check_matrix=[[0, 1]], phases=[1])
            Z_MINUS = StabRepr(check_matrix=[[0, 1]], phases=[-1])
            Y_PLUS = StabRepr(check_matrix=[[1, 1]], phases=[1])
            Y_MINUS = StabRepr(check_matrix=[[1, 1]], phases=[-1])
            qstates = set()
            for srepr in [X_PLUS, X_MINUS, Z_PLUS, Z_MINUS, Y_PLUS, Y_MINUS]:
                qubits = qapi.create_qubits(num_qubits=1)
                qapi.assign_qstate(qubits, srepr)
                qstates.add(qubits[0].qstate)
            return qstates
        else:
            raise NotImplementedError  # TODO

    def _has_seen_all_bases(self):
        return self._counter > len(self._stabilizer_bases) - 1

    def get_next_basis(self):
        if self._has_seen_all_bases():
            return None
        else:
            self._counter += 1
            return self._stabilizer_bases[self._counter - 1]
