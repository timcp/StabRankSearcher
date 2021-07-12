import numpy as np

import scipy
from netsquid.qubits.stabtools import StabRepr
import qiskit
import netsquid as ns
import netsquid.qubits.qubitapi as qapi
from stabranksearcher.quantum_state_tools import (
    ket_to_qstate,
    get_number_of_qubits_from_ket)


class Basis:
    """ is a list of QStates
    """

    class _Modification:

        def __init__(self, index, qstate):
            self.index = index
            self.qstate = qstate

    def __init__(self, qstates):
        self._qstates = qstates
        if len(list(set(qstate.num_qubits for qstate in qstates))) != 1:
            raise ValueError("QStates are not all on the same number of qubits")
        self._last_modification = None

    @property
    def qstates(self):
        return self._qstates

    def to_projector(self):
        """
        Returns
        -------
        numpy array
        """
        kets = [qstate.ket for qstate in self.qstates]
        number_of_qubits = get_number_of_qubits_from_ket(ket=kets[0])
        number_of_kets = self.size
        return Basis._kets_to_projector(kets=kets,
                                        number_of_qubits=number_of_qubits,
                                        number_of_kets=number_of_kets)

    @staticmethod
    def _kets_to_projector(kets, number_of_kets, number_of_qubits):

        # build a matrix of size `2^{number_of_qubits}` x `size`
        matrix = np.zeros((2 ** number_of_qubits, number_of_kets), dtype=np.complex_)

        ket_index = 0
        while ket_index < number_of_kets:
            matrix[:, ket_index] = kets[ket_index].flatten()
            ket_index += 1

        # The QR decomposition only works for linearly independent columns;
        # so we must first turn it into a linearly independent set
        matrix = scipy.linalg.orth(matrix)

        # compute the QR decomposition
        # TODO probably not needed any more since we have the orthonormal
        # basis set
        q, _ = np.linalg.qr(matrix)

        # compute the projector
        qconj = q.conj()
        projector = np.matmul(q, qconj.T)
        return projector

    @property
    def size(self):
        return len(self._qstates)

    @property
    def len(self):
        return self.size

    def __str__(self):
        return str([qstate.ket for qstate in self._qstates])

    def does_qstate_live_in_subspace(self, qstate):
        """
        Whether `qstate` lives in the subspace spanned by
        this basis.

        Parameters
        ----------
        ket: qstate

        Returns
        -------
        bool

        Internal working
        ----------------
        Checks if ||P|phi>|| == 1 where phi=ket and
        P is projector onto the basis
        """
        return np.isclose(self.score(qstate=qstate), 1)

    def score(self, qstate):
        projector = self.to_projector()
        output_ket = projector.dot(qstate.ket)
        norm = np.linalg.norm(output_ket)
        return norm

    @property
    def number_of_qubits(self):
        return self._qstates[0].num_qubits

    def undo_last_modification(self):
        if self._last_modification is None:
            raise Exception
        else:
            index = self._last_modification.index
            qstate = self._last_modification.qstate
            self._qstates[index] = qstate
            self._last_modification = None

    def randomly_modify(self):
        r"""Randomly choose a stabilizer state :math:`\ket{\phi}` in this basis
        and replace it by
        :math:`\ket{\phi'} := c(I + P)\ket{\phi}`, where :math:`I` is the
        identity operator, :math:`P` is a random Pauli (including
        phase :math:`\in \{\pm 1,\pm i\}` on the same number of qubits as
        :math:`\ket{\phi}`, and :math:`c` is the normalization constant.
        (This modification is restarted until :math:`\ket{\phi'}` is not
        the all-zero vector.)
        """
        accepted = False

        while not accepted:
            random_index = np.random.randint(low=0, high=self.size)
            random_pauli = qiskit.quantum_info.random_pauli(
                num_qubits=self.number_of_qubits,
                group_phase=True)
            accepted = self.deterministically_modify(
                qstate_index=random_index,
                pauli=random_pauli)

    def deterministically_modify(self, qstate_index, pauli):
        r"""Replace the k-th stabilizer state :math:`\ket{\phi}` in this basis
        and replace it by :math:`\ket{\phi'} := c(I + P)\ket{\phi}` where
        :math:`c` is a normalization constant.
        If :math:`\ket{\phi'} = 0`, then the replacement is not performed.

        Parameters
        ----------
        qstate_index: int
            The index of the stabilizer state in this basis that will be
            replaced
        pauli: :obj:`~qiskit.quantum_info.Pauli`
            Pauli string to be applied.

        Returns
        -------
        bool
            Whether the replacement was performed.
        """
        ket = self._qstates[qstate_index].ket
        outcome = Basis._deterministically_modify(ket=ket,
                                                  pauli_matrix=pauli.to_matrix())
        if outcome is None:
            return False
        else:
            new_qstate = ket_to_qstate(outcome)
            self._last_modification = Basis._Modification(index=qstate_index,
                                                          qstate=self._qstates[qstate_index])
            self._qstates[qstate_index] = new_qstate
            return True

    @staticmethod
    def _deterministically_modify(ket, pauli_matrix):
        """
        Returns
        -------
        ket or None
        """
        unnormalized_new_stabilizer_ket = ket + pauli_matrix.dot(ket)
        norm = np.linalg.norm(unnormalized_new_stabilizer_ket)
        if np.isclose(norm, 0.):
            return None
        else:
            return unnormalized_new_stabilizer_ket / norm


def get_basis_copy(basis):
    qstate_copies = []
    for qstate in basis._qstates:
        if isinstance(qstate.qrepr, StabRepr):
            qrepr = StabRepr(qstate.stab)  # TODO don't know if this is correct
        else:
            qrepr = ns.qubits.kettools.KetRepr(qstate.ket)
        qubits = qapi.create_qubits(num_qubits=qstate.num_qubits)
        qapi.assign_qstate(qubits, qrepr)
        qstate_copy = qubits[0].qstate
        qstate_copies.append(qstate_copy)
    return Basis(qstates=qstate_copies)
