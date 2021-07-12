import numpy as np
from stabranksearcher.basis import Basis
from stabranksearcher.stab_basis_provider.stab_basis_provider import StabBasisProvider
from stabranksearcher.stab_basis_provider.random import RandomStabBasisProvider


class MoveDecider:

    def should_move(self, current_score, tentative_next_score):
        return True

class SimulatedAnnealingMoveDecider:

    def __init__(self, beta):
        self.beta = beta

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, val):
        if val < 0:
            raise ValueError
        self._beta = val

    def should_move(self, current_score, tentative_next_score):
        if tentative_next_score > current_score:
            return True
        else:
            prob_accept = np.exp(-1 * self.beta * (current_score - tentative_next_score))
            return np.random.random() < prob_accept


class BasisWithTargetState(Basis):
    """:cls:`~stabranksearcher.basis.Basis` together with
    the target state. The intention of this class is
    improved speed, since now we can store the score
    of this basis with respect to this state, without
    having to recompute it again every time the `score`
    method is called.
    """

    def __init__(self, qstates, target_qstate):
        super().__init__(qstates=qstates)
        self._target_qstate = target_qstate
        self._score = None

    def score(self, qstate):
        if qstate == self._target_qstate and self._score is not None:
            return self._score
        else:
            return super().score(qstate=qstate)

    def move(self, move_decider, qstate_index=None, pauli=None):

        # store current score (for sake of speed when
        # undoing the move)
        current_score = self.score(qstate=self._target_qstate)

        # perform the move
        if qstate_index is None or pauli is None:
            self.randomly_modify()
        else:
            self.deterministically_modify(qstate_index=qstate_index, pauli=pauli)

        # store tentative next score (for sake of speed when
        # keep the move)
        tentative_next_score = self.score(qstate=self._target_qstate)

        # decide whether to keep the move
        if move_decider.should_move(current_score=current_score, tentative_next_score=tentative_next_score):
            self._score = tentative_next_score
        else:
            self.undo_last_modification()
            self._score = current_score

    def deterministically_modify(self, qstate_index, pauli):
        self._score = None
        return super().deterministically_modify(qstate_index=qstate_index, pauli=pauli)

    def undo_last_modification(self):
        self._score = None
        return super().undo_last_modification()


class RandomWalkStabBasisProvider(RandomStabBasisProvider):
    """Class for providing the next step of a random walk
    over all tuples of stabilizer states. Used as subroutine
    for implementing the random walk algorithm from Appendix B
    of:

    Sergey Bravyi, Graeme Smith, and John A. Smolin
    Trading Classical and Quantum Computational Resources
    (2016), https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.021043

    Parameters
    ----------
    number_of_qubits: int
    stabrank: int
        The target stabilizer rank, i.e. the size of the tuples
        of stabilizer states.
    """

    def __init__(self, target_qstate, stabrank=1):
        self._target_qstate = target_qstate
        self._number_of_qubits = self._target_qstate.num_qubits
        self._stabrank = stabrank
        self._counter = 0
        self._basis_with_target_state = None

    def get_next_basis(self, move_decider=None):
        r"""Modifies the previous_basis and returns the modified basis.

        Parameters
        ----------
        move_decider: :obj:`~stabranksearcher.stab_basis_provider.random_walk.MoveDecider`

        Notes
        -----
        The modification is done by choosing one of the states in the basis
        :math:`\ket{\phi}` uniformly at random, and replacing it by
        :math:`\ket{\phi'} := c(I + P)\ket{\phi}`, where :math:`I` is the
        identity operator, :math:`P` is a random Pauli (including
        phase :math:`\in \{\pm 1,\pm i\}` on the same number of qubits as
        :math:`\ket{\phi}`, and :math:`c` is the normalization constant.
        (This modification is restarted until :math:`\ket{\phi'}` is not
        the all-zero vector.)
        """
        if self._counter == 0:
            self._counter += 1
            basis = self.get_random_stabilizer_state_basis(number_of_qubits=self._number_of_qubits, size=self._stabrank)
            self._basis_with_target_state = \
                BasisWithTargetState(qstates=basis.qstates, target_qstate=self._target_qstate)
        else:
            self._basis_with_target_state.move(move_decider=move_decider)
        return self._basis_with_target_state
