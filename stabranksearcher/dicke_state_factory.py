import numpy as np


def _convert_integer_to_binary_list(integer=1):
    if integer < 0:
        raise ValueError
    bitstring = "{0:b}".format(integer)
    binary_list = [int(s) for s in bitstring]
    return binary_list


def _hamming_weight_of_binary_list(binary_list):
    return sum(binary_list)


def get_dicke_state(number_of_qubits=1, hamming_weight=0):
    """
    Returns
    -------
    numpy ndarray
    """
    state_vector = np.array([0] * (2 ** number_of_qubits))
    for integer in range(2 ** number_of_qubits):
        binary_list = _convert_integer_to_binary_list(integer=integer)
        if _hamming_weight_of_binary_list(binary_list=binary_list) == hamming_weight:
            state_vector[integer] = 1
    return state_vector / np.linalg.norm(state_vector)
