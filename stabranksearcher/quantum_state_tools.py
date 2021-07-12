import numpy as np
import netsquid as ns
import netsquid.qubits.qubitapi as qapi


def get_number_of_qubits_from_ket(ket):
    return int(np.log(ket.size) / np.log(2))


def ket_to_qstate(ket):
    krepr = ns.qubits.kettools.KetRepr(ket)
    number_of_qubits = get_number_of_qubits_from_ket(ket=ket)
    qubits = qapi.create_qubits(num_qubits=number_of_qubits)
    qapi.assign_qstate(qubits, krepr)
    return qubits[0].qstate

