import numpy as np
import netsquid as ns
from stabranksearcher.rank_searcher import NRandomStabRankSearcher
from stabranksearcher.quantum_state_tools import ket_to_qstate

if __name__ == "__main__":

    # Trying a stabilizer state
    ket = np.array([[1, 1j]])
    ket = np.array([[1, 1j]]) / np.linalg.norm(ket)
    stabrank = 1
    print("\n\nNow investigating {} with stabilizer rank {}".format(
            ket, stabrank))
    qstate = ket_to_qstate(ket)
    searcher = NRandomStabRankSearcher()
    basis = searcher.run(target_qstate=qstate, stabrank=stabrank, number_of_bases=100)
    print("Found basis:", basis)
    if basis is not None:
        print("Found rank:", basis.size)
    print("Number of attempts:", searcher.counter)

    # Trying a non-stabilizer state with...
    # ...stabilizer rank 1
    stabrank = 1
    ket = np.array([[1, 0.25]])
    ket = ket / np.linalg.norm(ket)
    qstate = ket_to_qstate(ket)
    searcher = NRandomStabRankSearcher()
    basis = searcher.run(target_qstate=qstate, stabrank=stabrank, number_of_bases=100)
    print("\n\nNow investigating {} with stabilizer rank {}".format(
            ket, stabrank))
    print("Found basis:", basis)
    print("Number of attempts:", searcher.counter)

    # ...and with stabilizer rank 2
    searcher.reset()
    stabrank = 2
    basis = searcher.run(target_qstate=qstate, stabrank=2, number_of_bases=1000)
    print("\n\nNow investigating {} with stabilizer rank {}".format(
            ket, stabrank))
    print("Found basis:", basis)
    if basis is not None:
        print("Found rank:", basis.size)
    print("Number of attempts:", searcher.counter)
