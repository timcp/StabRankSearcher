"""Usage:
python3 dicke_state_analyzer.py A B C D
with
- A = <number-of-qubits>
- B = <hamming-weight>
- C = <stabilizer-rank-to-search-for>
- D = number of random trials

"""
import sys
import logging
import argparse
import netsquid as ns
import numpy as np
from stabranksearcher.dicke_state_factory import get_dicke_state
from stabranksearcher.rank_searcher import RandomWalkStabRankSearcher
from stabranksearcher.quantum_state_tools import ket_to_qstate


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Randomly search for stabilizer bases for Dicke states.')
    parser.add_argument('--number_of_qubits', type=int, required=True)
    parser.add_argument('--hamming_weight', type=int, required=True)
    parser.add_argument('--stabrank', type=int, default=1)
    parser.add_argument('--number_of_attempts', type=int, default=1000)
    parser.add_argument('--beta_init', type=float, default=1)
    parser.add_argument('--beta_final', type=float, default=100)
    parser.add_argument('--number_of_betas', type=float, default=100)
    parser.add_argument('--loglevel', type=str, default=None)
    parser.add_argument('--outputfile', type=str, default=None)
    args = parser.parse_args()

    if args.loglevel == "INFO":
        loglevel = logging.INFO
    elif args.loglevel == "DEBUG":
        loglevel = logging.DEBUG
    else:
        loglevel = None
    logging.basicConfig(level=loglevel)

    logging.info("Attempting to find stabilizer rank of Dick state.")
    logging.info("Parameters:\n\t- number of qubits: {}\n\t- Hamming weight: {}\n\t- Stabilizer rank to search for: {}\n\t- beta_init: {}\n\t- beta_final: {}\n\t- number_of_betas: {}".format(args.number_of_qubits, args.hamming_weight, args.stabrank, args.beta_init, args.beta_final, args.number_of_betas))

    dicke_ket = get_dicke_state(number_of_qubits=args.number_of_qubits,
                                hamming_weight=args.hamming_weight)
    qstate = ket_to_qstate(dicke_ket)
    searcher = RandomWalkStabRankSearcher(beta_init=args.beta_init, beta_final=args.beta_final, number_of_betas=args.number_of_betas)
    basis = searcher.run(target_qstate=qstate, stabrank=args.stabrank,
                         number_of_bases=args.number_of_attempts)
    logging.info("Found basis:{}".format(basis))
    if basis is not None:
        logging.info("Found rank: {}".format(basis.size))

        should_write_to_file = (args.outputfile is not None)
        if should_write_to_file:
            data = np.array([qstate.ket.flatten() for qstate in basis.qstates])
            np.savetxt(args.outputfile, data)
    logging.info("Number of attempts: {}".format(searcher.counter))

    output = 0 if basis is None else basis.size
    print("{},{},{},{},{}".format(args.number_of_qubits,
                                  args.hamming_weight,
                                  args.stabrank,
                                  args.number_of_attempts,
                                  output))

    run_was_successful = (basis is not None)
    if run_was_successful:
        sys.exit(1)
    else:
        sys.exit(0)
