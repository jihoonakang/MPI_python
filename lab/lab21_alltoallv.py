from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

isend = np.array([1, 2, 2, 3, 3, 3])
irecv = np.zeros(3 * (rank + 1), dtype = int)
iscnt = np.array([1, 2, 3])
ircnt = np.full(3, rank + 1, dtype = int)
isend += size * rank

comm.Alltoallv((isend, iscnt), (irecv, ircnt))
print('Rank({0}) : isend = {1}'.format(rank, isend))
print('Rank({0}) : irecv = {1}'.format(rank, irecv))
