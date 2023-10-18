from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

isend = np.arange(1 + size * rank, 1 + size * rank + size, dtype = int)
irecv = np.zeros(size, dtype = int)
print('Rank({0}) : isend = {1}'.format(rank, isend))

comm.Alltoall(isend, irecv)

print('Rank({0}) : irecv = {1}'.format(rank, irecv))

irecv = isend

comm.Alltoall(MPI.IN_PLACE, irecv)

print('Rank({0}) : irecv(IN_PLACE) = {1}'.format(rank, irecv))
