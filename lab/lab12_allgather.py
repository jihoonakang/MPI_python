from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

isend = np.array([rank + 1])
irecv = np.zeros(size, dtype = int)

print('rank = {0}, isend = {1}'.format(rank, isend))
comm.Allgather(isend, irecv)
print('rank = {0}, irecv = {1}'.format(rank, irecv))

irecv[rank] = (rank + 1) * 10
comm.Allgather(MPI.IN_PLACE, irecv)
print('rank = {0}, irecv = {1}'.format(rank, irecv))
