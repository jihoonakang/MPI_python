from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

ircnt = np.array([1, 2, 3], dtype = int)
irecv = np.zeros(6, dtype = int)
isend = np.zeros(rank + 1, dtype = int)

for i in range(rank + 1) :
    isend[i] = rank + 1

print('Before : rank = {0}, irecv = {1}'.format(rank, isend))

comm.Allgatherv(isend, (irecv, ircnt))
if rank == 0 :
    print('After  : rank = {0}, irecv = {1}'.format(rank, irecv))

irecv[:] = 0
i = int((rank * rank + rank) / 2)
irecv[i : i + rank + 1] = (rank + 1) * 10

print('Before : rank = {0}, irecv = {1}'.format(rank, irecv))

comm.Allgatherv(MPI.IN_PLACE, (irecv, ircnt))

print('After  : rank = {0}, irecv = {1}'.format(rank, irecv))
