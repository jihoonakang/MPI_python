from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

ircnt = np.array([1, 2, 3], dtype = int)
isend = np.zeros(6, dtype = int)

for i in range(rank + 1) :
    isend[i] = (rank + 1) * 10

if rank == 0 :
    comm.Gatherv(MPI.IN_PLACE, (isend, ircnt), 0)
else :
    comm.Gatherv(isend[0:rank+1], 0)

if rank == 0 :
    print('rank = {0}, irecv = {1}'.format(rank, isend))
