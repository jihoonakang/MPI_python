from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

irecv = np.zeros(rank + 1, dtype = int)
iscnt = np.array([1, 2, 3])

if rank == 0 :
    isend = np.array([1, 2, 2, 3, 3, 3], dtype = int)
else :
    isend = np.zeros(6, dtype = int)

comm.Scatterv([isend, iscnt], irecv, 0)

print('After 1 : rank = {0}, irecv = {1}'.format(rank, irecv))

if rank == 0 :
    isend = isend * 10

if rank == 0 :
    irecv = isend[0]
    comm.Scatterv([isend, iscnt], MPI.IN_PLACE, 0)
else :
    comm.Scatterv([isend, iscnt], irecv, 0)

print('After 2 : rank = {0}, irecv = {1}'.format(rank, irecv))

if rank == 0 :
    isend = isend * 10

if rank == 0 :
    comm.Scatterv([isend, iscnt], MPI.IN_PLACE, 0)
else :
    i = int((rank * rank + rank) / 2)
    comm.Scatterv([isend, iscnt], isend[i:i + rank + 1], 0)

print('After 3 : rank = {0}, isend = {1}'.format(rank, isend))
