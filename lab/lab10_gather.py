from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

isend = np.array([rank + 1], dtype = int)
irecv = np.zeros(size, dtype = int)

print('rank = {0}, isend = {1}'.format(rank, isend))
ROOT = 0
comm.Gather(isend, irecv, ROOT)
print('rank = {0}, irecv = {1}'.format(rank, irecv))

# MPI.IN_PLACE
recvdata = np.zeros(4, dtype = int)

isend[0] = isend[0] * 10

if rank == (size - 1) :
    recvdata[size - 1] = isend[0]
    comm.Gather(MPI.IN_PLACE, recvdata, size - 1)
else :
    comm.Gather(isend, recvdata, size - 1)

print('rank = {0}, recvdata = {1}'.format(rank, recvdata))
