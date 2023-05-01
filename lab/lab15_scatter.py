from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

isend = np.zeros(size, dtype = int)
irecv = np.empty(1, dtype = int)

if rank == 0 :
    isend = np.arange(0, size, dtype = int)

print('sbuf : rank = {0}, irecv = {1}'.format(rank, isend))
comm.Scatter(isend, irecv, 0)
print('rbuf : rank = {0}, irecv = {1}'.format(rank, irecv))

if rank == 0 :
    isend = np.arange(10, (size + 1) * 10, 10, dtype = int)

if rank == 0 :
    irecv = isend[0]
    comm.Scatter(isend, MPI.IN_PLACE, 0)
else :
    comm.Scatter(isend, irecv, 0)

print('rbuf 2  : rank = {0}, irecv = {1}'.format(rank, irecv))
