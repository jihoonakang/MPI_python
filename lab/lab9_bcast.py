from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

ROOT = 0

buf =  np.zeros(4, dtype = int)
buf2 = np.zeros(4, dtype = int)

if rank == ROOT :
    buf = np.array([5, 6, 7, 8])

if rank == (size - 1) :
    buf2 = np.array([50, 60, 70, 80])

print('Before : rank = {0}, buf = {1}'.format(rank, buf))

comm.Bcast(buf, ROOT)
req = comm.Ibcast(buf2, size - 1)

MPI.Request.Wait(req)
print('After  : rank = {0}, buf = {1}'.format(rank, buf))
print('After  : rank = {0}, buf2 = {1}'.format(rank, buf2))
