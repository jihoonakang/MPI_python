from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

bufsize = 100
buf = np.arange(rank * bufsize, (rank + 1) * bufsize, dtype = np.int32)

with open('pr%d.npy'%rank, 'wb') as f:
    np.save(f, buf)

with open('pr%d.npy'%rank, 'rb') as f:
    a = np.load(f)
    print('Rank %d load data from pr%d.npy \n'%(rank,rank), a)

