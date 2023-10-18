from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

bufsize = 100
buf = np.arange(rank * bufsize, (rank + 1) * bufsize, dtype = np.int32)

amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
f = MPI.File.Open(comm = MPI.COMM_SELF, filename = 'pr%d.npy'%rank, amode = amode)
MPI.File.Write(f, buf)
MPI.File.Close(f)

a = np.zeros(bufsize, dtype = np.int32)
amode = MPI.MODE_RDONLY
f = MPI.File.Open(comm = MPI.COMM_SELF, filename = 'pr%d.npy'%rank, amode = amode)
MPI.File.Read(f, a)
MPI.File.Close(f)

print('Rank %d load data from pr%d.npy \n'%(rank,rank), a)

