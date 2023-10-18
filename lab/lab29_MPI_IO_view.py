from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

bufsize = 100
buf = np.arange(rank * bufsize, (rank + 1) * bufsize, dtype = np.int32)

amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
f = MPI.File.Open(comm = comm, filename = 'testfile', amode = amode)
disp = rank * bufsize * MPI.INTEGER4.Get_size()
MPI.File.Set_view(f, disp, MPI.INTEGER4, MPI.INTEGER4)
MPI.File.Write(f, buf)
MPI.File.Close(f)
 
if rank == 0 :
    a = np.zeros(bufsize * size, dtype = np.int32)
    amode = MPI.MODE_RDONLY
    f = MPI.File.Open(comm = MPI.COMM_SELF, filename = 'testfile', amode = amode)
    MPI.File.Read(f, a)
    MPI.File.Close(f)
    print('Rank %d load data from testfile \n'%rank, a)
