from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

bufsize = 20
blocksize = int(bufsize / size)
buf = np.arange(1, bufsize + 1, dtype = np.int32)
amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
fh = MPI.File.Open(comm = comm, filename = 'test2.out', amode = amode)
offset = int(rank * blocksize * MPI.INTEGER4.Get_size())
MPI.File.Write_at(fh, offset, buf[rank*blocksize:(rank + 1) * blocksize])
MPI.File.Close(fh)

amode = MPI.MODE_RDONLY
fh = MPI.File.Open(comm = comm, filename = 'test2.out', amode = amode)
filesize = MPI.File.Get_size(fh)
offset = int(filesize / size)
a = np.zeros(blocksize, dtype = np.int32)
MPI.File.Read_at(fh, rank*offset, a)
print('Rank = %d, buf = '%rank)
print(a)
MPI.File.Close(fh)

