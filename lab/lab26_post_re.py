from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

bufsize = 100
buf = np.arange(rank * bufsize, (rank + 1) * bufsize, dtype = np.int32)

if rank != 0 :
    comm.Send(buf, dest = 0, tag = 55)

elif rank ==0 :
    with open('pr.npy', 'wb') as f:
        np.save(f, buf)

        for i in range(1, size) :
            status = MPI.Status()
            comm.Recv(buf, source = i, tag = 55, status = status)
            np.save(f, buf)

if rank == 0 :
    with open('pr.npy', 'rb') as f:
        for i in range(0, size) :
            a = np.load(f)
            print(a)

