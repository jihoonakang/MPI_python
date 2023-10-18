from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.array([1, 2, 3], dtype = int)
recvbuf = np.zeros(1, dtype = int)
RECVBUF = sendbuf * 2
print('Rank({0}) : sendbuf = {1}'.format(rank, sendbuf))

comm.Reduce_scatter(sendbuf, recvbuf, None, MPI.SUM)

print('Rank({0}) : recvbuf = {1}'.format(rank, recvbuf))

comm.Reduce_scatter(MPI.IN_PLACE, RECVBUF, [1, 1, 1], MPI.SUM)

print('Rank({0}) : RECVBUF = {1}'.format(rank, RECVBUF))
