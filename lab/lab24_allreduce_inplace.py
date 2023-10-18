from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a = np.zeros(3, dtype = int)

ista = rank * 3
iend = ista + 3
a = np.arange(ista + 1, iend + 1)

sum = a.sum()
tsum = np.zeros_like(sum)
tsum += sum
comm.Allreduce(MPI.IN_PLACE, tsum, MPI.SUM)

if rank == 2 :
    print('Rank({0}) : sum = {1}'.format(rank, tsum))
