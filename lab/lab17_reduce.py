from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

a = np.zeros(9, dtype = int)

ista = rank * 3
iend = ista + 3

a[ista:iend] = np.arange(ista + 1, iend + 1)
sum = a.sum()
print('Rank({0}) : local_sum = {1}'.format(rank, sum))

tsum = np.zeros_like(sum)

comm.Reduce(sum, tsum, MPI.SUM, 0)

if rank == 0 :
    print('Rank({0}) : sum = {1}'.format(rank, tsum))

tsum = np.zeros_like(sum)

if rank == 0 :
    tsum += sum
    comm.Reduce(MPI.IN_PLACE, tsum, MPI.SUM, 0)
    print('Rank({0}) : sum = {1}'.format(rank, tsum))
else :
    comm.Reduce(sum, tsum, MPI.SUM, 0)
