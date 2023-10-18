from mpi4py import MPI
import numpy as np

def para_range(n1, n2, size, rank) :
    iwork = divmod((n2 - n1 + 1), size)
    ista = rank * iwork[0] + n1 + min(rank, iwork[1])
    iend = ista + iwork[0] - 1
    if iwork[1] > rank :
        iend = iend + 1
    return ista, iend

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_step = 1000000
dx = 1.0 / num_step

ista, iend = para_range(0,num_step-1, size, rank)
print('Rank = %d, (ista, iend) = (%d, %d)'%(rank, ista, iend))

sum = np.array([0],dtype=np.float64)
total_sum = np.array([0],dtype=np.float64)

for i in range(ista, iend+1) :
    x = (i + 0.5) * dx
    sum += 4.0/(1.0 + x*x)

comm.Reduce(sum,total_sum,op=MPI.SUM,root=0)
# Reduce results to rank 0
if rank == 0 :
    pi = dx * total_sum
    print('Numerical pi = %f'%pi)
