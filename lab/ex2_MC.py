import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

SCOPE = 1000000

mycount = 0
np.random.seed(rank)

for i in range(SCOPE) :
    x = np.random.rand()
    y = np.random.rand()
    z = (x*x + y*y)**(0.5)
    if z < 1 :
        mycount += 1

# comm.Reduce(mycount, total_count, MPI.SUM, root = 0)

if rank == 0 :
    for i in range (1, size) :
        other_count = comm.recv(source=MPI.ANY_SOURCE, tag=10)
        mycount = mycount + other_count
else :
    comm.send(mycount, dest=0, tag=10)

if rank == 0 :
    print('Rank : %d, Count = %d, Pi = %f'%(rank,mycount,mycount/SCOPE/size*4))
# print('Rank = %d, Count = %d, Pi = %f'%(rank, total_sum,total_sum/(SCOPE*size)*4))
