from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

# if rank == 0 :
#     a = np.array([9, 5, 2], dtype = np.int32)
# elif rank == 1 :
#     a = np.array([8, 6, 1], dtype = np.int32)
# elif rank == 2 :
#     a = np.array([7, 4, 3], dtype = np.int32)
# pos  = np.array([rank, rank, rank], dtype = np.int32)

# max = np.zeros(3, dtype = np.int32)
# loc = np.zeros(3, dtype = np.int32)

# comm.Reduce(a, max, MPI.MAX, 0)

# if rank == 0 :
#     print('Rank({0}) : res = {1}'.format(rank, max[0]))


if rank == 0 :
    a = np.array([9, rank], dtype = np.int32)
elif rank == 1 :
    a = np.array([8, rank], dtype = np.int32)
elif rank == 2 :
    a = np.array([7, rank], dtype = np.int32)

maxloc = np.zeros(2, dtype = np.int32)

comm.Reduce((a, MPI.INT_INT), (maxloc, MPI.INT_INT) , MPI.MAXLOC, 0)

if rank == 0 :
    print('Rank({0}) : res = {1}'.format(rank, maxloc))




# comm.Reduce((a[1], pos[1], MPI.INT_INT), (max, loc, MPI.INT_INT), MPI.MAXLOC, 0)

# max = np.zeros_like(a)
# loc = np.zeros_like(pos)
# for i in range(size) :
#     comm.Reduce((a[i:i+1], pos[i:i+1], MPI.INT_INT), (max[i:i+1], loc[i:i+1],MPI.INT_INT), MPI.MAXLOC, 0)

# if rank == 0 :
#     a = [9, 5, 2]
# elif rank == 1 :
#     a = [8, 6, 1]
# elif rank == 2 :
#     a = [7, 4, 3]
# pos  = [rank, rank, rank]

# res = [[0 for j in range(2)] for i in range(3)]
# res = comm.reduce(a, MPI.MAXLOC, 0)
# if rank == 0 :
#     print('Rank({0}) : res = {1}'.format(rank, res))

# if rank == 0 :
#     a = [[9, rank], [5, rank], [2, rank]]
# elif rank == 1 :
#     a = [[8, rank], [6, rank], [1, rank]]
# elif rank == 2 :
#     a = [[7, rank], [4, rank], [3, rank]]
# pos  = [rank, rank, rank]

# res = [[0 for j in range(2)] for i in range(3)]
# for i in range(size) :
#     res[i] = comm.reduce(a[i], MPI.MAXLOC, 0)
#     if rank == 0 :
#         print('Rank({0}) : res = {1}'.format(rank, res))


