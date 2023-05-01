import numpy as np
import random as rd
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
NP = 5
matrixA = np.zeros((NP, NP), dtype = np.int32)
matrixArow = np.zeros( NP, dtype = np.int32)

matrixB = np.zeros((NP, NP), dtype = np.int32)
matrixC = np.zeros(NP, dtype = np.int32)
matrixT = np.zeros((NP, NP), dtype = np.int32)

# only for rank 0
if rank == 0 :
    for i in range(NP) :
        for j in range(NP) :
            matrixA[i][j] = rd.randrange(1, 10)
            matrixB[i][j] = rd.randrange(1, 10)

    print(matrixA)
    print(matrixB)

# Broadcast A and B
comm.Scatter(matrixA, matrixArow, root=0)
comm.Bcast(matrixB,root=0)

for j in range(NP) :
    for i in range(NP) :
        matrixC[j] = matrixC[j] + matrixArow[i] * matrixB[i][j]

comm.Gather(matrixC, matrixT, root=0)

if rank == 0:
    print('Matrix T =')
    print(matrixT)
    print('Matrix A * B = ')
    print(matrixA@matrixB)

