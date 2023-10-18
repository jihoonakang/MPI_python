def para_range(n1, n2, size, rank) :
    iwork = divmod((n2 - n1 + 1), size)
    ista = rank * iwork[0] + n1 + min(rank, iwork[1])
    iend = ista + iwork[0] - 1
    if iwork[1] > rank :
        iend = iend + 1

    return ista, iend

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

n = 11

a = np.zeros(n, dtype = np.int32)
b = np.zeros(n, dtype = np.int32)

ista_b, iend_b = para_range(0, n - 1, size, rank)

p_next = rank + 1; p_prev = rank - 1

if rank == size - 1 :
    p_next = MPI.PROC_NULL
if rank == 0 :
    p_prev = MPI.PROC_NULL

for i in range(ista_b, iend_b+1) :
    b[i] = i + 1

req_i1 = comm.Isend(b[iend_b:iend_b+1], p_next, 11)
req_i2 = comm.Isend(b[ista_b:ista_b+1], p_prev, 12)
req_r1 = comm.Irecv(b[ista_b-1: ista_b], p_prev, 11)
req_r2 = comm.Irecv(b[iend_b+1: iend_b+2], p_next, 12)

MPI.Request.Wait(req_i1)
MPI.Request.Wait(req_i2)
MPI.Request.Wait(req_r1)
MPI.Request.Wait(req_r2)

ista_a = ista_b; iend_a = iend_b

if rank == 0 :
    ista_a = 1
if rank == size - 1 :
    iend_a = n - 2

for i in range(ista_a, iend_a+1) :
    a[i] = b[i-1] + b[i+1]

for i in range(size) :
    if i == rank :
        print(rank)
        print(b)
        print(a)
    comm.Barrier()

