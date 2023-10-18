import numpy as np
import matplotlib.pyplot as plt
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
LX = 6.0  # length of domain along x-direction
LY = 4.0  # length of domain along y-direction
EPSILON = 1e-8  # tolerance
MAX_ITER = 1000

M = 90 + 1
N = 60 + 1
dx = LX / (M - 1)
dy = LY / (N - 1)
beta = dx / dy
beta_1 = 1.0 / (2.0 * (1.0 + beta * beta))

def para_range(n1, n2, size, rank) :
    iwork = divmod((n2 - n1 + 1), size)
    ista = rank * iwork[0] + n1 + min(rank, iwork[1])
    iend = ista + iwork[0] - 1
    if iwork[1] > rank :
        iend = iend + 1
    return ista, iend

def gather_data(psi, psi_all) :

    ista, iend = para_range(0, N - 1, size, rank)
    domain_size = iend - ista + 1
    psi_tmp = np.array(psi[1:domain_size + 1,:])
    cound_send = np.array(comm.allgather(domain_size))
    comm.Gatherv(psi_tmp, recvbuf = (psi_all, cound_send*M), root = 0)

def plot(figID, psi, zmin, zmax):
    plt.figure(figID)
    plt.clf()
    plt.pcolormesh(psi, cmap=plt.cm.jet, vmin=zmin, vmax=zmax)
    plt.colorbar()
    plt.draw()
    plt.xlim([0, M])
    plt.ylim([0, N])
    plt.pause(0.1)

def Jacobi_iter(N, M, psi_new, beta, beta_1):
    ista, iend = para_range(0, N - 1, size, rank)
    domain_size = iend - ista + 1

    i_s = 1
    i_e = domain_size
    inext = rank + 1
    iprev = rank - 1

    if rank == 0:
        i_s = 2
        iprev = MPI.PROC_NULL
    elif rank == size - 1:
        i_e = domain_size - 1
        inext = MPI.PROC_NULL

    error = 0.0
    error_local = 0.0
    psi_old = np.zeros((domain_size + 2, M))

    for iter in range(0, MAX_ITER):

        for i in range(1, domain_size + 1):
            for j in range(M):
                psi_old[i][j] = psi_new[i][j]

        # MPI Communication
        reqs1 = []
        reqs2 = []

        if inext != MPI.PROC_NULL:
            reqs1.append(comm.Isend(psi_old[domain_size], inext, tag=1))
            reqs1.append(comm.Irecv(psi_old[domain_size + 1], inext, tag=2))

        if iprev != MPI.PROC_NULL:
            reqs2.append(comm.Isend(psi_old[1], iprev, tag=2))
            reqs2.append(comm.Irecv(psi_old[0], iprev, tag=1))

        if inext != MPI.PROC_NULL:
            for req in reqs1:
                MPI.Request.Wait(req)

        if iprev != MPI.PROC_NULL:
            for req in reqs2:
                MPI.Request.Wait(req)

        # MPI Communication

        for i in range(i_s, i_e + 1):
            for j in range(1, M - 1):
                psi_new[i][j] = beta_1 * (psi_old[i][j + 1] + psi_old[i][j - 1] +
                                          beta * beta * (psi_old[i + 1][j] + psi_old[i - 1][j]))

        # Right Neumann Boundary Condition
        for i in range(1, domain_size + 1):
            psi_new[i][M - 1] = psi_new[i][M - 2]

        error_local = 0.0
        error = 0.0
        for i in range(1, domain_size + 1):
            for j in range(M):
                error_local += (psi_new[i][j] - psi_old[i][j]) * (psi_new[i][j] - psi_old[i][j])
        error = comm.allreduce(error_local, op=MPI.SUM)
        error = math.sqrt(error / (M * N))

        psi_all = np.zeros((N, M))

        if iter % 100 == 0:
            gather_data(psi_new, psi_all)

            if rank == 0:
                plot(1, psi_all, -1, 101)
                print(f"Iteration = {iter}, Error = {error:.6e}",flush = True)

        if error <= EPSILON:
            break

    if rank == 0:
        print(f"Iteration = {iter}, Error = {error:.6e}")

def main():

    ista, iend = para_range(0, N - 1, size, rank)
    domain_size = iend - ista + 1

    psi_new = np.zeros((domain_size + 2, M))

    for i in range(1, domain_size + 1):
        for j in range(M):
            psi_new[i][j] = 0.0

    # Boundary conditions
    divide = int((M - 1) * 0.5)

    if rank == size - 1:
        for i in range(divide):
            psi_new[domain_size][i] = 0.0  # upper (left)
        for i in range(divide, M):
            psi_new[domain_size][i] = 100.0  # upper (right)
    for i in range(1, domain_size + 1):
        psi_new[i][0] = 0.0  # left wall
    if rank == 0 :
        for i in range(M):
            psi_new[1][i] = 0.0  # bottom wall

    # Jacobi iteration
    Jacobi_iter(N, M, psi_new, beta, beta_1)

if __name__ == "__main__":
    main()