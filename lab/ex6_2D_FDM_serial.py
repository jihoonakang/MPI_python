import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
LX = 6.0  # length of domain along x-direction
LY = 4.0  # length of domain along y-direction
EPSILON = 1e-8  # tolerance
MAX_ITER = 100000

M = 90 + 1
N = 60 + 1
dx = LX / (M - 1)
dy = LY / (N - 1)
beta = dx / dy
beta_1 = 1.0 / (2.0 * (1.0 + beta * beta))

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

    error = 0.0
    psi_old = np.zeros((N, M))

    for iter in range(0, MAX_ITER):

        psi_old = np.copy(psi_new)

        for i in range(1, N - 1):
            for j in range(1, M - 1):
                psi_new[i][j] = beta_1 * (psi_old[i][j + 1] + psi_old[i][j - 1] +
                                          beta * beta * (psi_old[i + 1][j] + psi_old[i - 1][j]))

        # Right Neumann Boundary Condition
        for i in range(N):
            psi_new[i][M - 1] = psi_new[i][M - 2]

        error = 0.0
        for i in range(N):
            for j in range(M):
                error += (psi_new[i][j] - psi_old[i][j]) * (psi_new[i][j] - psi_old[i][j])
        error = math.sqrt(error / (M * N))

        if iter % 100 == 0:
            plot(1, psi_new, -1, 101)
            print(f"Iteration = {iter}, Error = {error:.6e}")

        if error <= EPSILON:
            break

    print(f"Iteration = {iter}, Error = {error:.6e}")

def main():
    psi_new = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            psi_new[i][j] = 0.0

    # Boundary conditions
    divide = int((M - 1) * 0.5)
    for i in range(divide):
        psi_new[N - 1][i] = 0.0  # upper (left)
    for i in range(divide, M):
        psi_new[N - 1][i] = 100.0  # upper (right)
    for i in range(N):
        psi_new[i][0] = 0.0  # left wall
    for i in range(M):
        psi_new[0][i] = 0.0  # lower wall

    # Jacobi iteration
    Jacobi_iter(N, M, psi_new, beta, beta_1)

if __name__ == "__main__":
    main()