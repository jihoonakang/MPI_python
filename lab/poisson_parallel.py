import numpy as np
import math
import matplotlib.pyplot as plt
from mpi4py import MPI
from dataclasses import dataclass

global grid_x, grid_y

@dataclass
class mympi :
    nprocs : int
    myrank : int
    nx_mpi : int
    ny_mpi : int
    mpisize_x : int
    mpisize_y : int
    mpirank_x : int
    mpirank_y : int
    w_rank : int
    e_rank : int
    n_rank : int
    s_rank : int
    comm_x : MPI.Comm
    comm_y : MPI.Comm

def mpi_setup(nx, ny, mpi_info) :

    mpi_info.nprocs = MPI.COMM_WORLD.Get_size()
    mpi_info.myrank = MPI.COMM_WORLD.Get_rank()

    mpi_info.mpisize_x = 2
    mpi_info.mpisize_y = 4
    mpi_info.nx_mpi = int(nx / mpi_info.mpisize_x)
    mpi_info.ny_mpi = int(ny / mpi_info.mpisize_y)

    dimsize = [mpi_info.mpisize_x, mpi_info.mpisize_y]
    periods = [False, True]
    reorder = 0
    comm_cart = MPI.COMM_WORLD.Create_cart(dimsize, periods, reorder)

    coords = comm_cart.Get_coords(mpi_info.myrank)
    mpi_info.mpirank_x = coords[0]
    mpi_info.mpirank_y = coords[1]

    remain = [False, True]
    mpi_info.comm_y = comm_cart.Sub(remain)
    mpi_info.s_rank, mpi_info.n_rank = mpi_info.comm_y.Shift(0, 1)

    remain = [True, False]
    mpi_info.comm_x = comm_cart.Sub(remain)
    mpi_info.w_rank, mpi_info.e_rank = mpi_info.comm_x.Shift(0, 1)

def send_east(u, nx_mpi, ny_mpi, mpi_info) :

    if mpi_info.e_rank >= 0 :
        req_send = mpi_info.comm_x.Isend(u[nx_mpi, 1:ny_mpi+1], dest = mpi_info.e_rank, tag = 101)

    if mpi_info.w_rank >= 0 :
        req_recv = mpi_info.comm_x.Irecv(u[0, 1:ny_mpi+1], source = mpi_info.w_rank, tag = 101)

    if mpi_info.e_rank >= 0 :
        MPI.Request.Wait(req_send)

    if mpi_info.w_rank >= 0 :
        MPI.Request.Wait(req_recv)

def send_west(u, nx_mpi, ny_mpi, mpi_info) :

    if mpi_info.w_rank >= 0 :
        req_send = mpi_info.comm_x.Isend(u[1, 1:ny_mpi+1], dest = mpi_info.w_rank, tag = 102)

    if mpi_info.e_rank >= 0 :
        req_recv = mpi_info.comm_x.Irecv(u[nx_mpi+1, 1:ny_mpi+1], source = mpi_info.e_rank, tag = 102)

    if mpi_info.w_rank >= 0 :
        MPI.Request.Wait(req_send)

    if mpi_info.e_rank >= 0 :
        MPI.Request.Wait(req_recv)

def send_north(u, nx_mpi, ny_mpi, mpi_info) :

    sbuf = np.empty(nx_mpi, dtype = 'd')
    rbuf = np.empty(nx_mpi, dtype = 'd')

    sbuf[0:nx_mpi] = u[1:nx_mpi+1, ny_mpi]
    req_send = mpi_info.comm_y.Isend(sbuf, dest = mpi_info.n_rank, tag = 103)
    req_recv = mpi_info.comm_y.Irecv(rbuf, source = mpi_info.s_rank, tag = 103)

    MPI.Request.Wait(req_send)
    MPI.Request.Wait(req_recv)

    u[1:nx_mpi+1, 0] = rbuf[0:nx_mpi]

def send_south(u, nx_mpi, ny_mpi, mpi_info) :

    sbuf = np.empty(nx_mpi, dtype = 'd')
    rbuf = np.empty(nx_mpi, dtype = 'd')

    sbuf[0:nx_mpi] = u[1:nx_mpi+1, 1]
    req_send = mpi_info.comm_y.Isend(sbuf, dest = mpi_info.s_rank, tag = 104)
    req_recv = mpi_info.comm_y.Irecv(rbuf, source = mpi_info.n_rank, tag = 104)

    MPI.Request.Wait(req_send)
    MPI.Request.Wait(req_recv)

    u[1:nx_mpi+1, ny_mpi+1] = rbuf[0:nx_mpi]

def RB_gauss_seidel(dx, dy, nx, ny, u_solve, rhs, maxiteration, tolerance, mpi_info) :
    
    dxsqi = 1.0 / dx / dx
    dysqi = 1.0 / dy / dy
    
    for iter in range(maxiteration) :
        error_sum = 0.0
        u_sum = 0.0
        
        if mpi_info.w_rank < 0 :
            for j in range(1, ny + 1) :
                u_solve[0, j]    = -u_solve[1, j]

        if mpi_info.e_rank < 0 :
            for j in range(0, ny + 2) :
                u_solve[nx+1, j] = -u_solve[nx, j]

        send_east(u_solve, nx, ny, mpi_info)
        send_west(u_solve, nx, ny, mpi_info)
        send_north(u_solve, nx, ny, mpi_info)
        send_south(u_solve, nx, ny, mpi_info)

        for i in range(1, nx + 1) :
            js = 2 if iter%2 == i%2 else 1
            for j in range(js, ny + 1, 2) :
                a_w = dxsqi * u_solve[i - 1, j]
                a_e = dxsqi * u_solve[i + 1, j]
                a_s = dysqi * u_solve[i, j - 1]
                a_n = dysqi * u_solve[i, j + 1]

                a_p = 2.0 * (dxsqi + dysqi)

                uij_old = u_solve[i, j]
                uij_new = (a_w + a_e + a_s + a_n - rhs[i, j]) / a_p

                error_sum += abs(uij_new - uij_old)
                u_sum += abs(uij_new)

                u_solve[i, j] = uij_new

        error_sum_global = 0.0
        u_sum_global = 0.0

        error_sum_global = MPI.COMM_WORLD.allreduce(error_sum, MPI.SUM)
        u_sum_global = MPI.COMM_WORLD.allreduce(u_sum, MPI.SUM)

        if iter % 10 == 0 :
            if mpi_info.myrank == 0 :
                print('{0} th iteration : relative error = {1}'.format(iter, error_sum_global/u_sum_global))
                u_all = np.zeros((grid_x, grid_y), dtype = 'd')
                u_all[0:mpi_info.nx_mpi, 0:mpi_info.ny_mpi] = u_solve[1:mpi_info.nx_mpi+1, 1:mpi_info.ny_mpi+1]

                for i in range(1, mpi_info.nprocs) :
                    rbuf = np.zeros((mpi_info.nx_mpi + 2, mpi_info.ny_mpi + 2), dtype = 'd')
                    MPI.COMM_WORLD.Recv(rbuf, i, i)
                    ista = int((i/mpi_info.mpisize_y)) * mpi_info.nx_mpi
                    iend = ista + mpi_info.nx_mpi
                    jsta = int((i%mpi_info.mpisize_y) * mpi_info.ny_mpi)
                    jend = jsta + mpi_info.ny_mpi

                    u_all[ista:iend, jsta:jend] = rbuf[1:mpi_info.nx_mpi+1, 1:mpi_info.ny_mpi+1]

                plt.figure(2)
                plt.clf()
                plt.pcolormesh(u_all, cmap=plt.cm.jet, vmin=-0.25, vmax=0.25)
                plt.colorbar()
                plt.draw()
                plt.pause(0.1)
            else :
                MPI.COMM_WORLD.Send(u_solve, 0, mpi_info.myrank)

        if (error_sum_global/u_sum_global) < tolerance :
            break
    # plt.savefig('u_solve.png', dpi = 300)
    # print('Iteration ends in {0} th step'.format(iter))

grid_x = 32
grid_y = 32
len_x = 1.0
len_y = 1.0

maxiter = 1000
tol = 1.0e-8

mpi_info = mympi
mpi_setup(grid_x, grid_y, mpi_info)

grid_size_mpi = (mpi_info.nx_mpi + 2) * (mpi_info.ny_mpi + 2)
dx = len_x / grid_x
dy = len_y / grid_y

pos_x = np.zeros(mpi_info.nx_mpi + 2, dtype = 'd')
pos_y = np.zeros(mpi_info.ny_mpi + 2, dtype = 'd')
u_exact = np.zeros((mpi_info.nx_mpi + 2, mpi_info.ny_mpi + 2), dtype = 'd')
u_solve = np.zeros((mpi_info.nx_mpi + 2, mpi_info.ny_mpi + 2), dtype = 'd')
rhs = np.zeros((mpi_info.nx_mpi + 2, mpi_info.ny_mpi + 2), dtype = 'd')

for i in range(mpi_info.nx_mpi + 2) :
    pos_x[i] = (i - 0.5 + mpi_info.mpirank_x * mpi_info.nx_mpi) * dx

for j in range(mpi_info.ny_mpi + 2) :
    pos_y[j] = (j - 0.5 + mpi_info.mpirank_y * mpi_info.ny_mpi) * dy

for i in range(1, mpi_info.nx_mpi + 1) :
    x_val = pos_x[i] * (1.0 - pos_x[i])
    for j in range(1, mpi_info.ny_mpi + 1) :
        y_val = math.cos(2.0 * math.pi * pos_y[j])
        u_exact[i, j] = x_val * y_val
        u_solve[i, j] = 0.0
        rhs[i, j] = -2.0 * y_val - 4.0 * math.pi * math.pi * x_val * y_val

# plt.figure(1)
# plt.pcolormesh(u_exact, cmap=plt.cm.jet, vmin=-0.25, vmax=0.25)
# plt.colorbar()
# plt.draw()
# plt.pause(0.1)
# plt.savefig('u_exact.png', dpi = 300)

RB_gauss_seidel(dx,dy,mpi_info.nx_mpi,mpi_info.ny_mpi,u_solve,rhs,maxiter,tol,mpi_info)

