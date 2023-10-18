import numpy as np
import math
import matplotlib.pyplot as plt

def RB_gauss_seidel(dx, dy, grid_x, grid_y, u_solve, rhs, maxiteration, tolerance) :
    
    dxsqi = 1.0 / dx / dx
    dysqi = 1.0 / dy / dy
    
    for iter in range(maxiteration) :
        error_sum = 0.0
        u_sum = 0.0
        
        for i in range(1, grid_x + 1) :
            u_solve[i, 0]        = u_solve[i, grid_y]
            u_solve[i, grid_y+1] = u_solve[i, 1]

        for j in range(1, grid_y + 1) :
            u_solve[0, j]        = -u_solve[1, j]
            u_solve[grid_x+1, j] = -u_solve[grid_x, j]
            
        for i in range(1, grid_x + 1) :
            js = 2 if iter%2 == i%2 else 1
            for j in range(js, grid_y + 1, 2) :
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

        if iter % 1 == 0 :
            print('{0} th iteration : relative error = {1}'.format(iter, error_sum/u_sum))
            plt.figure(2)
            plt.clf()
            plt.pcolormesh(u_solve, cmap=plt.cm.jet, vmin=-0.25, vmax=0.25)
            plt.colorbar()
            plt.draw()
            plt.pause(0.1)
        if (error_sum/u_sum) < tolerance :
            break
    plt.savefig('u_solve.png', dpi = 300)
    print('Iteration ends in {0} th step'.format(iter))

grid_x = 8
grid_y = 8
len_x = 1.0
len_y = 1.0

maxiter = 10000
tol = 1.0e-12

grid_size = (grid_x + 2) * (grid_y + 2)
dx = len_x / grid_x
dy = len_y / grid_y

pos_x = np.zeros(grid_x + 2, dtype = 'd')
pos_y = np.zeros(grid_y + 2, dtype = 'd')
u_exact = np.zeros((grid_x + 2, grid_y + 2), dtype = 'd')
u_solve = np.zeros((grid_x + 2, grid_y + 2), dtype = 'd')
rhs = np.zeros((grid_x + 2, grid_y + 2), dtype = 'd')

for i in range(grid_x + 2) :
    pos_x[i] = (i - 0.5) * dx

for j in range(grid_y + 2) :
    pos_y[j] = (j - 0.5) * dy

for i in range(1, grid_x + 1) :
    x_val = pos_x[i] * (1.0 - pos_x[i])
    for j in range(1, grid_y + 1) :
        y_val = math.cos(2.0 * math.pi * pos_y[j])
        u_exact[i, j] = x_val * y_val
        u_solve[i, j] = 0.0
        rhs[i, j] = -2.0 * y_val - 4.0 * math.pi * math.pi * x_val * y_val

plt.figure(1)
plt.pcolormesh(u_exact, cmap=plt.cm.jet, vmin=-0.25, vmax=0.25)
plt.colorbar()
plt.draw()
plt.pause(0.1)
plt.savefig('u_exact.png', dpi = 300)

RB_gauss_seidel(dx,dy,grid_x,grid_y,u_solve,rhs,maxiter,tol)
