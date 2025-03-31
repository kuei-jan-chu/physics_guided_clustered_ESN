import numpy as np

def Lorenz(t0, x0, sigma, rho, beta):
    dxdt = np.matrix(np.zeros(x0.shape))
    dxdt[:,0] = sigma * (x0[:,1]-x0[:,0])
    dxdt[:,1] = x0[:,0] * (rho-x0[:,2]) - x0[:,1]
    dxdt[:,2] = x0[:,0] * x0[:,1] - beta*x0[:,2]
    return dxdt

def RK4(dxdt, x0, t0, dt, sigma, rho, beta):
    k1 = dxdt(t0,x0, sigma, rho, beta);
    k2 = dxdt(t0+dt/2.,x0+dt*k1/2., sigma, rho, beta);
    k3 = dxdt(t0+dt/2.,x0+dt*k2/2., sigma, rho, beta);
    k4 = dxdt(t0+dt,x0+dt*k3, sigma, rho, beta);
    x = x0 + 1./6*(k1+2*k2+2*k3+k4)*dt;
    return x