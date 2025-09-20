import numpy as np

def Lorenz96(t,x0, F, alpha):
    dxdt = np.matrix(np.zeros(x0.shape))
    end = x0.shape[1]
    dxdt[:,2:end-1] = -alpha*np.multiply(x0[:,0:end-3],x0[:,1:end-2]) + alpha*np.multiply(x0[:,1:end-2],x0[:,3:end]) - x0[:,2:end-1]+F
    dxdt[:,0] = -alpha*np.multiply(x0[:,end-2],x0[:,end-1])+alpha*np.multiply(x0[:,end-1],x0[:,1])-x0[:,0]+F
    dxdt[:,1] = -alpha*np.multiply(x0[:,end-1],x0[:,0])+alpha*np.multiply(x0[:,0],x0[:,2])-x0[:,1]+F
    dxdt[:,end-1] = -alpha*np.multiply(x0[:,end-3],x0[:,end-2])+alpha*np.multiply(x0[:,end-2],x0[:,0])-x0[:,end-1]+F
    return dxdt

def RK4(dxdt, x0, t0, dt, F, alpha):
    k1 = dxdt(t0,x0, F, alpha);
    k2 = dxdt(t0+dt/2.,x0+dt*k1/2., F, alpha);
    k3 = dxdt(t0+dt/2.,x0+dt*k2/2., F, alpha);
    k4 = dxdt(t0+dt,x0+dt*k3, F, alpha);
    x = x0 + 1./6*(k1+2*k2+2*k3+k4)*dt;
    return x




