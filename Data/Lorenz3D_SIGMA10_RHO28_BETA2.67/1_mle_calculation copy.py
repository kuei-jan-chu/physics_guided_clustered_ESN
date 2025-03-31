import os
import sys
import numpy as np
import pickle
sys.path.insert(0, "../Utils")
from lorenz import *

# Plotting parameters
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
plt.rcParams['xtick.major.pad']='20'
plt.rcParams['ytick.major.pad']='20'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

sigma = 10
rho = 28
beta = 8./3
DIM = 3

T=20
dt=0.01
epsilon=1e-8    # the perturbation

base_path = "."
save_path = base_path + "/Figures/"
os.makedirs(save_path, exist_ok=True)

file_name = base_path + "/Simulation_Data/lorenz3D_data.pickle"
with open(file_name, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["X"]
    dt = data["dt"]

# fix random seed for reproducibility
np.random.seed(100)

# time steps number of input sequence
N_ICS = np.shape(u)[0]

# points number used for calculating the average distance between 2 points in the attractor
N_MED=100

# times number for calculating the MLE used to be averaged finally
NUM_ICS=10

# time steps number used for propogate purturbation
N=int(T/dt)

# COMPUTING THE EXPECTED VALUE OF THE DISTANCE BETWEEN TWO POINTS IN THE ATTRACTOR
expDist=[]
for i in range(N_MED):
    print("MEAN EXPECTED DISTANCE IC {:}/{:}, {:2.3f}%".format(i+1, N_MED, (i+1)/N_MED*100), end="\r")
    for j in range(N_MED):
        ici = np.random.randint(0,N_ICS)
        icj = np.random.randint(0,N_ICS)
        dist_=np.linalg.norm(u[ici]-u[icj])
        expDist.append(dist_)

expDist=np.array(expDist)
mexpDist=expDist.mean()
print("\nMEAN EXPECTED DISTANCE: {:}\n".format(mexpDist))

logMexpDist=np.log(mexpDist)

logabs_error_evol_all=[]
slopes_all=[]
for i in range(NUM_ICS):
    print("IC = {:}/{:}, {:2.3f}%".format(i+1, NUM_ICS, (i+1)/NUM_ICS*100), end="\r")
    ic = np.random.randint(0,N_ICS)
    X0=np.reshape(u[ic], (1,-1))

    size_=np.shape(X0)
    X0_pert=X0+np.random.normal(0, epsilon, size=size_)
    
    logabs_error_evol=[]
    # Generate time series
    for i in range(N):
        X0 = RK4(Lorenz, X0, 0, dt, sigma, rho, beta);
        X0_pert =RK4(Lorenz, X0_pert, 0, dt, sigma, rho, beta);
        abs_error = np.linalg.norm(X0-X0_pert)
        logabs_error_evol.append(np.log(abs_error))

    # get the indices in logabs_error_evol where the values are less than 90% of the shreshold: logMexpDist. np.where returns a tuple of arrays, 
    # with each array representing indices for that dimension. In this case, since logabs_error_evol is 1D, it returns a tuple with a single array of indices.
    idx = np.where(logabs_error_evol<0.9*logMexpDist)
    # idx[0] accesses the array of indices. idx[0][-1] then selects the last index in this array, effectively giving you the last position where the condition is met.
    idx = idx[0][-1]
    # get a linear region of error growth.
    Y=logabs_error_evol[:idx]
    # Y (log of error values) and X (time values) are used to fit a line to the truncated error growth region.
    # The slope of this line is calculated using the formula for linear regression and represents the Maximum Lyapunov Exponent.
    X=np.linspace(0, len(Y)*dt, len(Y))
    Y=np.array(Y)
    X=np.array(X)
    slope=((X*Y).mean() - X.mean()*Y.mean()) / ((X**2).mean() - (X.mean())**2)

    logabs_error_evol=np.array(logabs_error_evol)
    logabs_error_evol_all.append(logabs_error_evol)
    slopes_all.append(slope)

logabs_error_evol_all=np.array(logabs_error_evol_all)
slopes_all=np.array(slopes_all)

LE=np.mean(slopes_all)
LT=1/LE

print("\nMAXIMUM LYAPUNOV EXPONENT = {:}".format(LE))
print("\nLYAPUNOV TIME = {:}".format(LT))

# draw the log abs error propogation
time_axis=np.linspace(0, T, N)
log_val_error_0=np.mean(logabs_error_evol_all[:,0])     # the start point log abs error for all NUM_ICS times
plt.plot(time_axis, logabs_error_evol_all.T)
plt.plot(time_axis, np.ones_like(time_axis)*logMexpDist, "--om")
plt.plot(time_axis, log_val_error_0+time_axis*LE, "--k")
plt.title("LE={:}".format(LE))
plt.savefig(save_path + "LYAPUNOV_EXPONENT", dpi = 300)
# plt.show()
plt.close()


data["MLE"] = LE
with open(base_path + "/Simulation_Data/lorenz3D_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
del data



