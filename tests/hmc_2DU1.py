# %%
import torch
import math
import sys
import os
from timeit import default_timer as timer
from functools import reduce
import matplotlib.pyplot as plt

class Param:
    def __init__(self, beta = 6.0, lat = [64, 64], tau = 2.0, nstep = 50, ntraj = 256, nrun = 4, nprint = 256, seed = 11*13, randinit = False, nth = int(os.environ.get('OMP_NUM_THREADS', '2')), nth_interop = 2):
        self.beta = beta
        self.lat = lat
        self.nd = len(lat)
        self.volume = reduce(lambda x,y:x*y, lat)
        self.tau = tau
        self.nstep = nstep
        self.dt = self.tau / self.nstep
        self.ntraj = ntraj
        self.nrun = nrun
        self.nprint = nprint
        self.seed = seed
        self.randinit = randinit
        self.nth = nth
        self.nth_interop = nth_interop
    def initializer(self):
        if self.randinit:
            return torch.empty([param.nd] + param.lat).uniform_(-math.pi, math.pi)
        else:
            return torch.zeros([param.nd] + param.lat)
    def summary(self):
        return f"""latsize = {self.lat}
volume = {self.volume}
beta = {self.beta}
trajs = {self.ntraj}
tau = {self.tau}
steps = {self.nstep}
seed = {self.seed}
nth = {self.nth}
nth_interop = {self.nth_interop}
"""
    def uniquestr(self):
        lat = ".".join(str(x) for x in self.lat)
        return "test.out"

def action(param, f):
    return (-param.beta)*torch.sum(torch.cos(plaqphase(f)))
def force(param, f):
    f.requires_grad_(True)
    s = action(param, f)
    s.backward()
    ff = f.grad
    f.requires_grad_(False)
    return ff

plaqphase = lambda f: f[0,:] - f[1,:] - torch.roll(f[0,:], shifts=-1, dims=1) + torch.roll(f[1,:], shifts=-1, dims=0)
topocharge = lambda f: torch.floor(0.1 + torch.sum(regularize(plaqphase(f))) / (2*math.pi))
def regularize(f):
    p2 = 2*math.pi
    f_ = (f - math.pi) / p2
    return p2*(f_ - torch.floor(f_) - 0.5)

def leapfrog(param, x, p):
    dt = param.dt
    x_ = x + 0.5*dt*p
    p_ = p + (-dt)*force(param, x_)
    for i in range(param.nstep-1):
        x_ = x_ + dt*p_
        p_ = p_ + (-dt)*force(param, x_)
    x_ = x_ + 0.5*dt*p_
    return (x_, p_)
def hmc(param, x):
    p = torch.randn_like(x)
    act0 = action(param, x) + 0.5*torch.sum(p*p)
    x_, p_ = leapfrog(param, x, p)
    xr = regularize(x_)
    act = action(param, xr) + 0.5*torch.sum(p_*p_)
    prob = torch.rand([], dtype=torch.float64)
    dH = act-act0
    exp_mdH = torch.exp(-dH)
    acc = prob < exp_mdH
    newx = xr if acc else x
    return (dH, exp_mdH, acc, newx)

put = lambda s: sys.stdout.write(s)

if __name__ == '__main__':
    param = Param(
        beta = 6.0, # 4.0
        lat = [64, 64], # [8, 8]
        tau = 2, # 0.3
        nstep = 50, # 3
        ntraj = 1024, # 2**16 # 2**10 # 2**15
        nprint = 256,
        seed = 1984)

    torch.manual_seed(param.seed)

    torch.set_num_threads(param.nth)
    torch.set_num_interop_threads(param.nth_interop)
    os.environ["OMP_NUM_THREADS"] = str(param.nth)
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

    torch.set_default_tensor_type(torch.DoubleTensor)

    def run(param):
        with open(param.uniquestr(), "w") as O:
            params = param.summary()
            O.write(params)
            put(params)
            field = param.initializer() # mu, x, t
            plaq, topo = (action(param, field) / (-param.beta*param.volume), topocharge(field))
            status = f"Initial configuration:  plaq: {plaq}  topo: {topo}\n"
            O.write(status)
            put(status)
            ts = []
            # Initialize list to store topological charge history
            topo_history = [topo.item()]
            
            for n in range(param.nrun):
                t = -timer()
                for i in range(param.ntraj):
                    dH, exp_mdH, acc, field = hmc(param, field)
                    plaq = action(param, field) / (-param.beta*param.volume)
                    topo = topocharge(field)
                    # Append each new topological charge
                    topo_history.append(topo.item())
                    ifacc = "ACCEPT" if acc else "REJECT"
                    status = f"Traj: {n*param.ntraj+i+1:4}  {ifacc}:  dH: {dH:< 12.8}  exp(-dH): {exp_mdH:< 12.8}  plaq: {plaq:< 12.8}  topo: {topo:< 3.3}\n"
                    O.write(status)
                    if (i+1) % (param.ntraj//param.nprint) == 0:
                        put(status)
                t += timer()
                ts.append(t)
            print("Run times: ", ts)
            print("Per trajectory: ", [t/param.ntraj for t in ts])
            return topo_history
    topo_history = run(param)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    iterations = range(len(topo_history))
    plt.plot(iterations, topo_history, 'b.-', markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Topological Charge')
    plt.title(f'Topological Charge Evolution (β={param.beta}, L={param.lat[0]})')
    plt.grid(True)
    plt.savefig('topo_charge_history.png')
    plt.show()
# %%
