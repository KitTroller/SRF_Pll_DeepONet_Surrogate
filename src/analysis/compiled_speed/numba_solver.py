import numpy as np, time
from numba import njit
@njit(cache=False, fastmath=False)
def sat(u,L,b):
    return u - b*np.log1p(np.exp((u-L)/b)) + b*np.log1p(np.exp((-u-L)/b))
@njit
def dsat(u,L,b):
    return 1.0 - 1/(1+np.exp(-(u-L)/b)) - 1/(1+np.exp(-(-u-L)/b))
@njit
def park(Va,Vb,Vc,th):
    c=2*np.pi/3
    Vd=(2/3)*(Va*np.cos(th)+Vb*np.cos(th-c)+Vc*np.cos(th+c))
    Vq=-(2/3)*(Va*np.sin(th)+Vb*np.sin(th-c)+Vc*np.sin(th+c))
    return Vd,Vq
@njit
def run(Va,Vb,Vc,th0,om0,dt,Kp,Ki,L,b):
    N=Va.shape[0]; th=np.empty(N); th[0]=th0; om=om0; w0=2*np.pi*50; Kaw=Ki/Kp
    _,Vqp=park(Va[0],Vb[0],Vc[0],th0); iters=0
    for k in range(1,N):
        up=om+Kp*Vqp; sp=sat(up,L,b)
        kt=th[k-1]+dt/2*(2*w0+sp); ko=om+dt/2*(Ki*Vqp+Kaw*(sp-up))
        t=th[k-1]+dt*(w0+sp); o=om
        for it in range(10):
            Vd,Vq=park(Va[k],Vb[k],Vc[k],t); u=o+Kp*Vq
            F=t-kt-dt/2*sat(u,L,b); dF=1+dt/2*dsat(u,L,b)*Kp*Vd
            s=F/dF; t-=s
            Vd,Vq=park(Va[k],Vb[k],Vc[k],t); u=o+Kp*Vq
            o=ko+dt/2*(Ki*Vq+Kaw*(sat(u,L,b)-u)); iters+=1
            if abs(s)<1e-10: break
        th[k]=t; om=o; Vqp=Vq
    return th,iters
