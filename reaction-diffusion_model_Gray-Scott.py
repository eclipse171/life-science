import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import animation, rc, cm
import os

def prep_laplacian_op(n,m):
    A = np.zeros((n,m))
    for p in range(0,n):
        xi = 2*math.pi*p/n if p<=n//2 else 2*math.pi*(p-n)/n
        for q in range(0,m):
            eta = 2*math.pi*q/m if q<=m//2 else 2*math.pi*(q-m)/m
            A[p,q] = -xi*xi-eta*eta
    return A
            
def laplacian(a, op):
    a2 = np.fft.fft2(a)
    a2 = op * a2
    return np.fft.ifft2(a2).real

###

N = 100

p_a = 0.02
p_b = 0.05
p_c = 0.4

D_u = 0.2
D_v = 0.1

op = prep_laplacian_op(N,N)

u = np.random.uniform(low=0.0, high=1.0, size=(N,N))
v = np.random.uniform(low=0.0, high=1.0, size=(N,N))
t = 0
dt = 0.005

def nextstep():
    global u,v,t
    for cnt in range(100):
        dudt = D_u * laplacian(u,op) - u*v*v + p_a*(1.0-u)
        dvdt = D_v * laplacian(v,op) + u*v*v - (p_a+p_b)*v
        u = u + dudt*dt
        v = v + dvdt*dt
        t = t + dt

if not os.path.exists('Gray-Scott_images'):
    os.mkdir('Gray-Scott_images')

fig = plt.figure()
images = []
cnt = 0
while t<10.0:
    img = plt.imshow(u,cmap="binary")
    images.append([img])
    plt.savefig("Gray-Scott_images/{0:03d}.png".format(cnt))
    cnt += 1
    nextstep()

anim = animation.ArtistAnimation(fig, images, interval=200)
rc('animation', html='jshtml')
plt.show()
plt.close()