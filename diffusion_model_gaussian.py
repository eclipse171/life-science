import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import animation, rc
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

N = 100

D_u = 10.0

op = prep_laplacian_op(N,N)

def gaussian_func(x,y,sigma):
    return math.exp(-(x*x+y*y)/(2*sigma*sigma))/(2*math.pi*sigma*sigma)

u = np.empty((N,N))

for i in range(0,N):
    for j in range(0,N):
        u[i,j] = gaussian_func(i-N//2,j-N//2,10)

t = 0
dt = 0.005

def nextstep():
    global u,t
    for cnt in range(100):
        dudt = D_u * laplacian(u,op)
        u = u + dudt*dt
        t = t + dt

if not os.path.exists('Gaussian_images'):
    os.mkdir('Gaussian_images')

fig = plt.figure()
images = []
cnt = 0
while t<10.0:
    img = plt.imshow(u,cmap="binary")
    images.append([img])
    plt.savefig("Gaussian_images/{0:03d}.png".format(cnt))
    cnt += 1
    nextstep()

anim = animation.ArtistAnimation(fig, images, interval=200)
rc('animation', html='jshtml')
plt.show()
plt.close()