from particle import *
import matplotlib.pyplot as plt
import pandas as pd
import vedo

# Parameters
dt = 0.01
t  = 0

# Grid
L = 10; n = 11
grid = Grid(0,L,n,0,L,n)

# Particles
p1 = Particle(ID=1, x0 = [1,5])
p2 = Particle(ID=2, x0 = [3,5], v0 = [1,1])
p3 = Particle(ID=3, x0 = [5,8], v0 = [-1,1])
p4 = Particle(ID=4, x0 = [5,8])
p5 = Particle(ID=5, x0 = [2,7])
p6 = Particle(ID=6, x0 = [2,7])
p7 = Particle(ID=7, x0 = [3,7])
p8 = Particle(ID=8, x0 = [2.1,7.1])
p9 = Particle(ID=9, x0 = [2.1,7.2])
p10 = Particle(ID=10, x0 = [2.1,7.4])
particles = {i:Particle(ID=i, x0=np.random.rand(2)*10) for i in range(50)}
#[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]

# Store results | t, x, y, z, particle 
states = []

# Plot
#axes = vedo.Axes(xrange=(-20,20), yrange=(-20,20), zrange=(-1,20))
#plt = vedo.Plotter(interactive=False)
#plt.show(axes, viewup='z')
#objvec = [vedo.Sphere(c="blue5", r=p.r, pos=p.x) for p in pvec]

force = lambda p: force_gravity(p) + force_buoyancy(p)

# Main loop
while t <= 10.0:
    # particle loop
    for p in pvec:
        verlet_update(p, force, dt)
        states.append([t, *p.state()])
    #for p,obj in zip(pvec,objvec):
    #    obj.pos(p.x)

    # show
    #plt.show(*objvec, resetcam=False, rate=15)
    
    t += dt

#plt.interactive().close()
df = pd.DataFrame(states, columns=['t', 'ID', 'x', 'y'])

dfsub = df.query("ID==3")
plt.scatter(x=dfsub.x, y=dfsub.y, c=dfsub.t)


