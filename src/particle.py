# particle.py
import numpy as np
from numpy import array, zeros, zeros_like
from toolz import concat, keyfilter
from scipy.spatial.distance import cdist

# Constants
G = 9.81
RHOWATER = 1000

# Ball of radius r
class Particle:
    def __init__(self, ID=1, x0=[0,0], v0=[0,0], r=0.2, rho=100, bounciness=0.7):
        self.ID = ID
        self.x = array(x0)
        self.v = array(v0)
        self.r = r
        self.rho = rho
        self.m = 4/3*np.pi*r**3 * rho
    def state(self):
        # ID, x, y, [z]
        return [self.ID, *self.x]
    
# Update rules
def verlet_update(p, F, dt):
    # update position
    a = F(p)/p.m
    p.x = p.x + p.v*dt + a*dt**2/2
    # use new position to calculate acceleration and update velocity
    anew = F(p)/p.m
    p.v = p.v + (a+anew)*dt/2

# Bounding box
class Grid:
    def __init__(self, x1, x2, nx, y1, y2, ny):
        self.x1 = x1; self.x2 = x2; self.nx = nx
        self.xg = np.linspace(x1,x2,nx)
        self.y1 = y1; self.y2 = y2; self.ny = ny
        self.yg = np.linspace(y1,y2,ny)

def check_collision_wall(p, grid):
    # left/right wall
    if p.x[0] <= grid.x1 or p.x[0] >= grid.x2:
        p.v[0] = -p.v[0]
    # bottom/top wall
    if p.x[1] <= grid.y1 or p.x[1] >= grid.y2:
        p.v[1] = -p.v[1]

def ppos(particles):
    return np.array([p.x for p in particles.values()])

def pdist(particles):
    X = ppos(particles)
    D = cdist(X, X) 
    distances = D[np.triu_indices(len(D), k=1)] # <1,2>, <1,3>, ... , <1,n>, <2,3>, etc.
    pids = list(sp.keys())
    pairs = [(i,j) for idx,i in enumerate(pids) for j in pids[idx+1:]]
    return dict(zip(pairs, distances))

def psubset(particles, indices):
    return keyfilter(lambda i: i in indices, particles)

def particle_bins(particles, grid):
    # Finds which grid each particle is in,  ID : [i,j]
    pbins = {p.ID: [np.digitize(p.x[0], bins=grid.xg), np.digitize(p.x[1], bins=grid.yg)] for p in particles.values()}

    # Inverse mapping,  (i,j) : [IDs]
    pbinmap = {(i,j) : [] for i in range(grid.nx) for j in range(grid.ny)}
    for ID, p in pbins.items():
        pbinmap[tuple(p)].append(ID)
    return pbins, pbinmap

def check_collision_particle_grid(particles, grid):
    pbins, pbinmap = particle_bins(particles, grid)

    # X = np.vstack([[i,*x] for i,x in pbins.items()])
    for i in range(1, grid.nx-1):
        for j in range(1, grid.ny-1):
            # Find all particles in the 3x3 grid surrounding the cell i,j
gridparticles = list(concat([pbinmap[(i-1,j-1)], pbinmap[(i-1,j)], pbinmap[(i-1,j+1)], 
                             pbinmap[(i,j-1)],   pbinmap[(i,j)],   pbinmap[(i,j+1)], 
                             pbinmap[(i+1,j-1)], pbinmap[(i+1,j)], pbinmap[(i+1,j+1)]]))
            # Find collisions
            if len(gridparticles) > 2:
                for i,p in enumerate(gridparticles):
                    particles[p]
subdict = keyfilter(lambda i: i in gridparticles, particles)

    

def elastic_collision():
    


# Forces
def force_gravity(p):
    Fg = zeros_like(p.x)
    Fg[-1] = -G * p.m
    return Fg
def force_buoyancy(p):
    depth = p.r - p.x[-1]
    Fb = zeros_like(p.x)
    Fb[-1] = G * RHOWATER * Vsubmerged(p.r, depth)
    return Fb

# Submerged volume of a sphere with radius r at depth d
def Vsubmerged(r,d):
    if d <= 0:
        return 0
    elif d >= 2*r:
        return 4/3*np.pi*r**3
    else:
        return np.pi*(2*r*d**2/2 - d**3/3)



    # pbinmap = np.empty((grid.nx, grid.ny), dtype=object)
    # for ID, p in pbins.items():
    #     if pbinmap[p[0], p[1]] is None:
    #         pbinmap[p[0], p[1]] = [ID]
    #     else:
    #         pbinmap[p[0], p[1]].append(ID)



    # Xsort, counts = np.unique(Xbins,axis=0,return_counts=True)
    # collision_cells = [tuple(r,count) for r in Xsort[counts>1]]
    # (count, (ix, iy))
    # collisions = [tuple([c, tuple(Xsort[i,:])]) for i,c in enumerate(counts) if c>1]

    # particles = []
    # for row in collisions:
    #     particles.append(np.argwhere(np.all(Xbins==row[1], axis=1)).flatten())