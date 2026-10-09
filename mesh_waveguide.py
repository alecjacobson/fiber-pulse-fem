"""Host-only, conforming Triangle meshes for a cubic-Bezier slab waveguide.

Lengths in micrometres. No solution-dependent refinement is performed.
"""
from dataclasses import dataclass, asdict
import time
import igl
import igl.triangle
import numpy as np
from fiber_pulse import spline_offset, spline_to_poly, subdivide_loop, cubic_eval


@dataclass
class MeshConfig:
    h: float = 0.04
    clad_ratio: float = 2.0
    halo: float = 0.35
    bend: float = 1.5
    length: float = 18.0
    height: float = 8.0
    width: float = 0.5
    y0: float = 3.0
    bend_start: float = 6.0
    bend_end: float = 12.0


def geometry(c):
    a, b, y, z = c.bend_start, c.bend_end, c.y0, c.bend
    m = (a + b) / 2
    # Straight leads and two cubics with horizontal tangents at each join.
    curves = np.array([[[0,y],[a/3,y],[2*a/3,y],[a,y]],
                       [[a,y],[a+(m-a)/3,y],[m-(m-a)/3,y+z],[m,y+z]],
                       [[m,y+z],[m+(b-m)/3,y+z],[b-(b-m)/3,y],[b,y]],
                       [[b,y],[b+(c.length-b)/3,y],[b+2*(c.length-b)/3,y],[c.length,y]]])
    P = np.vstack([curves[0], *(v[1:] for v in curves[1:])])
    C = np.array([[0,1,2,3],[3,4,5,6],[6,7,8,9],[9,10,11,12]])
    def side(sign):
        po, co, _, _ = spline_offset(P, C, sign*c.width/2, min(1e-4, c.h**2/10))
        v, e = spline_to_poly(po, co, min(1e-4, c.h**2/10))
        idx, _, orient = igl.edges_to_path(e)
        if orient[0] == 1:
            idx = idx[::-1]
        return v[idx]
    outline = np.vstack([side(1)[::-1], side(-1)])
    center = np.vstack([cubic_eval(curve, np.linspace(0,1,501))[:-1] for curve in curves]+[curves[-1,-1:]])
    return outline, center


def build(c: MeshConfig):
    if c.h <= 0 or c.clad_ratio < 1 or c.halo < 0:
        raise ValueError('h > 0, clad_ratio >= 1 and halo >= 0 are required')
    if not (0 < c.bend_start < c.bend_end < c.length):
        raise ValueError('bend must lie inside domain')
    t = time.perf_counter()
    outline, center = geometry(c)
    rv, re = subdivide_loop(outline, c.h)
    # Fine lattice throughout the core and its evanescent halo. Two geometric
    # transition bands prevent an abrupt jump to the coarser radiation region.
    points = []
    for spacing, lo, hi in [(c.h, 0., c.width/2+c.halo),
                             (min(2*c.h,c.h*c.clad_ratio), c.width/2+c.halo, c.width/2+c.halo+0.5)]:
        x = np.arange(spacing/2, c.length, spacing)
        yy = np.arange(spacing/2, c.height, spacing*np.sqrt(3)/2)
        X,Y = np.meshgrid(x,yy)
        X[1::2] += spacing/2
        q = np.column_stack((X.ravel(),Y.ravel()))
        dist = np.abs(q[:,1]-np.interp(q[:,0],center[:,0],center[:,1]))
        q = q[(dist>=lo)&(dist<hi)&(q[:,0]<c.length-spacing/4)]
        d2 = igl.point_mesh_squared_distance(q,rv,re)[0]
        points.append(q[d2>(0.38*spacing)**2])
    corners = np.array([[0,0],[c.length,0],[c.length,c.height],[0,c.height]])
    v,f = igl.triangle.triangulate(np.vstack([rv,*points,corners]), re,
             flags=f'Qcq30a{0.5*(c.h*c.clad_ratio)**2:.16f}')[:2]
    f = np.asarray(f,dtype=np.int32)
    cross = np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    flip = cross<0
    f[flip] = f[flip][:,[0,2,1]]
    core = np.abs(igl.winding_number(rv,re,v[f].mean(axis=1)))>0.5
    # Every constrained core-outline segment must survive as a union of mesh edges.
    # Triangle preserves the PSLG; material jumps are assigned to whole elements.
    return dict(v=v, f=f, core=core, outline=outline, center=center,
                config=asdict(c), mesh_seconds=time.perf_counter()-t)


def operators(mesh):
    """Independent double-precision host operators for CFL bounds and validation."""
    v,f = mesh['v'],mesh['f']
    p = v[f]
    area = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])/2
    grad = np.stack((p[:,[1,2,0],1]-p[:,[2,0,1],1],
                     p[:,[2,0,1],0]-p[:,[1,2,0],0]),axis=-1)/(2*area[:,None,None])
    eps = np.where(mesh['core'],2.04**2,1.46**2)
    mass = np.bincount(f.ravel(),np.repeat(area*eps/3,3),minlength=len(v))
    vol = np.bincount(f.ravel(),np.repeat(area/3,3),minlength=len(v))
    # Triangle-wise absolute row sums bound rho(M^-1 K), including obtuse cells.
    local = area[:,None]*np.sum(np.abs(grad)*np.abs(grad).sum(axis=1)[:,None,:],axis=2)
    bound = np.max(np.bincount(f.ravel(),local.ravel(),minlength=len(v))/mass)
    return area,grad,mass,vol,2/np.sqrt(bound)
