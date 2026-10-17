"""Uniform-grid axisymmetric current-to-flux operator; no profile closure."""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4e-7*np.pi


def grid(r, z):
    r, z = map(np.asarray, (r, z))
    if any(a.ndim != 1 or len(a)<4 or a.dtype != np.float64
           or not np.isfinite(a).all() or np.any(np.diff(a)<=0) for a in (r,z)) or np.any(r<=0):
        raise ValueError('Expected finite increasing positive-radius float64 grids')
    dr, dz = np.diff(r), np.diff(z)
    if not np.allclose(dr,dr[0],rtol=1e-10,atol=1e-12) or not np.allclose(dz,dz[0],rtol=1e-10,atol=1e-12):
        raise ValueError('Only uniform grids are defined')
    return r, z, dr[0], dz[0]


def delta_star(psi, r, z):
    r,z,hr,hz = grid(r,z); psi=np.asarray(psi)
    if psi.shape!=(len(z),len(r)) or psi.dtype!=np.float64 or not np.isfinite(psi).all():
        raise ValueError('Invalid float64 flux grid')
    drr=(psi[1:-1,2:]-2*psi[1:-1,1:-1]+psi[1:-1,:-2])/hr**2
    dr=(psi[1:-1,2:]-psi[1:-1,:-2])/(2*hr)
    dzz=(psi[2:,1:-1]-2*psi[1:-1,1:-1]+psi[:-2,1:-1])/hz**2
    return drr-dr/r[None,1:-1]+dzz


def solve_current(jtor, r, z, boundary, deadline):
    r,z,hr,hz=grid(r,z); jtor,boundary=map(np.asarray,(jtor,boundary))
    n,m=len(z)-2,len(r)-2
    if (jtor.shape!=(n,m) or boundary.shape!=(len(z),len(r))
        or any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (jtor,boundary))):
        raise ValueError('Invalid current density or manufactured boundary')
    matrix=lil_matrix((n*m,n*m),dtype=np.float64)
    rhs=-MU0*jtor*r[None,1:-1]
    for iz in range(n):
        deadline()
        for ir in range(m):
            index=iz*m+ir;radius=r[ir+1]
            matrix[index,index]=-2/hr**2-2/hz**2
            for rz,rr,coefficient in [(iz,ir-1,1/hr**2+1/(2*radius*hr)),
                (iz,ir+1,1/hr**2-1/(2*radius*hr)),(iz-1,ir,1/hz**2),(iz+1,ir,1/hz**2)]:
                if 0<=rz<n and 0<=rr<m: matrix[index,rz*m+rr]=coefficient
                else: rhs[iz,ir]-=coefficient*boundary[rz+1,rr+1]
    deadline();solution=boundary.copy()
    solution[1:-1,1:-1]=spsolve(matrix.tocsc(),rhs.ravel()).reshape(n,m)
    if not np.isfinite(solution).all(): raise ValueError('Nonfinite elliptic solution')
    residual=delta_star(solution,r,z)+MU0*jtor*r[None,1:-1]
    scale=max(float(np.max(abs(MU0*jtor*r[None,1:-1]))),1.)
    deadline()
    return solution,float(np.max(abs(residual)))/scale
