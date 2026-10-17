"""Signed Ampere current on a fixed provided-grid rectangle, no envelopes."""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from participant.mast_green import MU0
from participant.mast_inputs import SIGNALS


def rectangle_current(psi,r,z,order):
    psi,r,z=map(np.asarray,(psi,r,z))
    if (psi.shape!=(len(z),len(r)) or r.ndim!=1 or z.ndim!=1 or min(len(r),len(z))<4
        or not all(np.isrealobj(v) and np.isfinite(v).all() for v in (psi,r,z))
        or np.any(r<=0) or np.any(np.diff(r)<=0) or np.any(np.diff(z)<=0)
        or order not in (2,4,8)):
        raise ValueError('Invalid fixed rectangular Ampere grid/order')
    spline=RectBivariateSpline(z,r,psi)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    integral=0.
    for coordinate,fixed,horizontal,direction in [(r,z[0],True,1),(z,r[-1],False,1),
                                                (r,z[-1],True,-1),(z,r[0],False,-1)]:
        mid=(coordinate[:-1]+coordinate[1:])/2
        half=np.diff(coordinate)/2
        query=(mid[:,None]+half[:,None]*nodes).ravel()
        quadrature=(half[:,None]*np.broadcast_to(weights,(len(mid),order))).ravel()
        if horizontal:
            field=-spline.ev(np.full(len(query),fixed),query,dx=1)/query
        else:
            field=spline.ev(query,np.full(len(query),fixed),dy=1)/fixed
        integral+=direction*float(np.sum(quadrature*field))
    current=-integral/MU0
    if not np.isfinite(current):raise ValueError('Nonfinite boundary circulation')
    return current


def expected_currents(row,currents,weights):
    r,z=row['efit_grid_R'],row['efit_grid_Z']
    names=np.asarray(row['coil_input_column']).astype(str)
    components={'plasma':weights.sum(1)}
    for name in np.unique(names):
        mask=names==name;inside=0
        for a in (-.25,.25):
            for b in (-.25,.25):
                rc=row['coil_R'][mask]+a*row['coil_width'][mask]
                zc=row['coil_Z'][mask]+b*row['coil_height'][mask]
                onR=(np.isclose(rc,r[0],rtol=0,atol=1e-12)|np.isclose(rc,r[-1],rtol=0,atol=1e-12))&(zc>=z[0])&(zc<=z[-1])
                onZ=(np.isclose(zc,z[0],rtol=0,atol=1e-12)|np.isclose(zc,z[-1],rtol=0,atol=1e-12))&(rc>=r[0])&(rc<=r[-1])
                if np.any(onR|onZ):raise ValueError('Original coil filament lies on fixed boundary')
                inside+=int(((rc>r[0])&(rc<r[-1])&(zc>z[0])&(zc<z[-1])).sum())
        normalization=1 if name=='magnetics_sol_current' else int(mask.sum())
        components[name]=currents[:,SIGNALS.index(name.removeprefix('magnetics_'))]*1000*inside/(4*normalization)
    values=np.stack(list(components.values()))
    expected=values.sum(0);scale=np.abs(values).sum(0)
    if not np.isfinite(values).all() or np.any(scale<=0):raise ValueError('Undefined expected enclosed-current normalization')
    return expected,scale,components
