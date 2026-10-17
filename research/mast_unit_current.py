"""Direct unit-ring circulation compared with its fixed grid representation."""
import numpy as np
from participant.mast_green import MU0,greens
from research.mast_vertical_force import flux_dz
from research.mast_ampere_closure import rectangle_current


def response(rc,zc,width,height,r,z,h,deadline):
    out=np.zeros((3,len(r)))
    for start in range(0,len(r),128):
        deadline();a,b=start,min(start+128,len(r));rq,zq=r[a:b,None],z[a:b,None]
        for x in (-.25,.25):
            for y in (-.25,.25):
                source_r=(rc+x*width)[None];source_z=(zc+y*height)[None]
                out[0,a:b]+=(-flux_dz(source_r,source_z,rq,zq)/rq).sum(1)/4
                for index,step in [(1,h),(2,h/2)]:
                    out[index,a:b]+=((greens(source_r,source_z,rq+step,zq)
                        -greens(source_r,source_z,rq-step,zq))/(2*step*rq)).sum(1)/4
    return out


def direct(rc,zc,width,height,r,z,order,h,deadline):
    nodes,weights=np.polynomial.legendre.leggauss(order);integral=np.zeros(2)
    for axis,fixed,horizontal,sign in [(r,z[0],True,1),(z,r[-1],False,1),
                                      (r,z[-1],True,-1),(z,r[0],False,-1)]:
        mid=(axis[:-1]+axis[1:])/2;half=np.diff(axis)/2
        query=(mid[:,None]+half[:,None]*nodes).ravel()
        w=(half[:,None]*np.broadcast_to(weights,(len(mid),order))).ravel()
        rq,zq=(query,np.full(len(query),fixed)) if horizontal else (np.full(len(query),fixed),query)
        fields=response(rc,zc,width,height,rq,zq,h,deadline)
        chosen=np.stack([fields[0],fields[0]]) if horizontal else fields[1:]
        integral+=sign*(chosen*w).sum(1)
    return -integral/MU0


def grid_current(rc,zc,width,height,r,z,normalization,deadline):
    rr,zz=np.meshgrid(r,z);rf,zf=rr.ravel(),zz.ravel();field=np.zeros(len(rf))
    for start in range(0,len(rf),128):
        deadline();a,b=start,min(start+128,len(rf))
        for x in (-.25,.25):
            for y in (-.25,.25):
                field[a:b]+=greens((rc+x*width)[None],(zc+y*height)[None],
                                  rf[a:b,None],zf[a:b,None]).sum(1)/4/normalization
    return rectangle_current(field.reshape(len(z),len(r)),r,z,8)


def unit_group(rc,zc,width,height,r,z,normalization,plan,deadline):
    inside=0
    for a in (-.25,.25):
        for b in (-.25,.25):
            source_r,source_z=rc+a*width,zc+b*height
            on=(np.isclose(source_r,r[0],atol=1e-12,rtol=0)|np.isclose(source_r,r[-1],atol=1e-12,rtol=0))&(source_z>=z[0])&(source_z<=z[-1])
            on|=(np.isclose(source_z,z[0],atol=1e-12,rtol=0)|np.isclose(source_z,z[-1],atol=1e-12,rtol=0))&(source_r>=r[0])&(source_r<=r[-1])
            if np.any(on):raise ValueError('Unitcoil filament on fixed boundary')
            inside+=int(((source_r>r[0])&(source_r<r[-1])&(source_z>z[0])&(source_z<z[-1])).sum())
    expected=inside/(4*normalization)
    values=np.stack([direct(rc,zc,width,height,r,z,k,1e-5,deadline)/normalization for k in [16,32]])
    tolerance=max(abs(expected)*1e-4,1e-4)
    grid=grid_current(rc,zc,width,height,r,z,normalization,deadline)
    return {'expected_unit_A':expected,'direct_orders_h_h2_A':values.tolist(),'grid_unit_A':grid,
        'direct_exact_passed':bool(np.max(abs(values-expected))<=tolerance),
        'paired_passed':bool(np.max(abs(values-values[-1,-1]))<=tolerance),
        'grid_error_A':float(grid-expected),'direct_max_error_A':float(np.max(abs(values-expected)))}
