import numpy as np
import torch
from scipy.optimize import minimize
torch.set_default_dtype(torch.float64)

def make_env(c, n=12000, nuisance_scale=1.0, seed=0):
    rng=np.random.default_rng(seed)
    y=rng.choice([-1.,1.], size=n)
    # Fixed oracle roles: x0 is invariant shape, x1 is predictive/spurious colour,
    # x2 is a mean-zero nuisance with environment-dependent variance.
    x0=y*1.5+rng.normal(size=n)
    x1=y*c+rng.normal(size=n)
    x2=rng.normal(scale=nuisance_scale,size=n)
    return torch.tensor(np.c_[x0,x1,x2]), torch.tensor(y)

def risk(w,x,y):
    return torch.nn.functional.softplus(-y*(x@w)).mean()

def qfun(w,x,y):
    # IRMv1 scalar-classifier derivative at alpha=1.
    z=x@w; t=y*z
    return (-torch.sigmoid(-t)*t).mean()

def analytic_task_hessian_direction(w, v, x, y):
    t=y*(x@w)
    p=torch.sigmoid(t)
    third=p*(1-p)*(1-2*p)
    return ((third*y*(x@v))[:,None,None] * (x[:,:,None]*x[:,None,:])).mean(0)

def base_obj_np(w, samples, ridge):
    wt=torch.tensor(w,requires_grad=True)
    val=sum(risk(wt,x,y) for x,y in samples)/len(samples)+ridge*(wt@wt)/2
    g=torch.autograd.grad(val,wt)[0]
    return float(val.detach()),g.detach().numpy()

def metrics(delta, method='IRM', n=12000, seed=17, ridge=.05, h=1e-4,
            nuisance_scales=(1.0, 1.0)):
    cs=[.35-delta/2,.35+delta/2]
    samples=[make_env(c,n,nuisance_scales[i],seed+i) for i,c in enumerate(cs)]
    opt=minimize(lambda w:base_obj_np(w,samples,ridge), np.zeros(3),jac=True,method='BFGS',options={'gtol':1e-11})
    w=torch.tensor(opt.x,requires_grad=True)
    # task source Hessian A and regularizer gradient g at theta0
    rb=sum(risk(w,x,y) for x,y in samples)/2+ridge*(w@w)/2
    A=torch.autograd.functional.hessian(lambda z: sum(risk(z,x,y) for x,y in samples)/2+ridge*(z@z)/2,w)
    if method=='IRM':
        om=sum(qfun(w,x,y)**2 for x,y in samples)/2
    elif method=='VREx':
        rr=[risk(w,x,y) for x,y in samples]; om=sum((r-sum(rr)/2)**2 for r in rr)/2
    else: raise ValueError(method)
    g=torch.autograd.grad(om,w,create_graph=True)[0]
    v=-torch.linalg.solve(A,g)
    # task Hessian directional response, D^3 R_e[v]
    dots=[]; analytic=[]; Hs=[]
    for x,y in samples:
        He=torch.autograd.functional.hessian(lambda z:risk(z,x,y),w)
        # directional finite difference of exact Hessian, used only as an independent check
        wp=(w+h*v).detach(); wm=(w-h*v).detach()
        Hp=torch.autograd.functional.hessian(lambda z:risk(z,x,y),wp)
        Hm=torch.autograd.functional.hessian(lambda z:risk(z,x,y),wm)
        fd=(Hp-Hm)/(2*h)
        dots.append(fd.detach().numpy())
        analytic.append(analytic_task_hessian_direction(w,v,x,y).detach().numpy())
        Hs.append(He.detach().numpy())
    return {'delta':delta,'w':w.detach().numpy(),'g':g.detach().numpy(),'v':v.detach().numpy(),
            'dotH':np.mean(dots,axis=0),'dotH_analytic':np.mean(analytic,axis=0),
            'fd_err':float(np.max(np.abs(np.mean(dots,axis=0)-np.mean(analytic,axis=0)))),
            'dotH_env':dots,'H':np.mean(Hs,axis=0),
            'q':[float(qfun(w,x,y).detach()) for x,y in samples], 'success':opt.success}

if __name__=='__main__':
    # Separate correlation and nuisance sweeps so delta=0 has an unambiguous
    # interpretation.  This is a calculus sanity check, not a zero-residual
    # IRMv1 mixed-branch experiment.
    for label, scales in [('correlation_sweep_equal_nuisance', (1.0, 1.0)),
                          ('nuisance_sweep_fixed_correlation', (1.0, 1.6))]:
      print('\n', label)
      for method in ['IRM','VREx']:
        print(method)
        for d in [0,.1,.3,.6,1.0,1.5]:
          z=metrics(d,method=method,nuisance_scales=scales)
          print('delta %.2f w=(%.3f,%.3f,%.3f) q=%s dotH_IS=%.3e dotH_SS=%.3e fd_err=%.3e'%
            (d,*z['w'],np.round(z['q'],4),z['dotH'][0,1],z['dotH'][1,1],z['fd_err']))
