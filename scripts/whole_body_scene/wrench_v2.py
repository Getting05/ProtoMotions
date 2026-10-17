"""Cached centroidal feasibility for conservative static-hypothesis selection.

This scores root wrench coverage only. It does not replace the bounded joint
torque validator or establish physical execution / scene ground truth.
"""
import numpy as np
import mujoco
from scipy.optimize import nnls
from .geometry import tangent_basis


class WrenchScorer:
    def __init__(self, collision, motion, frame_ids=None, friction=.7):
        self.collision = collision
        self.friction = friction
        m, d = collision.model, collision.data
        q = collision.qpos(motion)
        v = np.zeros((len(q),m.nv))
        for t in range(1,len(q)): mujoco.mj_differentiatePos(m,v[t],motion.dt,q[t-1],q[t])
        v[0] = v[1]; acceleration = np.gradient(v,motion.dt,axis=0)
        self.ids = np.arange(len(q)) if frame_ids is None else np.asarray(frame_ids)
        self.weight = float(m.body_mass.sum()*9.81)
        self.targets, self.origin, self.angular = {}, {}, {}
        for t in self.ids:
            d.qpos[:]=q[t];d.qvel[:]=v[t];mujoco.mj_forward(m,d)
            mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM)
            target=(mass@acceleration[t]+d.qfrc_bias)[:6]
            jac=np.zeros((3,m.nv));jacr=np.zeros_like(jac)
            mujoco.mj_jac(m,d,jac,jacr,d.xpos[1],1)
            self.origin[t]=d.xpos[1].copy()
            self.angular[t]=jacr[:,:6].copy()
            self.targets[t]=target / np.r_[np.full(3,self.weight),np.full(3,self.weight*.5)]
        self.frames=len(q)

    def score(self, patches):
        # patches[t]: list of (point, environment normal). All force variables
        # are unilateral; friction comes from the same .7 prior as validation.
        errors=np.zeros(self.frames)
        for t in self.ids:
            entries=patches[t]
            columns=[]
            for point,normal in entries:
                basis=tangent_basis(normal)
                mu=self.friction
                rays=np.array([[mu,0,1],[-mu,0,1],[0,mu,1],[0,-mu,1]])@basis.T
                lever=np.asarray(point)-self.origin[t]
                for force in rays:
                    moment=np.cross(lever,force)
                    root=np.r_[force, np.zeros(3)] + self.angular[t].T@moment
                    columns.append(root / np.r_[np.ones(3),np.full(3,.5)])
            target=self.targets[t]
            if columns:
                A=np.column_stack(columns)
                try:
                    coef,_=nnls(A,target,maxiter=max(500,5*A.shape[1]))
                    residual=A@coef-target
                except RuntimeError:
                    from scipy.optimize import lsq_linear
                    fit=lsq_linear(A,target,bounds=(0,np.inf),tol=1e-6,max_iter=100)
                    residual=A@fit.x-target
            else: residual=target
            errors[t]=float(residual@residual)
        return errors


class EffortScorer(WrenchScorer):
    """Root equilibrium plus a normalized actuator-effort prior.

    Feet alone can explain a seated pose's root wrench. The effort term makes
    relief of the articulated legs observable to model selection. It is a
    soft prior, not a claim that humans or the retargeter minimize this cost.
    The independent final validator still enforces actual torque bounds.
    """
    def __init__(self, collision, motion, frame_ids=None, friction=.7):
        super().__init__(collision,motion,frame_ids,friction)
        m,d=collision.model,collision.data
        self.q=collision.qpos(motion)
        v=np.zeros((len(self.q),m.nv))
        for t in range(1,len(v)):mujoco.mj_differentiatePos(m,v[t],motion.dt,self.q[t-1],self.q[t])
        v[0]=v[1];acc=np.gradient(v,motion.dt,axis=0)
        limits=np.full(m.nv-6,np.nan)
        for a in range(m.nu):
            joint=m.actuator_trnid[a,0];index=m.jnt_dofadr[joint]-6
            if index>=0 and m.actuator_forcelimited[a]:
                limits[index]=max(abs(m.actuator_forcerange[a]))*abs(m.actuator_gear[a,0])
        if len(limits) and (not np.isfinite(limits).all() or (limits<=0).any()):
            raise ValueError('V2 effort prior requires explicit MJCF motor force limits')
        self.scale=np.r_[np.full(3,1/self.weight),np.full(3,2/self.weight),np.sqrt(.1)/limits]
        self.jacobians={}
        for t in self.ids:
            d.qpos[:]=self.q[t];d.qvel[:]=v[t];mujoco.mj_forward(m,d)
            mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM)
            self.targets[t]=(mass@acc[t]+d.qfrc_bias)*self.scale
            for b in collision.collision_bodies:
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
                mujoco.mj_jac(m,d,jp,jr,d.xpos[b+1],int(b+1))
                self.jacobians[t,int(b)]=(d.xpos[b+1].copy(),jp,jr)

    def score(self,patches):
        errors=np.zeros(self.frames)
        for t in self.ids:
            columns=[]
            for point,normal,body in patches[t]:
                origin,jp,jr=self.jacobians[t,int(body)]
                mu=self.friction;basis=tangent_basis(normal)
                rays=np.array([[mu,0,1],[-mu,0,1],[0,mu,1],[0,-mu,1]])@basis.T
                for force in rays:
                    columns.append((jp.T@force+jr.T@np.cross(np.asarray(point)-origin,force))*self.weight*self.scale)
            target=self.targets[t]
            if columns:
                A=np.column_stack(columns)
                try:coef,_=nnls(A,target,maxiter=max(500,5*A.shape[1]))
                except RuntimeError:
                    from scipy.optimize import lsq_linear
                    coef=lsq_linear(A,target,bounds=(0,np.inf),tol=1e-6,max_iter=100).x
                residual=A@coef-target
            else:residual=target
            errors[t]=float(residual@residual)
        return errors


def merge_patches(*collections):
    return [sum((list(c[t]) for c in collections),[]) for t in range(len(collections[0]))]
