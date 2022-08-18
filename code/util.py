import time,torch,math,os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def pr2T(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel()
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def rpy2r(rpy):
    """
        roll,pitch,yaw to R
    """
    roll  = rpy[0]
    pitch = rpy[1]
    yaw   = rpy[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    rot   = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert rot.shape == (3, 3)
    return rot

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def quaternion_to_euler_angle(w, x, y, z):
    """
        Quaternion to Euler angle in degree
    """
    y_sqr = y*y

    t_0 = +2.0 * (w*x + y*z)
    t_1 = +1.0 - 2.0 * (x*x + y_sqr)
    X = math.degrees(math.atan2(t_0, t_1))
	
    t_2 = +2.0 * (w*y - z*x)
    t_2 = +1.0 if t_2 > +1.0 else t_2
    t_2 = -1.0 if t_2 < -1.0 else t_2
    Y = math.degrees(math.asin(t_2))
	
    t_3 = +2.0 * (w*z + x*y)
    t_4 = +1.0 - 2.0 * (y_sqr + z*z)
    Z = math.degrees(math.atan2(t_3, t_4))
	
    return X, Y, Z

class TicTocClass():
    def __init__(self,name='tictoc'):
        """
            Init tic-toc
        """
        self.name  = name
        # Reset
        self.reset()

    def reset(self):
        """
            Reset 't_init'
        """
        self.t_init    = time.time()
        self.t_elapsed = 0.0

    def toc(self,VERBOSE=False):
        """
            Compute elapsed time
        """
        self.t_elapsed = time.time() - self.t_init
        if VERBOSE:
            print ("[%s] [%.3f]sec elapsed."%(self.name,self.t_elapsed))
        return self.t_elapsed

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def get_colors(n,cm=plt.cm.rainbow):
    """
        Get different colors
    """
    colors = cm(np.linspace(0.0,1.0,n))
    return colors

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
    x      = np.random.randn(100,5),
    x_min  = -np.ones(5),
    x_max  = np.ones(5),
    margin = 0.1):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)    

def block_mtx(M11,M12,M21,M22):
    M_upper = np.concatenate((M11,M12),axis=1)
    M_lower = np.concatenate((M21,M22),axis=1)
    M = np.concatenate((M_upper,M_lower),axis=0)
    return M    

def det_inc(det_A,inv_A,b,c):
    """
        Incremental determinant computation
    """
    out = det_A * (c - b.T @ inv_A @ b)
    return out

def inv_inc(inv_A,b,c):
    """
        Incremental inverse using matrix inverse lemma
    """
    k   = c - b.T @ inv_A @ b
    M11 = inv_A + 1/k * inv_A @ b @ b.T @ inv_A
    M12 = -1/k * inv_A @ b
    M21 = -1/k * b.T @ inv_A
    M22 = 1/k
    M   = block_mtx(M11=M11,M12=M12,M21=M21,M22=M22)
    return M    

def ikdpp(
    xs_total,              # [N x D]
    qs_total = None,       # [N]
    n_select = 10,
    n_trunc  = np.inf,
    hyp      = {'g':1.0,'l':1.0}
    ):
    """
        (Truncated) Incremental k-DPP
    """
    n_total     = xs_total.shape[0]
    idxs_remain = np.arange(0,n_total,1,dtype=np.int32)

    if n_total <= n_select: # in case of selecting more than what we already have
        xs_ikdpp   = xs_total
        idxs_ikdpp = idxs_remain
        return xs_ikdpp,idxs_ikdpp

    idxs_select = []
    for i_idx in range(n_select+1): # loop
        n_remain = len(idxs_remain)
        if i_idx == 0: # random first
            idx_select = np.random.permutation(n_total)[0]
            if qs_total is not None:
                q = 1.0+qs_total[idx_select]
            else:
                q = 1.0
            det_K_prev = q
            K_inv_prev = 1/q*np.ones(shape=(1,1))
        else:
            xs_select = xs_total[idxs_select,:]
            # Compute determinants
            dets = np.zeros(shape=n_remain)
            # for r_idx in range(n_remain): # for the remained indices
            for r_idx in np.random.permutation(n_remain)[:min(n_remain,n_trunc)]:
                # Compute the determinant of the appended kernel matrix 
                k_vec     = kernel_se(
                    X1  = xs_select,
                    X2  = xs_total[idxs_remain[r_idx],:].reshape(1,-1),
                    hyp = hyp)
                if qs_total is not None:
                    q = 1.0+qs_total[idxs_remain[r_idx]]
                else:
                    q = 1.0
                det_check = det_inc(
                    det_A = det_K_prev,
                    inv_A = K_inv_prev,
                    b     = k_vec,
                    c     = q)
                # Append the determinant
                dets[r_idx] = det_check
            # Get the index with the highest determinant
            idx_temp   = np.where(dets == np.amax(dets))[0][0]
            idx_select = idxs_remain[idx_temp]
            
            # Compute 'det_K_prev' and 'K_inv_prev'
            det_K_prev = dets[idx_temp]
            k_vec      = kernel_se(
                xs_select,
                xs_total[idx_select,:].reshape(1,-1),
                hyp = hyp)
            if qs_total is not None:
                q = 1+qs_total[idx_select]
            else:
                q = 1.0
            K_inv_prev = inv_inc(
                inv_A = K_inv_prev,
                b     = k_vec,
                c     = q)
        # Remove currently selected index from 'idxs_remain'
        idxs_remain = idxs_remain[idxs_remain != idx_select]
        # Append currently selected index to 'idxs_select'
        idxs_select.append(idx_select)
    # Select the subset from 'xs_total' with removing the first sample
    idxs_select = idxs_select[1:] # excluding the first one
    idxs_ikdpp  = np.array(idxs_select)
    xs_ikdpp    = xs_total[idxs_ikdpp]
    return xs_ikdpp,idxs_ikdpp

def torch2np(x_torch):
    """
        Torch to Numpy
    """
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np

def np2torch(x_np,device='cpu'):
    """
        Numpy to Torch
    """
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch

def plot_topdown_trajectory(
    sec_list,
    xyrad_list,
    n_arrow           = 5,
    scale             = 1.0,
    arrow_blen        = 0.5,
    arrow_flen        = 0.7,
    arrow_width       = 0.5,
    arrow_head_width  = 1.0,
    arrow_head_length = 0.5,
    arrow_alpha       = 0.3,
    bbox_ec           = (0.0,0.0,0.0),
    bbox_fc           = (1.0,0.9,0.8),
    bbox_alpha        = 0.8,
    cm                = plt.cm.rainbow_r,
    figsize           = (10,5),
    title_str         = 'Topdown trajectory'
    ):
    """
        Plot top-down view of a snapbot torso trajectory
    """
    L = len(sec_list)
    plt.figure(figsize=figsize)
    plt.plot(xyrad_list[:,0],xyrad_list[:,1],'-',lw=2,color='k')
    ticks   = np.linspace(0,L-1,n_arrow).astype(np.int16)
    colors  = get_colors(n_arrow,cm=cm)
    for t_idx,tick in enumerate(ticks):
        sec   = sec_list[tick]
        xyrad = xyrad_list[tick,:]
        u,v   = np.cos(xyrad[2]),np.sin(xyrad[2])
        color = colors[t_idx]
        plt.arrow(x           = xyrad[0]-arrow_blen*u*scale,
                  y           = xyrad[1]-arrow_blen*v*scale,
                  dx          = arrow_flen*u*scale,
                  dy          = arrow_flen*v*scale,
                  width       = arrow_width*scale,
                  head_width  = arrow_head_width*scale,
                  head_length = arrow_head_length*scale,
                  color       = color,
                  alpha       = arrow_alpha,
                  ec          = 'k',
                  lw          = 1)
        plt.text(x=xyrad[0],y=xyrad[1],s='%.2fs'%(sec),size=10,ha='center',va='center',
                bbox=dict(boxstyle="round",ec=bbox_ec,fc=color,alpha=bbox_alpha),
                rotation=xyrad[2]*180/np.pi)
    plt.axis('equal')
    plt.grid('on')
    plt.xlabel('X [m]',fontsize=13); plt.ylabel('Y [m]',fontsize=13)
    plt.title(title_str,fontsize=15)
    plt.show()

def plot_topdown_trajectory_and_joint_trajectory(
    sec_list,
    xyrad_list,
    n_arrow           = 5,
    arrow_blen        = 0.05,
    arrow_flen        = 0.07,
    arrow_width       = 0.05,
    arrow_head_width  = 0.1,
    arrow_head_length = 0.05,
    arrow_alpha       = 0.3,
    bbox_ec           = (0.0,0.0,0.0),
    bbox_fc           = (1.0,0.9,0.8),
    bbox_alpha        = 0.8,
    cm                = plt.cm.rainbow_r,
    figsize           = (10,5),
    title_str         = 'Topdown trajectory',
    traj_secs         = np.zeros((10,1)),
    traj_joints       = np.zeros((10,8)),
    t_anchor          = None,
    x_anchor          = None,
    fontsize          = 12
    ):
    """
        Plot top-down view of a snapbot torso trajectory
    """
    L = len(sec_list)
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()

    plt.subplot(1,2,1)
    plt.plot(xyrad_list[:,0],xyrad_list[:,1],'-',lw=2,color='k')
    ticks   = np.linspace(0,L-1,n_arrow).astype(np.int16)
    colors  = get_colors(n_arrow,cm=cm)
    for t_idx,tick in enumerate(ticks):
        sec   = sec_list[tick]
        xyrad = xyrad_list[tick,:]
        u,v   = np.cos(xyrad[2]),np.sin(xyrad[2])
        color = colors[t_idx]
        plt.arrow(x=xyrad[0]-arrow_blen*u,y=xyrad[1]-arrow_blen*v,dx=arrow_flen*u,dy=arrow_flen*v,
                width=arrow_width,head_width=arrow_head_width,head_length=arrow_head_length,
                color=color,alpha=arrow_alpha,ec='k',lw=1)
        plt.text(x=xyrad[0],y=xyrad[1],s='%.2fs'%(sec),size=10,ha='center',va='center',
                bbox=dict(boxstyle="round",ec=bbox_ec,fc=color,alpha=bbox_alpha),
                rotation=xyrad[2]*180/np.pi)
    plt.axis('equal')
    plt.grid('on')
    plt.xlabel('X [m]',fontsize=fontsize); plt.ylabel('Y [m]',fontsize=fontsize)
    plt.title('Top-down trajectory',fontsize=fontsize)

    plt.subplot(1,2,2)
    colors = get_colors(traj_joints.shape[1])
    for i_idx in range(traj_joints.shape[1]):
        color = colors[i_idx]
        plt.plot(traj_secs,traj_joints[:,i_idx],'-',color=color)
    if (t_anchor is not None) and (x_anchor is not None):
        for i_idx in range(x_anchor.shape[1]):
            color = colors[i_idx]
            plt.plot(t_anchor,x_anchor[:,i_idx],'o',color=color,ms=8,mfc='none',lw=1/2)
    plt.xlabel('Time [s]',fontsize=fontsize); plt.ylabel('Joint position [rad]',fontsize=fontsize)
    plt.title('Joint trajectory',fontsize=fontsize)

    plt.suptitle(title_str,fontsize=fontsize)
    plt.show()

def get_anchors_from_traj(t_test,traj,n_anchor=20):
    """
    Get equidist anchors from a trajectory
    """
    n_test = len(t_test)
    idxs = np.round(np.linspace(start=0,stop=n_test-1,num=n_anchor)).astype(np.int16)
    t_anchor,x_anchor = t_test[idxs],traj[idxs]
    return t_anchor,x_anchor

class NormalizerClass(object):
    def __init__(self,
                 name    = 'NZR',
                 x       = np.random.rand(100,4),
                 eps     = 1e-6,
                 axis    = 0,     # mean and std axis (0 or None)
                 CHECK   = True,
                 VERBOSE = False):
        super(NormalizerClass,self).__init__()
        self.name    = name
        self.x       = x
        self.eps     = eps
        self.axis    = axis
        self.CHECK   = CHECK
        self.VERBOSE = VERBOSE
        # Set data
        self.set_data(x=self.x,eps=self.eps)
        
    def set_data(self,x=np.random.rand(100,4),eps=1e-6):
        """
            Set data
        """
        self.mean = np.mean(x,axis=self.axis)
        self.std  = np.std(x,axis=self.axis)
        if np.min(self.std) < 1e-4:
            self.eps = 1.0 # numerical stability
        # Check
        if self.CHECK:
            x_org        = self.x
            x_nzd        = self.get_nzd_data(x_org=x_org)
            x_org2       = self.get_org_data(x_nzd=x_nzd)
            x_err        = x_org - x_org2
            max_abs_err  = np.max(np.abs(x_err)) # maximum absolute error
            mean_abs_err = np.mean(np.abs(x_err),axis=None) # mean absolute error
            if self.VERBOSE:
                print ("[NormalizerClass][%s] max_err:[%.3e] min_err:[%.3e]"%
                    (self.name,max_abs_err,mean_abs_err))
            
    def get_nzd_data(self,x_org):
        x_nzd = (x_org-self.mean)/(self.std + self.eps)
        return x_nzd
    
    def get_org_data(self,x_nzd):
        x_org = x_nzd*(self.std + self.eps) + self.mean
        return x_org

def whitening(x=np.random.rand(5,2)):
    """
        Whitening
    """
    if len(x.shape) == 1:
        x_mean  = np.mean(x,axis=None)
        x_std   = np.std(x,axis=None)
    else:
        x_mean  = np.mean(x,axis=0)
        x_std   = np.std(x,axis=0)
    return (x-x_mean)/x_std

def whitening_torch(x=torch.rand(5,2)):
    """
        Whitening
    """
    if len(x.shape) == 1:
        x_mean  = torch.mean(x)
        x_std   = torch.std(x)
    else:
        x_mean  = torch.mean(x,axis=0)
        x_std   = torch.std(x,axis=0)
    return (x-x_mean)/x_std

def save_torch_wb(OBJ,folder_path='../weight',pth_name='wb.pth',VERBOSE=True):
    """
        Save torch weights and biases
    """
    os.makedirs(folder_path,exist_ok=True)
    pth_path = os.path.join(folder_path,pth_name)
    torch.save(obj=OBJ.state_dict(),f=pth_path)
    if VERBOSE:
        print ("[%s] saved."%(pth_path))

def load_torch_wb(OBJ,folder_path='../weight',pth_name='wb.pth',VERBOSE=True):
    """
        Load torch weights and biases
    """
    pth_path = os.path.join(folder_path,pth_name)
    OBJ.load_state_dict(torch.load(pth_path))
    if VERBOSE:
        print ("[%s] loaded."%(pth_path))
