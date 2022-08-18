import os
import time
from turtle import shape # wall-clock time
import glfw # rendering purpose
import cv2  # image plot
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from screeninfo import get_monitors # get monitor size
from util import r2w,trim_scale,get_colors

class MuJoCoParserClass():
    def __init__(self,
                 name         = 'Robot',
                 rel_xml_path = '../asset/panda/franka_panda.xml',
                 VERBOSE      = False,
                 SKIP_GLFW    = False):
        """
            Initialize MuJoCo parser
        """
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.VERBOSE      = VERBOSE
        self.SKIP_GLFW    = SKIP_GLFW
        # Internal buffers
        self.SIM_MODE     = 'Idle' # Idle / Dynamics / Kinematics
        self.tick         = 0 # internal tick
        self.time_init    = time.time() # to compute wall-clock time
        self.max_sec      = None
        self.max_tick     = None
        # Parse xml 
        self._parse_xml()

        if self.VERBOSE:
            print ("[%s] Instantiated from [%s]"%(self.name,self.full_xml_path))
            print ("- Simulation timestep is [%.4f]sec and frequency is [%d]HZ"%(self.dt,self.HZ))
            print ("- [%s] has [%d] bodies and body names are\n%s"%(self.name,self.n_body,self.body_names))
            print ("- [%s] has [%d] joints"%(self.name,self.n_joint))
            for j_idx in range(self.n_joint):
                joint_name = self.joint_names[j_idx]
                joint_type = self.joint_types[j_idx]
                if joint_type == 0:
                    joint_type_str = 'free'
                elif joint_type == 1:
                    joint_type_str = 'ball'
                elif joint_type == 2:
                    joint_type_str = 'prismatic'
                elif joint_type == 3:
                    joint_type_str = 'revolute'
                else:
                    joint_type_str = 'unknown'
                print (" [%02d] name:[%s] type:[%s] joint range:[%.2f to %.2f]"%
                    (j_idx,joint_name,joint_type_str,self.joint_range[j_idx,0],self.joint_range[j_idx,1]))
            print ("- [%s] has [%d] revolute joints"%(self.name,self.n_rev_joint))
            for j_idx in range(self.n_rev_joint):
                rev_joint_idx  = self.rev_joint_idxs[j_idx]
                rev_joint_name = self.rev_joint_names[j_idx]
                print (" [%02d] joint index:[%d] and name:[%s]"%(j_idx,rev_joint_idx,rev_joint_name))
            print  ("- [%s] has [%d] actuators"%(self.name,self.n_actuator))
            for a_idx in range(self.n_actuator):
                actuator_name = self.actuator_names[a_idx]
                print (" [%02d] actuator name:[%s] torque range:[%.2f to %.2f]"%
                (a_idx,actuator_name,self.torque_range[a_idx,0],self.torque_range[a_idx,1]))

    def _parse_xml(self):
        """
            Parse xml file
        """
        # Basic MuJoCo model and sim
        self.cwd           = os.getcwd()
        self.full_xml_path = os.path.abspath(os.path.join(self.cwd,self.rel_xml_path))
        self.model         = mujoco_py.load_model_from_path(self.full_xml_path)
        self.sim           = mujoco_py.MjSim(self.model)
        # Parse model information
        self.dt              = self.sim.model.opt.timestep 
        self.HZ              = int(1/self.dt)
        self.n_body          = self.model.nbody
        self.body_names      = list(self.sim.model.body_names)
        self.n_joint         = self.model.njnt
        self.joint_idxs      = np.arange(0,self.n_joint,1)
        self.joint_names     = [self.sim.model.joint_id2name(x) for x in range(self.n_joint)]
        self.joint_types     = self.sim.model.jnt_type # 0:free, 1:ball, 2:slide, 3:hinge
        self.joint_range     = self.sim.model.jnt_range
        self.actuator_names  = list(self.sim.model.actuator_names)
        self.n_actuator      = len(self.actuator_names)
        self.torque_range    = self.sim.model.actuator_ctrlrange
        self.rev_joint_idxs  = np.where(self.joint_types==3)[0].astype(np.int32) # revolute joint indices
        self.rev_joint_names = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.n_rev_joint     = len(self.rev_joint_idxs)
        self.rev_qvel_idxs   = [self.sim.model.get_joint_qvel_addr(x) for x in self.rev_joint_names]

    def init_viewer(self,
                    TERMINATE_GLFW  = False,
                    INITIALIZE_GLFW = False,
                    window_width    = None,
                    window_height   = None,
                    cam_distance    = None,
                    cam_elevation   = None,
                    cam_lookat      = None):
        """
            Initialize viewer
        """
        if TERMINATE_GLFW:
            # To avoid sudden kernel death, we init glfw, init viewer, AND THEN TERMINATE!
            if not self.SKIP_GLFW:
                glfw.init()
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()
            if not self.SKIP_GLFW:
                glfw.terminate()
        if INITIALIZE_GLFW:
            if not self.SKIP_GLFW:
                glfw.init()
        # Initialize viewer
        self.viewer = mujoco_py.MjViewer(self.sim)
        # Set window size
        if (window_width is not None) and (window_height is not None):
            window = self.viewer.window
            width  = int(window_width*get_monitors()[0].width)
            height = int(window_height*get_monitors()[0].height)
            if not self.SKIP_GLFW:
                glfw.set_window_size(window=window,width=width,height=height)
        # Viewer setting
        if cam_distance is not None:
            self.viewer.cam.distance = cam_distance
        if cam_elevation is not None:
            self.viewer.cam.elevation = cam_elevation
        if cam_lookat is not None:
            self.viewer.cam.lookat[0] = cam_lookat[0]
            self.viewer.cam.lookat[1] = cam_lookat[1]
            self.viewer.cam.lookat[2] = cam_lookat[2]
         
    def terminate_viewer(self):
        """
            Terminate viewer
        """
        if not self.SKIP_GLFW:
            glfw.terminate()

    def render(self,render_speedup=None,cam_lookat=None,RENDER_ALWAYS=False):
        """
            Render
        """
        # Set camera lookat pose
        if cam_lookat is not None:
            for r_idx in range(len(self.sim.render_contexts)):
                self.sim.render_contexts[r_idx].cam.lookat[0] = cam_lookat[0]
                self.sim.render_contexts[r_idx].cam.lookat[1] = cam_lookat[1]
                self.sim.render_contexts[r_idx].cam.lookat[2] = cam_lookat[2]
        # Render
        if RENDER_ALWAYS:
            self.viewer._render_every_frame = True
        else:
            self.viewer._render_every_frame = False
        if RENDER_ALWAYS:
            self.viewer.render()
        elif render_speedup is None:
            self.viewer.render()
        elif (self.get_sec_sim() >= render_speedup*self.get_sec_wall()):
            self.viewer.render()
    
    def get_sec_sim(self):
        """
            Get simulation time
        """
        self.sim_state = self.sim.get_state()
        self.sec_sim   = self.tick*self.dt # self.sim_state.time
        if self.SIM_MODE=='Kinematics':
            self.sec_sim = self.tick*self.dt # forward() does not increase 'sim_state.time'
        return self.sec_sim

    def get_sec_wall(self):
        """
            Get wall-clock time
        """
        self.sec_wall = time.time() - self.time_init
        return self.sec_wall

    def set_max_sec(self,max_sec=10.0):
        """
            Set maximum second
        """
        self.max_sec  = max_sec
        self.max_tick = int(self.max_sec*self.HZ)

    def set_max_tick(self,max_tick=5000):
        """
            Set maximum tick
        """
        self.max_tick = max_tick
        self.max_sec  = self.max_tick*self.dt

    def plot_scene(self,
                   figsize       = (12,8),
                   render_w      = 1200,
                   render_h      = 800,
                   title_str     = None,
                   title_fs      = 10,
                   cam_distance  = None,
                   cam_elevation = None,
                   cam_lookat    = None,
                   RETURN_IMG    = False):
        """
            Plot scene
        """
        for _ in range(5): # render multiple times to properly apply plot configurations
            for r_idx in range(len(self.sim.render_contexts)):
                if cam_distance is not None:
                    self.sim.render_contexts[r_idx].cam.distance  = cam_distance
                if cam_elevation is not None:
                    self.sim.render_contexts[r_idx].cam.elevation = cam_elevation
                if cam_lookat is not None:
                    self.sim.render_contexts[r_idx].cam.lookat[0] = cam_lookat[0]
                    self.sim.render_contexts[r_idx].cam.lookat[1] = cam_lookat[1]
                    self.sim.render_contexts[r_idx].cam.lookat[2] = cam_lookat[2]
            img = self.sim.render(width=render_w,height=render_h)
        img = cv2.flip(cv2.rotate(img,cv2.ROTATE_180),1) # 0:up<->down, 1:left<->right
        if RETURN_IMG: # return RGB image
            return img
        else: # plot image
            plt.figure(figsize=figsize)
            plt.imshow(img)
            if title_str is not None:
                plt.title(title_str,fontsize=title_fs)
            plt.show()

    def step(self,ctrl=None,ctrl_idxs=None):
        """
            Forward dynamcis
        """
        self.SIM_MODE = 'Dynamics' # Idle / Dynamics / Kinematics
        if ctrl is not None:
            if ctrl_idxs is None:
                self.sim.data.ctrl[:] = ctrl
            else:
                self.sim.data.ctrl[ctrl_idxs] = ctrl
        # Forward dynamics
        self.sim.step()
        # Increase tick
        self.tick = self.tick + 1

    def forward(self,q_pos=None,q_pos_idxs=None):
        """
            Forward kinematics
        """
        self.SIM_MODE = 'Kinematics' # Idle / Dynamics / Kinematics
        if q_pos is not None:
            if q_pos_idxs is None:
                self.sim.data.qpos[:] = q_pos
            else:
                self.sim.data.qpos[q_pos_idxs] = q_pos
        # Forward kinematics
        self.sim.forward()
        # Increase tick
        self.tick = self.tick + 1

    def reset(self,RESET_SIM=False):
        """
            Reset
        """
        self.SIM_MODE     = 'Idle' # Idle / Dynamics / Kinematics
        self.tick         = 0 # internal tick
        self.time_init    = time.time() # to compute wall-clock time
        if RESET_SIM:
            self.sim.reset()
        
    def IS_ALIVE(self):
        """
            Is alive
        """
        return (self.tick < self.max_tick)

    def print(self,print_every_sec=None,print_every_tick=None,VERBOSE=1):
        """
            Print
        """
        if print_every_sec is not None:
            if (((self.tick)%int(print_every_sec*self.HZ))==0) or (self.tick==1):
                if (VERBOSE>=1):
                    print ("tick:[%d/%d], sec_wall:[%.3f]sec, sec_sim:[%.3f]sec"%
                    (self.tick,self.max_tick,self.get_sec_wall(),self.get_sec_sim()))
        if print_every_tick is not None:
            if (((self.tick)%print_every_tick)==0) or (self.tick==1):
                if (VERBOSE>=1):
                    print ("tick:[%d/%d], sec_wall:[%.3f]sec, sec_sim:[%.3f]sec"%
                    (self.tick,self.max_tick,self.get_sec_wall(),self.get_sec_sim()))

    def get_q_pos(self,q_pos_idxs=None):
        """
            Get current revolute joint position
        """
        self.sim_state = self.sim.get_state()
        if q_pos_idxs is None:
            q_pos = self.sim_state.qpos[:]
        else:
            q_pos = self.sim_state.qpos[q_pos_idxs]
        return q_pos

    def get_p_body(self,body_name):
        """
            Get body position
        """
        self.sim_state = self.sim.get_state()
        p = np.array(self.sim.data.body_xpos[self.body_name2idx(body_name)])
        return p

    def get_R_body(self,body_name):
        """
            Get body rotation
        """
        self.sim_state = self.sim.get_state()
        R = np.array(self.sim.data.body_xmat[self.body_name2idx(body_name)].reshape([3, 3]))
        return R

    def apply_extnal_force(self,body_name,ft):
        """
            Apply external force (6D) to body
        """
        self.sim.data.xfrc_applied[self.body_name2idx(body_name),:] = ft

    def add_marker(self,pos,type=2,radius=0.02,color=np.array([0.0,1.0,0.0,1.0]),label=''):
        """
            Add a maker to renderer
        """
        self.viewer.add_marker(
            pos   = pos,
            type  = type, # mjtGeom: 2:sphere, 3:capsule, 6:box, 9:arrow
            size  = radius*np.ones(3),
            mat   = np.eye(3).flatten(),
            rgba  = color,
            label = label)

    def get_J_body(self,body_name):
        """
            Get body Jacobian
        """
        J_p    = np.array(self.sim.data.get_body_jacp(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_R    = np.array(self.sim.data.get_body_jacr(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def one_step_ik(self,body_name,p_trgt=None,R_trgt=None,th=1.0*np.pi/180.0):
        """
            One-step inverse kinematics
        """
        J_p,J_R,J_full = self.get_J_body(body_name=body_name)
        p_curr = self.get_p_body(body_name=body_name)
        R_curr = self.get_R_body(body_name=body_name)
        if (p_trgt is not None) and (R_trgt is not None): # both p and R targets are given
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_full,np.concatenate((p_err,w_err))
        elif (p_trgt is not None) and (R_trgt is None): # only p target is given
            p_err = (p_trgt-p_curr)
            J,err = J_p,p_err
        elif (p_trgt is None) and (R_trgt is not None): # only R target is given
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_R,w_err
        else:
            raise Exception('At least one IK target is required!')
        dq = np.linalg.solve(a=(J.T@J)+1e-6*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq,err

    def body_idx2name(self,body_idx=0):
        return self.sim.model.body_id2name(body_idx)

    def body_name2idx(self,body_name='panda_eef'):
        return self.sim.model.body_name2id(body_name)

    def sleep(self,sec=0.0):
        """
            Sleep
        """
        if sec > 0.0:
            time.sleep(sec)

def get_env_object_names(env,prefix='obj_'):
    """
        Accumulate object names by assuming that the prefix is 'obj_'
    """
    object_names = [x for x in env.joint_names if x[:len(prefix)]==prefix]
    return object_names

def set_env_object(env,object_name='obj_box_01',object_pos=np.array([1.0,0.0,0.75]),color=None):
    """
        Set a single object
    """
    # Get address
    qpos_addr = env.sim.model.get_joint_qpos_addr(object_name)
    # Set position
    env.sim.data.qpos[qpos_addr[0]]   = object_pos[0] # x
    env.sim.data.qpos[qpos_addr[0]+1] = object_pos[1] # y
    env.sim.data.qpos[qpos_addr[0]+2] = object_pos[2] # z
    # Set rotation (upstraight)
    env.sim.data.qpos[qpos_addr[0]+3:qpos_addr[1]] = [0,0,0,1] # quaternion
    # Color
    if color is not None:
        idx = env.sim.model.geom_name2id(object_name)
        env.sim.model.geom_rgba[idx,:] = color
    
def set_env_objects(env,object_names=['obj_box_01','obj_box_02'],object_pos_list=np.zeros((10,2)),colors=None):
    """
        Set multiple objects
    """
    for o_idx,object_name in enumerate(object_names):
        object_pos = object_pos_list[o_idx,:]
        if colors is not None:
            color = colors[o_idx,:]
        else:
            color = None
        set_env_object(env,object_name=object_name,object_pos=object_pos,color=color)
        
def put_env_objets_in_a_row(env,prefix='obj_',x_obj=-1.0,z_obj=0.0):
    """
        Put objects in a row with modifying colors
    """
    object_names      = get_env_object_names(env,prefix='obj_')
    n_object          = len(object_names)
    object_pos_list      = np.zeros(shape=(n_object,3))
    object_pos_list[:,0] = x_obj
    object_pos_list[:,1] = np.linspace(start=0,stop=(n_object-1)*0.1,num=n_object)
    object_pos_list[:,2] = z_obj
    set_env_objects(env,object_names=object_names,object_pos_list=object_pos_list,colors=get_colors(n_object))

def get_env_object_poses(env,object_names=[]):
    """
        Get object poses
    """
    n_object    = len(object_names)
    object_xyzs  = np.zeros(shape=(n_object,3))
    object_quats = np.zeros(shape=(n_object,4))
    for o_idx,object_name in enumerate(object_names):
        qpos_addr = env.sim.model.get_joint_qpos_addr(object_name)
        # Get position
        x = env.sim.data.qpos[qpos_addr[0]]
        y = env.sim.data.qpos[qpos_addr[0]+1]
        z = env.sim.data.qpos[qpos_addr[0]+2]
        object_xyzs[o_idx,:] = np.array([x,y,z])
        # Set rotation (upstraight)
        quat = env.sim.data.qpos[qpos_addr[0]+3:qpos_addr[1]]
        # quat = quat / np.linalg.norm(quat)
        object_quats[o_idx,:] = quat
    return object_xyzs,object_quats

def quat2r(quat):
    '''
    Convenience function for mju_quat2Mat.
    '''
    res = np.zeros(9)
    mujoco_py.functions.mju_quat2Mat(res, quat)
    res = res.reshape(3,3)
    return res