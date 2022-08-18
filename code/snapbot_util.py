import numpy as np
from util import quaternion_to_euler_angle,r2rpy

def get_snapbot_q(env):
    """
        Get joint position from Snapbot env
    """
    q = env.sim.data.qpos.flat
    q = np.asarray([q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22]])
    return q

def wait_until_snapbot_on_ground(env,PID,q_init=None,wait_sec=2.0):
    """
        Wait until Snapbot is on the ground
    """
    if q_init is None:
        q_init = np.zeros(env.n_actuator)
    env.reset()
    PID.reset()
    while (env.get_sec_sim()<=wait_sec):
        q = get_snapbot_q(env)
        PID.update(x_trgt=q_init,t_curr=env.get_sec_sim(),x_curr=q,VERBOSE=False)
        env.step(ctrl=PID.out(),ctrl_idxs=None)

def snapbot_rollout(env,PID,traj_joints,n_traj_repeat=5,DO_RENDER=True,
                    window_width=0.5,window_height=0.5,cam_distance=1.5,cam_elevation=-20,
                    cam_lookat=[1.0,0,-0.2],lookat_body=None,
                    TERMINATE_AFTER_FINISH=False):
    """
        Rollout of Snapbot
    """
    if DO_RENDER:
        env.init_viewer(TERMINATE_GLFW=True,INITIALIZE_GLFW=True,
                        window_width=window_width,window_height=window_height,
                        cam_distance=cam_distance,cam_elevation=cam_elevation,
                        cam_lookat=cam_lookat)
    env.reset(RESET_SIM=True)
    wait_until_snapbot_on_ground(env,PID)
    PID.reset()
    L,cnt = traj_joints.shape[0],0
    sec_list    = np.zeros(shape=(int(L*n_traj_repeat)))
    q_curr_list = np.zeros(shape=(int(L*n_traj_repeat),env.n_actuator))
    q_trgt_list = np.zeros(shape=(int(L*n_traj_repeat),env.n_actuator))
    torque_list = np.zeros(shape=(int(L*n_traj_repeat),env.n_actuator))
    xyrad_list  = np.zeros(shape=(int(L*n_traj_repeat),3))
    for r_idx in range(n_traj_repeat): # repeat
        for tick in range(L): # for each tick in trajectory
            sec,q_curr,q_trgt=cnt*env.dt,get_snapbot_q(env),traj_joints[tick,:]
            PID.update(x_trgt=q_trgt,t_curr=sec,x_curr=get_snapbot_q(env),VERBOSE=False)
            torque = PID.out()
            if DO_RENDER:
                env.step(ctrl=torque,ctrl_idxs=None)
                if lookat_body is not None:
                    env.render(render_speedup=1.0,cam_lookat=env.get_p_body(body_name=lookat_body))
                elif cam_lookat is not None:
                    env.render(render_speedup=1.0,cam_lookat=cam_lookat)
                else:
                    env.render(render_speedup=1.0)
            else:
                env.step(ctrl=torque,ctrl_idxs=None)
            p_torso = env.get_p_body(body_name='torso')
            heading_rad = r2rpy(env.get_R_body(body_name='torso'),unit='rad')[2]
            # Append
            sec_list[cnt],q_curr_list[cnt,:],q_trgt_list[cnt,:] = sec,q_curr,q_trgt
            torque_list[cnt,:],xyrad_list[cnt,:] = torque,np.concatenate((p_torso[:2],[heading_rad]))
            cnt = cnt + 1 # tick
    res = {'sec_list':sec_list,'q_curr_list':q_curr_list,'q_trgt_list':q_trgt_list,
           'torque_list':torque_list,'xyrad_list':xyrad_list}
    if DO_RENDER and TERMINATE_AFTER_FINISH:
        env.terminate_viewer()
    return res
        