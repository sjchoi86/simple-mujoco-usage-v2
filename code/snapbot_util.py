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