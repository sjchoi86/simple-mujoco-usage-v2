## Simple `MuJoCo` usage

What can we get out of `MuJoco`?

```
env = gym.make('Reacher-v2')
obs,info = env.reset()
for tick in range(1000):
    env.render()
    action = policy(obs)
    obs,reward,done,_ = env.step(action)
```
For those who have run the code above, you are already running `MuJoCo` under the hood. However, `MuJoCo` is not just some physics engines that simulates some robots. In this repository, we focus on the core functionalities of `MuJoCo` (or any other proper simulators) and how we can leverage such information in robot learning tasks through the lens of a Roboticist. 

In particular, we will distinguish `kinematic` and `dynamic` simulations (e.g., forward/inverse kinematics/dynamcis).

Contact: sungjoon-choi@korea.ac.kr 
