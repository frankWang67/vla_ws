# vla_ws

This is my workspace for VLA experiments.

## Install

```bash
git clone https://github.com/frankWang67/vla_ws.git
colcon build --symlink-install
```

## Deploy with `pi0`

1. Bring up the `pi0` policy server (assume you have set up the environment of `pi0`):

```bash
cd /path/to/openpi
uv run scripts/serve_policy.py --env DROID
```

2. Bring up the franka controller:

```bash
ssh robotics@192.168.52.5
export ROS_DOMAIN_ID=24
cd wshf_ws
source install/setup.bash
ros2 launch my_exp_bringup exp_bringup.launch.py
```

3. Bring up the high frequency command forwarder:

```bash
cd /path/to/vla_ws
source install/setup.bash
ros2 run pi0_ros main_robot_real_high_freq
```

4. Run the main node:

```bash
cd /path/to/vla_ws
source install/setup.bash
ros2 run exp_main --prompt "${YOUR PROMPT}"
```

Feel free to modify `pi0_ros/pi0_ros/exp_main.py` to enhance the performance. Enjoy it!
