![image](https://micchang002.github.io/QDRIVE.github.io/background2.png
)
# Code
## Link
https://github.com/TigerWuu/IsaacLab
## Hierarchy
Branch : **tiger_hrl** and **hyc_hrl**.
The main modification for this project are **anymal_c_tripod** and **anymal_c_hrl** directory

```
source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/
├── __init__.py
├── anymal_c # <-original 
├── anymal_c_tripod
│          ├── agent  
│          ├── __init__.py 
│          ├── anymal_c_env.py # <-main environment for reaching target point
│          ├── anymal_c_env_cfg.py
│          └── targetVisualization.py # <-visualize the target point
└── anymal_c_hrl
           ├── agent  
           ├── __init__.py 
           ├── anymal_c_env.py # <-main environment for selecting low-level policy
           ├── anymal_c_env_cfg.py
           ├── on_policy_runner.py # <-load single env observation
           └── targetVisualization.py

```
# Installion
## Prerequisite
* Ubuntu 22.04LTS
* ROS Humble (Optional)
## Isaac sim
### Workstation Installation
1. Download and install Omniverse Launcher.
2. Install Cache from the Omniverse Launcher.
3. Install Nucleus from the Omniverse Launcher.

Follow this to the end. 
https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html#workstation-installation
## Isaac lab
### Installation using Isaac Sim Binaries
Follow this to the end. 
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html
* **Don't have to run Section: Verifying the Isaac Lab installation**
run this to verify successful installation
```bash=
./isaaclab.sh -p source/standalone/demos/quadrupeds.py
```
* Git clone **my own repository** instead of the official one
```bash=
git clone git@github.com:TigerWuu/IsaacLab.git
```
---
Notes:
Always remember to activate the conda env and go to the `cd {DIR of IsaacLab}/IsaacLab` folder

# Training with an RL Agent
![image](https://hackmd.io/_uploads/rk2jbE7fJl.png)

<!-- ## Code structure (manager based) 
```
cd IsaacLab/source
```
```
- apps
- extensions # Contains robot description file and training env etc.
    - omni.isaac.lab
        - /omni/isaac/lab/sensors/contact_sensor # Contain contact sensor, might be helpfull if we want to add sersors
    - omni.isaac.lab_assets
    -omni.isaac.lab_task
        -omni/isaac/lab_tasks/manager_based/locomotion # One of the examples in the folder
        ├── __init__.py
        └── velocity
            ├── config
            │   └── anymal_c
            │       ├── agent  # <- this is where we store the learning agent configurations
            │       ├── __init__.py  # <- this is where we register the environment and configurations to gym registry
            │       ├── flat_env_cfg.py
            │       └── rough_env_cfg.py
            ├── __init__.py
            └── velocity_env_cfg.py  # <- this is the base task configuration
- standalone # example codes for training RL agents
    - workflows # RL libraries for training
    - ... # other exapmles
```

<!-- ## Adding sensors on a robot
https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html -->

## Training the agent
run 
```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task {Envs} --num_envs 64 --headless

### 其他可用參數解釋

- `--headless`:no render during training
- `--video`：加入此選項將在訓練過程中錄製視頻，方便在訓練後回顧模型的表現。預設為 `False`。
- `--video_length`：錄製視頻的長度（以步數計算）。例如，設置為 `200` 表示每段視頻包含 200 步的訓練過程。
- `--video_interval`：設置錄製視頻的間隔（以步數計算）。例如，設置為 `2000` 表示每隔 2000 步錄製一次視頻。
- `--num_envs`：設定同時執行的環境數量，以進行並行訓練，從而加速訓練過程。
- `--task`：指定任務的名稱。例如 `Isaac-Cartpole-v0` 表示訓練模型在 Cartpole 任務環境中運行。
- `--seed`：設置環境的隨機種子，有助於確保訓練的重現性。
- `--max_iterations`：指定強化學習策略訓練的迭代次數。

# ### 指令範例（包括新參數）
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless
```
<!-- ### Environment
```
source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/mdp
```
* reward
* termination
* curriculums (I dont know what is this) -->
Envs list:
*  Isaac-Velocity-Flat-Anymal-C-Direct-tripod-v0
*  Isaac-Velocity-Flat-Anymal-C-Direct-hrl-v2
## Play the policy you trained
```
# execute from the root directory of the repository
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task {Envs} --num_envs 32
```
The above command will load the latest checkpoint from the `IsaacLab/logs/` directory. 
# Final Project
<!-- 1. Frist download our IsaacLab folder
https://github.com/TigerWuu/IsaacLab/tree/tiger -->
<!-- 
```
git clone https://github.com/TigerWuu/IsaacLab.git
``` -->
Please following the procedures to reproduce our results.
Download the trained weight .pt files first. (for low-level policy)
https://drive.google.com/drive/folders/1UIb5gnXBBzI0rTkQyX31mk1W8MAeAhbJ?usp=sharing

Put the newest directory under the path 
```
~/IsaacLab/logs/rsl_rl/anymal_c_flat_direct_tripod
```
e.g. if we want to load the right front foot, the log file hierachy would be
```
~/IsaacLab/logs/rsl_rl/anymal_c_flat_direct_tripod/2024-12-11_12-14-31
```
## Low-level
1. Reach target point 
   Switch the branch to **tiger** or **tiger_hrl** and run the command
   ```
     ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-C-Direct-tripod-v0 --num_envs 1
    ```
## High-level
2. HRL for reaching target point using the nearest foot - 1st design.
    Switch the branch to **tiger_hrl** and run the command
   ```
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-C-Direct-hrl-v2 --num_envs 1
    ```

3. HRL for reaching target point using the nearest foot - 2nd design.
    Switch the branch to **hyc_hrl** and run the command
   ```
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-C-Direct-hrl-v2 --num_envs 1
    ```
        
<!-- 3. Run the policy
```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-C-Direct-hrl-v0 --num_envs 1
``` -->

