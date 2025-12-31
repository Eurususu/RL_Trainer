import wandb
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback

from wandb.integration.sb3 import WandbCallback
import ale_py


# n_envs=8 表示并行运行8个环境，这能极大加速采样过程
config = {
    "env_name": "BreakoutNoFrameskip-v4",
    "num_envs": 8,
    "total_timesteps": int(10e6),
    "seed": 4089164106,    
}

run = wandb.init(
    project="PPO_Breakout",
    config = config,
    sync_tensorboard = True,  # Auto-upload sb3's tensorboard metrics
    monitor_gym = True, # Auto-upload the videos of agents playing the game
    save_code = True, # Save the code to W&B
    )

# make_atari_env 会自动添加很多Wrapper（如缩放图像、裁剪顶部记分板等）
env = make_atari_env(config["env_name"], n_envs=config["num_envs"], seed=config["seed"]) # BreakoutNoFrameskip-v4

print("ENV ACTION SPACE: ", env.action_space.n)

# Pong是动态游戏，单张图片看不出球的运动方向。
# 我们将连续4帧堆叠在一起作为输入，这样神经网络就能感知速度和方向。
env = VecFrameStack(env, n_stack=4)
# Video recorder
env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)

# https://github.com/DLR-RM/rl-trained-agents/blob/10a9c31e806820d59b20d8b85ca67090338ea912/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
model = PPO(policy = "CnnPolicy", # policy="CnnPolicy": 因为输入是图像，我们需要使用卷积神经网络(CNN)提取特征
            env = env,
            batch_size = 256,
            clip_range = 0.1, # PPO的核心剪切参数
            ent_coef = 0.01, # 熵系数，鼓励探索，防止过早收敛
            gae_lambda = 0.9, # GAE参数
            gamma = 0.99, # 折扣因子
            learning_rate = 2.5e-4,
            max_grad_norm = 0.5,
            n_epochs = 4,
            n_steps = 128, # 每个环境采样步数
            vf_coef = 0.5,
            tensorboard_log = f"runs",
            verbose=1,
            )
# Pong通常需要训练 200万 到 500万 步才能达到很高水平
print("开始训练...")  
model.learn(
    total_timesteps = config["total_timesteps"],
    callback = [
        WandbCallback(
        gradient_save_freq = 1000,
        model_save_path = f"models/{run.id}",
        ), 
        CheckpointCallback(save_freq=10000, save_path='./breakout',
                                         name_prefix=config["env_name"]),
        ]
)

model.save("ppo-BreakoutNoFrameskip-v4.zip")

# --- 测试代码 ---
# 如果你想看它玩，需要创建一个非并行的环境并设置render_mode
# obs = env.reset()ExpVar
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
