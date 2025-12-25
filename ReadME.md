### RL 案例一：cartPole
`python dqn_cartPole.py`\
这个是gymnasium里面的一个环境，名为倒立摆，不过这里输入不是图片，而是一个向量，向量长度为4，分别表示：1. 杆的横向位置；2. 杆的横向速度；3. 杆的竖直角度；4. 杆的竖直角度的角速度。使用DQN算法进行训练。\
### RL 案例二：pong
`python dqn_pong.py`\
这也是gymnasium里面的一个环境，这里输入的就是图片，通过卷积网络来输出action。使用CNN DQN来训练的。\
### RL 案例三：breakout
`python ddqn_breakout.py`\
这也是gymnasium里面的一个环境，这里输入的也是图片，不过这里用的是Dueling DQN来训练的。收敛速度更快。Dueling_DQN 是在 CNN_DQN 基础上的架构优化。它没有增加太多计算量（参数量几乎一样），但通过解耦状态价值和动作优势，极大地提高了学习效率和稳定性。在实际应用中，几乎总是推荐优先使用 Dueling DQN 架构。\
另外这里的动作选择使用的是policy_net，通过target_net的查表得到该动作的值。之前的DQN是直接通过traget_net的最大得到动作的Q值，然后选择最大的动作。
