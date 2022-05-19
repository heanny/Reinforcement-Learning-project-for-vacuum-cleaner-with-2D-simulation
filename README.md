# DIC_project

Dear reader,

If you would like to get the heatmap figures for the similar results, you can run "headless_pair_tuning_A2.py" of tuning the gamma and epislon for two TD algorithms and Monte Carlo (on-policy version). If you would like to get the xxxxx figures for tuning learning rate of TD algorithms and gamma for Monte Carlo (off-policy version), please run "xxxxx.py".


#todo: 提交代码的时候，每个robot的robot_epoch_（有下划线）的episode都写成小的，方便sb运行多处理器的headless。
#rodo：提交代码的时候，episode在哪里改我们已经在comments中标注出来了，省得sb眼睛有问题找不到。
#todo：提交代码的时候，每个robot的robot_epoch（无下划线）的参数都写成optimal的，方便sb运行app.py。

We would recommend you to choose smaller episode number (as the default setting) for showing results figures in a short time (two hours per algorithm), though with some performance loss. But if you are interested in getting the highly similar figures, please change the episode number for "sarsa_robot.py" and "Q_learning_robot.py" as 500 in "robot_epoch_" function. Changing the episode for both Monte_Carlo_robot files as 200 in "robot_epoch_" function. We had the specific comment for where to change that.

In addition, we also recommend you to play around with the best parameter for each robot we choose in "app.py". You can test some interesting boarding to see the performance as well. 

....


Have fun! :)
