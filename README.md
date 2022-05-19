# DIC_project

Dear reader,

Here we provide how to run our codes. You can simply using the "app.py" to check the robot, and the default parameter settings for each robot are optimal paramters we found from our experiments. 

If you would like to get the heatmap figures and the line charts for the similar results, you can run "headless_multiple_processor.py" for parameter tuning of model-free algorithms. Our testing computer is Macbook Pro with M1 pro chip with 10-core CPU and 16-core GPU. 


Here are some sepecific tutorial for running "headless_multiple_processor.py" as follows.
1. If you would like to obtain the heatmap for tuning gamma and epsilon for TD algorithms, please set single_para_flag = False and TD_algo = True;
2. If you would like to obtain the heatmap for tuning gamma and epsilon for on-policy Monte Carlo, please set single_para_flag = False and TD_algo = False;
3. If you would like to obtain the linechart for tuning learning rate (alpha) for TD algorithms, please set single_para_flag = True and TD_algo = True;
4. If you would like to obtain the linechart for tuning gamma for off-policy Monte Carlo, please set single_para_flag = True and TD_algo = False;


#todo: 提交代码的时候，每个robot的robot_epoch_（有下划线）的episode都写成小的，方便sb运行多处理器的headless。
#todo：提交代码的时候，episode在哪里改我们已经在comments中标注出来了，省得sb眼睛有问题找不到。
#todo：提交代码的时候，每个robot的robot_epoch（无下划线）的参数都写成optimal的，方便sb运行app.py。
#todo: 整理一遍需要的库，看看有没有不在list里面的。

We would like to highly recommend you to choose smaller episode number (the default setting we give) for showing figures in a short time (within two hours per algorithm), though with some performance loss. But if you are interested in getting the highly similar figures in our report, please change the episode number for "sarsa_robot.py" and "Q_learning_robot.py" as 500 in "robot_epoch_" function, and the episode number for both Monte_Carlo_robot files as 200 in "robot_epoch_" function. We had the specific comment for it in "robot_epoch_" funtion for each model-free robot.

If you would like to get the results of our table in report, please use "headless_average.py" to test each robot (including the model-based robots).

Note that when running the "headless_multiple_processor.py" and "headless_average.py", please uncomment the robot import to run the robot you would like, and as the same time comment other robots import out. 


Have fun! :)
