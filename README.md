# DIC_project

Dear reader,

The required packages in our implementation are included in the requirements.txt, please make sure you correctly install them. Here we provide how to run our codes. You can simply using the "app.py" to check the robot, and the default parameter settings for each robot are optimal paramters we found from our experiments. The original "headless.py" can also be used as the commen way to check the performance of our robots by uncommenting the robot import you want and commenting out other robot imports. Since the original "headless.py" uses the optimal settings with 100 runs, which would cost a lot of time on some robots, so we also recommend you to use other headless we provided as following to test.

But if you would like to get the heatmap figures and the line charts for the similar results, you can run "headless_multiple_processor.py" for parameter tuning of model-free algorithms. Our testing computer is Macbook Pro with M1 pro chip with 10-core CPU and 16-core GPU. 


Here are some sepecific tutorial for running "headless_multiple_processor.py" as follows.
1. If you would like to obtain the heatmap for tuning gamma and epsilon for TD algorithms, please set single_para_flag = False and TD_algo = True;
2. If you would like to obtain the heatmap for tuning gamma and epsilon for on-policy Monte Carlo, please set single_para_flag = False and TD_algo = False;
3. If you would like to obtain the linechart for tuning learning rate (alpha) for TD algorithms, please set single_para_flag = True and TD_algo = True;
4. If you would like to obtain the linechart for tuning gamma for off-policy Monte Carlo, please set single_para_flag = True and TD_algo = False;


#todo: 提交代码的时候，每个robot的robot_epoch_（有下划线）的episode都写成小的，方便sb运行多处理器的headless。

#todo：提交代码的时候，episode在哪里改我们已经在comments中标注出来了，省得sb眼睛有问题找不到。

#todo：提交代码的时候，每个robot的robot_epoch（无下划线）的参数都写成optimal的，方便sb运行app.py。

#todo: 整理一遍需要的库，看看有没有不在list里面的。(已检查）

#todo：提交代码的时候，每个robot的robot_epoch_a的参数都写成optimal的,有较小的episode，方便sb运行headless_average.py。(已检查）

#todo：提交的时候，headless_comparisom, headless_policy, headless_value删掉。

We highly recommend you to choose smaller episode number (the default setting we give) for showing figures (heatmaps and line plots) in a short time (within two hours per algorithm), though with some performance loss. But if you are interested in getting the highly similar figures as shown in our report, please change the episode number for "sarsa_robot.py" and "Q_learning_robot.py" as 500 in "robot_epoch_" function, and the episode number for both Monte_Carlo_robot files as 200 in "robot_epoch_" function. We had the specific comment for it in "robot_epoch_" funtion for each model-free robot.

If you would like to get the results of the table in our report, please use "headless_average.py" to test each robot (including the model-based robots) to get the average running time, average efficiency, and average cleanliness. Note that the default setting is to successfully get the result in a short running time with some performance loss, but is you are interested in getting the very similar results as shown in our table, please import the "robot_epoch" for the robot you would like test, which means you only need to delete the "\_a" in the import lines. There are some comments for guiding. Note that if you would to like to run monte_carlo_robot_on_policy and monte_carlo_robot_on_policy, please set the "battery_drain_p=0.5, battery_drain_lam=1.0" in line 38 and "range(10)" in line 30, since it makes sure to stop one run in a relatively short time with less runs (probabily will take more than 2 hours for off-policy MC robot). There are comments for how to change it. 

Note that when running the "headless_multiple_processor.py" and "headless_average.py", please uncomment the robot import to run the robot you would like to test, and at the same time comment other robots import out. 

Have fun! :)
