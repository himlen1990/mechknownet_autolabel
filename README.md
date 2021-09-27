# Install srhandnet
pip install torch
pip install -e .

# Function part learning
rosrun mechknownet_autolabel function_demonstration_activated
roscd mechknownet_autolabel/scripts/
python function_part_detector.py
rosrun rviz rviz

subscribe the point cloud and pointstamp topic
hold the object in front of the rgbd camera and make sure your fingers are within the view of the rgbd camera, if you hand is detected, a point will be shown in the rviz.
keep your hand stable until the object point cloud is shown in the rviz.
use your index finger to point out the function part

# Trajectroy learning
rosrun mechknownet_autolabe trajectory_demonstration_tabletop
roscd mechknownet_autolabel/scripts/
rosrun rviz rviz
subscribe the point cloud and pointstamp topic
python trajectory_detector.py

make sure there are only 2 objects showed in the rviz, otherwise, you may need to adjust the workspace in the source code.

make sure your fingers are within the view of the rgbd camera and hold the object, then wait until the point cloud is showned. After that move slowly to achieve the function.