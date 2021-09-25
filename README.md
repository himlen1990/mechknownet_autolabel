# Install srhandnet
pip install torch
pip install -e .

# Function part learning
rosrun mechknownet_autolabel function_demonstration_activated
roscd mechknownet_autolabel/scripts/
python function_part_detector.py

# Trajectroy learning
rosrun mechknownet_autolabe trajectory_demonstration_tabletop
roscd mechknownet_autolabel/scripts/
python trajectory_detector.py
