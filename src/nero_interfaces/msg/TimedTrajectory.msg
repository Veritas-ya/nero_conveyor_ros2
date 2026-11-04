std_msgs/Header header
string task_id
string target_id
uint64 version
uint64 clock_epoch
# header.stamp is the ROS-time start; point time_from_start is relative to it.
trajectory_msgs/JointTrajectoryPoint[] points
string[] joint_names
string certificate_id
