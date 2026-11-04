# header.stamp: exposure in ROS clock; header.frame_id: calibrated base frame
std_msgs/Header header
string target_id
uint64 observation_version
uint64 clock_epoch
builtin_interfaces/Time host_receive_ros
float64 exposure_monotonic_s
float64 host_receive_monotonic_s
float64 device_time_s
bool device_time_valid
geometry_msgs/Pose pose
geometry_msgs/Vector3 velocity_m_s
int32 category_id
float32 category_confidence
