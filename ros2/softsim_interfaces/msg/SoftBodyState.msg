# Full state of one soft body. Vertices are in mesh order (stable for a given
# model build) and expressed in header.frame_id. SI units.
std_msgs/Header header
string name
geometry_msgs/Point[] vertices
geometry_msgs/Vector3[] velocities
