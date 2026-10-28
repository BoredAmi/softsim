# Net contact interaction between two bodies ("ground" = static ground).
# force: total force exerted BY body_a ON body_b [N], world frame.
# point: normal-force-weighted centre of pressure [m].
# torque: residual moment about point [N m].
string body_a
string body_b
geometry_msgs/Vector3 force
geometry_msgs/Vector3 torque
geometry_msgs/Point point
uint32 n_contacts
