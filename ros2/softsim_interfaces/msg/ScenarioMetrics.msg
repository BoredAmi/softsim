# Scenario-specific scalar metrics for one output sample (SI units).
# names[i] <-> values[i], e.g. drop_test: indentation, compression, contact_force_z, ...
std_msgs/Header header
string scenario
string[] names
float64[] values
