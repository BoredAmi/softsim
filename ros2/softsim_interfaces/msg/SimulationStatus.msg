std_msgs/Header header
string scenario
string config_name
string backend
float64 sim_time           # [s]
float64 duration           # configured duration [s]
float64 realtime_factor    # simulated seconds per wall second (recent average)
bool paused
bool finished
string[] warnings          # pre-run findings and backend warnings
