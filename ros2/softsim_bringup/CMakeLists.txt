cmake_minimum_required(VERSION 3.8)
project(softsim_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})
# Scenario configs live at the repository root (shared with the pure-Python CLI).
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../scenarios/
        DESTINATION share/${PROJECT_NAME}/scenarios)

ament_package()
