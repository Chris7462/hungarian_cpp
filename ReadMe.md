# hungarian\_cpp
Modern C++ implementation of Hungarian Algorithm using Eigen.

## Usage Examples
### For ROS2 projects (default):
```bash
cd ${ros2_workspace}/src
git clone git@github.com:Chris7462/hungarian_cpp.git
cd ${ros2_workspace}
colcon build --symlink-install --packages-select hungarian
```
### For non-ROS2 projects:
```bash
git clone git@github.com:Chris7462/hungarian_cpp.git
cd hungarian_cpp
mkdir build && cd build
cmake -DENABLE_ROS2=OFF ..
cmake --build . -j$(nproc)  # or make -j$(nproc)
sudo cmake --install .  # or sudo make install
```
Use `sudo ldconfig` to update library cache.

## Running unit tests (optional)
### For ROS2 projects (default):
```bash
colcon test --packages-select hungarian --event-handlers console_direct+
```
### For non-ROS2 projects:
```bash
cmake -DENABLE_ROS2=OFF -DBUILD_TESTING=ON ..
cmake --build . -j$(nproc)
ctest --verbose
```
