# SLAM Playground

A minimal educational repo to understand **occupancy grids** and **mapping** before jumping into ROS 2 SLAM.

## Features
- Occupancy grid with log-odds update
- Configurable grid resolution (cells/meter)
- LIDAR sensor simulation with adjustable range and FOV
- A* path planning with frontier exploration
- Two modes: exploration (full mapping) and navigation (goal-seeking)
- Real-time visualization

## Usage
```bash
git clone https://github.com/botastark/slam_playground.git
cd slam_playground
pip install -r requirements.txt
python -m slam.simulate
```

Select mode:
1. **Exploration Mode** - Map the entire environment
2. **Navigation Mode** - Navigate to a specific goal

## Configuration
Edit `slam/simulate.py` to adjust:
- Room dimensions (`ROOM_WIDTH_M`, `ROOM_HEIGHT_M`)
- Grid resolution (`CELLS_PER_METER`)
- Sensor range and FOV (`SENSOR_RANGE_M`, `SENSOR_FOV_DEG`)

## TODO
- Localization - SLAM with particle filter
- Multi-robot exploration