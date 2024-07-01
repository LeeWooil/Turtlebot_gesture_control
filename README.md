# Control tutlebot with Hand gesture recognition

# Hand Gesture Controlled TurtleBot

## Overview

This project aims to control a TurtleBot using hand gestures. The system leverages computer vision to recognize hand gestures and translate them into commands for the TurtleBot, enabling intuitive and touchless interaction.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Hand Gestures](#hand-gestures)
- [Experiment result](#experiment-result)
- [Demo](#demo)
- [Conclusion](#conclusion)

  
## Introduction

Controlling robots in a more natural and intuitive way is a growing field of interest. This project integrates hand gesture recognition with TurtleBot, allowing users to control the robot using simple hand movements. The project is built using Python, OpenCV for computer vision, and ROS (Robot Operating System) for robot control.


### Motivation
- Integration of Vision and Robotics
- Control without input devices through Hand Gesture recognition
- Interest in Multi-modal approaches

### Objectives

- Build an accurate hand-gesture recognition model.
> - Experiment with various machine learning models, including 1D CNN and LSTM.
> - Select the most suitable model for hand-gesture recognition.

- Develop a real-time control package for TurtleBot.
> - Convert recognized hand gestures into control commands.
> - Implement a communication node with TurtleBot.

- Achieve real-time control and verification of the TurtleBot through accurate and responsive hand-gesture recognition.
> - Validate the system's effectiveness through simulations and, eventually, real-world applications.
> - Enhance the accessibility and usability of TurtleBot by eliminating the need for traditional input devices.



## Features

- Real-time hand gesture recognition using OpenCV
- Mapping of hand gestures to TurtleBot commands
- Easy to set up and extend


## Installation

### Prerequisites

- ROS installed on your machine (http://wiki.ros.org/ROS/Installation)
- TurtleBot setup (http://wiki.ros.org/turtlebot)
- Python 3.x
- OpenCV

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/LeeWooil/Turtlebot_gesture_control.git
    cd hand-gesture-controlled-turtlebot
    ```

2. **Build the ROS package:**
    ```bash
    cd ~/catkin_ws/src
    ln -s ~/path/to/hand-gesture-controlled-turtlebot .
    cd ~/catkin_ws
    catkin_make
    ```

3. **Source the ROS workspace:**
    ```bash
    source /setup.bash
    ```

## Usage

1. **Launch the TurtleBot:**
    ```bash
    ros2 launch teleop_simualtion_pkg launch_test.launch.py
    ```

2. **Control your TurtleBot using hand gestures!**

## Hand Gestures

The following hand gestures are recognized and mapped to TurtleBot commands:

![그림1](https://github.com/LeeWooil/Turtlebot_gesture_control/assets/69248251/3b55a760-6e67-456e-ac71-21dae70c8244)

## Experiment result

### 1D CNN vs LSTM
#### 1D CNN
|-|precision|recall|F1 score|
|--------|---|---|---|
|left|1.00|0.99|0.99|
|right|1.00|0.50|0.67|
|front|1.00|1.00|1.00|
|back|1.00|1.00|1.00|
|stop|0.91|0.92|0.91|
|non-class|0.74|0.95|0.83|


#### LSTM
|-|precision|recall|F1 score|
|--------|---|---|---|
|left|0.95|0.98|0.97|
|right|0.70|0.83|0.76|
|front|0.85|0.99|0.91|
|back|0.93|1.00|0.96|
|stop|1.00|0.84|0.91|
|non-class|0.77|0.63|0.69|


## Demo
https://github.com/LeeWooil/Turtlebot_gesture_control/assets/69248251/09fa2d3f-e80f-4736-852b-5a1627765f63

## Conclusion
### Results
- The 1D CNN model is suitable because it has better real-time performance, despite similar accuracy to other models.
- Successfully built a more sophisticated model using the Non-class approach.
- Capable of accurate real-time control of the TurtleBot.

### Improvements
- Could not apply to the actual TurtleBot and only verified in simulation.
- Applying other models besides 1D CNN & LSTM.

