from setuptools import setup
import os
from glob import glob

package_name = 'teleop_simulation_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lee',
    maintainer_email='lwi2765@khu.ac.kr',
    description='Teleoperation for TurtleBot3 using keyboard',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'turtlebot3_teleop_key = teleop_simulation_pkg.turtlebot3_teleop_key:main',
        	'hand_tracking_node2 = teleop_simulation_pkg.hand_tracking_node2:main',
        	'img_subscriber = teleop_simulation_pkg.hand_tracking_subscriber:main',
        ],
    },
)
