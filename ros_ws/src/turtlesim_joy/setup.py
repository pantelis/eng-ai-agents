from setuptools import find_packages, setup
import os 
from glob import glob

from ros_ws.src.turtlesim_joy import turtlesim_joy
from ros_ws.src.turtlesim_joy.turtlesim_joy import turtlesim_joy_node

package_name = 'turtlesim_joy'

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vscode",
    maintainer_email="vscode@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "turtlesim_joy_node = turtlesim_joy.turtlesim_joy_node:main",
        ],
    },
)
