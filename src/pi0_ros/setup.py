from setuptools import find_packages, setup

package_name = 'pi0_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wshf',
    maintainer_email='wshf21@mails.tsinghua.edu.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'exp_main = pi0_ros.exp_main:main',
            'main_robot_real_high_freq = pi0_ros.main_robot_real_high_freq:main',
            'main_robot_real_vel_high_freq = pi0_ros.main_robot_real_vel_high_freq:main',
        ],
    },
)
