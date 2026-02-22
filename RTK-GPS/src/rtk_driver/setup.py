from setuptools import find_packages, setup
import os
package_name = 'rtk_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),  
            ['launch/rtk_driver_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yash',
    maintainer_email='yashphallem@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rtk_driver = rtk_driver.driver:main'
        ],
    },
)
