from setuptools import setup
from glob import glob
setup(name='nero_bringup',version='0.1.0',packages=['nero_bringup'],data_files=[('share/ament_index/resource_index/packages',['resource/nero_bringup']),('share/nero_bringup',['package.xml']),('share/nero_bringup/launch',glob('launch/*.launch.py')),('share/nero_bringup/config',glob('config/*.yaml')),('share/nero_bringup/config/calibration',glob('config/calibration/*.yaml') + glob('config/calibration/*.json'))],install_requires=['setuptools'],zip_safe=False,entry_points={'console_scripts':['offline_demo = nero_bringup.offline_demo:main']})
