from pathlib import Path
from setuptools import setup, find_namespace_packages

experiment_dir = Path('experiments/graspgenx_sim')
extra_assets = [str(path.relative_to(experiment_dir)) for path in experiment_dir.rglob('*')
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
setup(name='nero_core', version='0.1.0',
      packages=find_namespace_packages(include=['nero_core*','mujoco_conveyor_sorting*','vendor*','configs','experiments*']),
      include_package_data=True,
      package_data={'experiments.graspgenx_sim': extra_assets,
                    'nero_core.control_panel': ['web/*.html', 'web/*.css', 'web/*.js'],
                    'vendor.nero7_speed_reference': ['*.yaml','*.json','*.md'],
                    'configs':['*.yaml','*.json'],
                    'mujoco_conveyor_sorting':['configs/*.yaml','models/*.xml','assets/robots/nero/*.stl',
                        'assets/robots/nero/LICENSE','assets/robots/nero/SOURCE.md',
                        'assets/tless/mujoco/*.obj','assets/tless/SOURCE.md']},
      data_files=[('share/ament_index/resource_index/packages',['resource/nero_core']),
                  ('share/nero_core',['package.xml'])],
      install_requires=['setuptools','numpy','scipy','PyYAML','opencv-python','mujoco'],
      extras_require={'preview':['PyQt5>=5.15,<6']},
      zip_safe=False, entry_points={'console_scripts':[
          'move_to_wait = nero_core.move_to_wait:main',
          'grasp_control_panel = nero_core.control_panel.server:main',
          'two_second_pick = nero_core.two_second.main:main',
          'record_taught_path = nero_core.two_second.teach_record:main',
          'camera_grasp_view = nero_core.two_second.camera_view:main',
          'camera_fixed_freeze_test = nero_core.two_second.fixed_freeze_test:main',
          'fixed_freeze_pick = nero_core.two_second.fixed_freeze_pick:main',
          'fixed_freeze_coarse_first_pick = nero_core.two_second.fixed_freeze_coarse_first_pick:main',
          'offline_scene_collision = nero_core.two_second.offline_scene_collision:main',
          'fit_joint_response = nero_core.fit_joint_response:main',
          'physics_sim = experiments.continuous_intercept_sim.run:main',
          'vision_sim = experiments.graspgenx_sim.full_preplanned_entry:main',
          'vision_full_trace = experiments.graspgenx_sim.full_preplanned_entry:main']})
