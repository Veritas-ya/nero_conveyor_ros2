# Nero7_grasping speed measurement

Source: https://github.com/Veritas-ya/Nero7_grasping
Commit: 57735899ab138ce293982059ec642b5dec669d78

Imported at the user’s request from their colleague’s project. The upstream
nero7_perception package declares MIT in setup.py/package.xml. Original modules
and tests are retained with source hashes in source_manifest.json. projection.py
extracts the reviewed pure project_cloud function from alignment.py without ROS
imports; its function body is unchanged. Local SDK/CLI adaptation is separate.

The original base_plane.py is also retained verbatim for the RGB-D detection
binding and multi-frame measured-green-belt calibration. Its hash is recorded
in source_manifest.json; the same upstream MIT declaration applies.
validation.py is retained verbatim as well to run the upstream synthetic
RGB-D fixture without requiring the old project directory.
