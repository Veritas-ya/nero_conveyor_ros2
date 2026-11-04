# T-LESS / BOP asset provenance

The meshes under `mujoco/` are derived from the T-LESS `models_eval` release
distributed by the BOP benchmark. T-LESS was introduced by Tomáš Hodaň,
Pavel Haluza, Štěpán Obdržálek, Jiří Matas, Manolis Lourakis and Xenophon
Zabulis in *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less
Objects* (WACV 2017).

- Dataset: https://www.t-less.eu/
- BOP dataset page: https://bop.felk.cvut.cz/datasets/
- Distribution mirror used by the preparation script:
  https://huggingface.co/datasets/bop-benchmark/tless
- Dataset/model license: Creative Commons Attribution 4.0 International
  (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

Downloaded archives and pinned SHA-256 checksums:

| Archive | SHA-256 |
|---|---|
| `tless_base.zip` | `dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7e` |
| `tless_models.zip` | `6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928` |

The default 15-slot scene uses BOP object IDs 1, 2, 3, 4, 5, 6, 10, 11,
12, 13, 14, 15, 16, 20 and 24. `scripts/prepare_tless_assets.py` converts
their binary PLY vertices from millimetres to metres and triangulates polygon
faces if necessary. It does not recenter, rotate, normalize or otherwise
change the BOP model coordinate system. The MJCF geom translation places the
bbox-centred model on the conveyor support plane and is recorded separately in
`configs/tless_objects.yaml`.

MuJoCo treats a mesh collision geom as a convex collision shape. Consequently
the rendered surface follows the selected T-LESS mesh, while holes and other
concavities are approximated by its convex hull for contact. Mass, inertia,
friction, bins and grasp candidates are project-specific simulation metadata;
they are not annotations supplied by T-LESS.

This notice applies only to the T-LESS source data and derived meshes. It does
not relicense the repository code or the separately sourced NERO assets.
