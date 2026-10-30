# H1_2 model provenance

Current training model: `h1_2.xml` and its referenced meshes.
Source: https://github.com/unitreerobotics/unitree_rl_mjlab
Commit: `1425b15f73bd4095f0df53709d7c389c3eb9e790`
Upstream path: `src/assets/robots/unitree_h1_2/xmls/h1_2.xml`.
Upstream license: `LICENSE.mjlab` (Apache 2.0).

Adaptations: mesh directory renamed from `assets` to `meshes`; inherited geom type and collision/visual attributes expanded explicitly for the Isaac Sim MJCF importer. Joint transforms, limits, masses, principal inertia tensors, sites and collision shapes follow the upstream model. The spawn adapter removes the importer's extra worldBody articulation root and applies collision exclusions before simulation.

Only the 28 meshes referenced by the current model are included. `LICENSE` preserves the original Unitree asset notice; `LICENSE.mjlab` records the upstream repository license.
