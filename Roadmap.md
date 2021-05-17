# Roadmap Towards MIIND version 1.07
## Planned release date: December 2020
## New features
1. The current workflow is suitable for 2D and 3D projects. For traditional 1D models, such as leaky-integrate-and-fire (LIF), quadratic-integrate-and-fire (QIF),
the workflow is quite unwieldy. It should suffice to create a mesh file for the 1D model. The generation of stat, rev, model and matrix files should automated 
from just the mesh files and the simulation XML. 
2. 3D simulations should be accessible to general users.

## Improvements
1. The generation of marginal densities seems too slow and needs to be improved.
2. ROOT visualization for 1D densities should be re-enabled.
3. The realization of adaptive steppers in CUDA should be investigated.

## Bug fixes (those that need not immediate fixing by patch)
