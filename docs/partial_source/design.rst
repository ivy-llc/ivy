Design
======

| Ivy can fulfill two distinct purposes:
|
| 1. enable automatic code conversions between frameworks
| 2. serve as a new ML framework with multi-framework support
|
| The Ivy codebase can then be split into three categories, and can be further split into 8 distinct submodules, each of which fall into one of these three categories as follows:
|
| IMAGE
|
| (a) **Building Blocks**
| back-end functional APIs ✅
| Ivy functional API ✅
| Framework Handler ✅
| Ivy Compiler 🚧
|
| (b) **Ivy as a Transpiler**
| front-end functional APIs 🚧
|
| (c) **Ivy as a Framework**
| Ivy stateful API ✅
| Ivy Container ✅
| Ivy Array 🚧