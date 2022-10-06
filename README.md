# vs-dfttest2
DFTTest re-implemetation (CUDA and x86)

## Usage
```python3
from dfttest2 import DFTTest
output = DFTTest(input)
```

See also [VapourSynth-DFTTest](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest)

## Compilation
```bash
# additional options: -D ENABLE_CUDA=ON -D ENABLE_CPU=ON
cmake -S . -B build

cmake --build build

cmake --install build
```

If the vapoursynth library cannot be found by pkg-config, then the cmake variable `VS_INCLUDE_DIR` should be set.
