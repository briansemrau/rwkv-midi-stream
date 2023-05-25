# rwkv-midi-stream

Currently a bit messy. Set up to run with my local configuration across WSL2 and Windows.

## Getting set up on windows

For using CUDA:
- need to manually install cuda torch: `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu117`
- download ninja, add to your path
- copy `python39.lib` from python install dir to `venv/Scripts/libs/python39.lib`
- add cl to path (e.g. `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.33.31629\bin\Hostx64\x64`)

FluidSynth:
- install / add to path

To connect to FluidSynth:
- download loopMIDI and run
