cls
mkdir Build
cd Build
cmake -DCMAKE_TOOLCHAIN_FILE="C:\Library\vcpkg\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -G "Visual Studio 16 2019" -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" ..
cd ..
