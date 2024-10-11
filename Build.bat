cls
mkdir Build
cd Build
cmake -DCMAKE_TOOLCHAIN_FILE="C:\Library\vcpkg\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -G "Visual Studio 16 2019" ..
cd ..
