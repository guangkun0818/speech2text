# Linux Version 1.12.1
set(PYTORCH_VERSION "1.13.1")
set(LIBTORCH_URL
    "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
)

FetchContent_Declare(libtorch URL ${LIBTORCH_URL})
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS} -fPIC -DC10_USE_GLOG")
