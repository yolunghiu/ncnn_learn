set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips32el)

set(CMAKE_C_COMPILER "mips-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "mips-linux-gnu-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=mips32r2 -mabi=32 -mnan=2008 -mfp64")
set(CMAKE_CXX_FLAGS "-march=mips32r2 -mabi=32 -mnan=2008 -mfp64")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
