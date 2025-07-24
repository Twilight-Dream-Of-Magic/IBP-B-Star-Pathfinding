cmake_minimum_required(VERSION 3.10)
project(IBP_B_Star_Pathfinding)

# 始终设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 如果没指定，就默认 Release（大小写要和下面比较一致）
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)
	set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# 设置 Debug 和 Release 的编译选项

if(MSVC)
	add_compile_options(/W4 /Zc:__cplusplus /utf-8 /std:c++17)
else()
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11")
			message(FATAL_ERROR "GNU CXX compiler version is too small !")
		endif()
		if(CMAKE_BUILD_TYPE EQUAL "Release")
			add_compile_options(-Wall -O2 -Wextra -fsigned-char -finput-charset=UTF-8 -fexec-charset=UTF-8)
		else()
			add_compile_options(-g -O0 -Wall -Wextra -fsigned-char -finput-charset=UTF-8 -fexec-charset=UTF-8)
		endif()
	endif()
endif()

# 源文件列表，至少包含一个 .cpp
set(SOURCES
    main.cpp
)

# 最终生成可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# （可选）如果有头文件目录
target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
