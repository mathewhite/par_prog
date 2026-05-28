cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication CXX)

# Стандарт C++11 достаточен для этой задачи
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME matrix_mult)
set(SOURCES matrix_mult.cpp)

add_executable(${TARGET_NAME} ${SOURCES})

# Настройка флагов компиляции в зависимости от ОС и конфигурации
if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Od /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /DNDEBUG")
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    # -O3 для максимальной производительности в релизе
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Цель для простого запуска программы
add_custom_target(run
    COMMAND ${TARGET_NAME}
    DEPENDS ${TARGET_NAME}
    COMMENT "Running default test (5x5)..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Цель для запуска с генерацией данных и последующей проверкой через Python
add_custom_target(verify
    COMMAND ${TARGET_NAME} -r 5 results.txt
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_results.py 
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_A.txt 
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_B.txt 
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_C.txt
    DEPENDS ${TARGET_NAME}
    COMMENT "Generating random matrices and verifying against NumPy..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)