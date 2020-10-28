################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../test/link_a.cu \
../test/link_b.cu \
../test/test_allocator.cu \
../test/test_block_histogram.cu \
../test/test_block_load_store.cu \
../test/test_block_radix_sort.cu \
../test/test_block_reduce.cu \
../test/test_block_scan.cu \
../test/test_device_histogram.cu \
../test/test_device_radix_sort.cu \
../test/test_device_reduce.cu \
../test/test_device_reduce_by_key.cu \
../test/test_device_run_length_encode.cu \
../test/test_device_scan.cu \
../test/test_device_select_if.cu \
../test/test_device_select_unique.cu \
../test/test_grid_barrier.cu \
../test/test_iterator.cu \
../test/test_warp_reduce.cu \
../test/test_warp_scan.cu 

CPP_SRCS += \
../test/link_main.cpp 

OBJS += \
./test/link_a.o \
./test/link_b.o \
./test/link_main.o \
./test/test_allocator.o \
./test/test_block_histogram.o \
./test/test_block_load_store.o \
./test/test_block_radix_sort.o \
./test/test_block_reduce.o \
./test/test_block_scan.o \
./test/test_device_histogram.o \
./test/test_device_radix_sort.o \
./test/test_device_reduce.o \
./test/test_device_reduce_by_key.o \
./test/test_device_run_length_encode.o \
./test/test_device_scan.o \
./test/test_device_select_if.o \
./test/test_device_select_unique.o \
./test/test_grid_barrier.o \
./test/test_iterator.o \
./test/test_warp_reduce.o \
./test/test_warp_scan.o 

CU_DEPS += \
./test/link_a.d \
./test/link_b.d \
./test/test_allocator.d \
./test/test_block_histogram.d \
./test/test_block_load_store.d \
./test/test_block_radix_sort.d \
./test/test_block_reduce.d \
./test/test_block_scan.d \
./test/test_device_histogram.d \
./test/test_device_radix_sort.d \
./test/test_device_reduce.d \
./test/test_device_reduce_by_key.d \
./test/test_device_run_length_encode.d \
./test/test_device_scan.d \
./test/test_device_select_if.d \
./test/test_device_select_unique.d \
./test/test_grid_barrier.d \
./test/test_iterator.d \
./test/test_warp_reduce.d \
./test/test_warp_scan.d 

CPP_DEPS += \
./test/link_main.d 


# Each subdirectory must supply rules for building sources it contributes
test/%.o: ../test/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -D__device__ -D__global__ -D__shared__ -D__forceinline__ -D__host__ -D__device_builtin__ -D__device_builtin_texture_type__ -DTEST_ARCH=200 -D__launch_bounds__(...) -D__align__(...) -D__CUDA_ARCH__=350 -D__CUDACC__=1 -I"/include/device_launch_parameters.h" -I"/include/device_functions.h" -I"/include" -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

test/%.o: ../test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -D__device__ -D__global__ -D__shared__ -D__forceinline__ -D__host__ -D__device_builtin__ -D__device_builtin_texture_type__ -DTEST_ARCH=200 -D__launch_bounds__(...) -D__align__(...) -D__CUDA_ARCH__=350 -D__CUDACC__=1 -I"/include/device_launch_parameters.h" -I"/include/device_functions.h" -I"/include" -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


