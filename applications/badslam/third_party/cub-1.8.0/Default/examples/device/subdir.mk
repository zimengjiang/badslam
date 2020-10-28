################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../examples/device/example_device_partition_flagged.cu \
../examples/device/example_device_partition_if.cu \
../examples/device/example_device_radix_sort.cu \
../examples/device/example_device_reduce.cu \
../examples/device/example_device_scan.cu \
../examples/device/example_device_select_flagged.cu \
../examples/device/example_device_select_if.cu \
../examples/device/example_device_select_unique.cu \
../examples/device/example_device_sort_find_non_trivial_runs.cu 

OBJS += \
./examples/device/example_device_partition_flagged.o \
./examples/device/example_device_partition_if.o \
./examples/device/example_device_radix_sort.o \
./examples/device/example_device_reduce.o \
./examples/device/example_device_scan.o \
./examples/device/example_device_select_flagged.o \
./examples/device/example_device_select_if.o \
./examples/device/example_device_select_unique.o \
./examples/device/example_device_sort_find_non_trivial_runs.o 

CU_DEPS += \
./examples/device/example_device_partition_flagged.d \
./examples/device/example_device_partition_if.d \
./examples/device/example_device_radix_sort.d \
./examples/device/example_device_reduce.d \
./examples/device/example_device_scan.d \
./examples/device/example_device_select_flagged.d \
./examples/device/example_device_select_if.d \
./examples/device/example_device_select_unique.d \
./examples/device/example_device_sort_find_non_trivial_runs.d 


# Each subdirectory must supply rules for building sources it contributes
examples/device/%.o: ../examples/device/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -D__device__ -D__global__ -D__shared__ -D__forceinline__ -D__host__ -D__device_builtin__ -D__device_builtin_texture_type__ -DTEST_ARCH=200 -D__launch_bounds__(...) -D__align__(...) -D__CUDA_ARCH__=350 -D__CUDACC__=1 -I"/include/device_launch_parameters.h" -I"/include/device_functions.h" -I"/include" -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


