################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../examples/block/example_block_radix_sort.cu \
../examples/block/example_block_reduce.cu \
../examples/block/example_block_scan.cu \
../examples/block/reduce_by_key.cu 

OBJS += \
./examples/block/example_block_radix_sort.o \
./examples/block/example_block_reduce.o \
./examples/block/example_block_scan.o \
./examples/block/reduce_by_key.o 

CU_DEPS += \
./examples/block/example_block_radix_sort.d \
./examples/block/example_block_reduce.d \
./examples/block/example_block_scan.d \
./examples/block/reduce_by_key.d 


# Each subdirectory must supply rules for building sources it contributes
examples/block/%.o: ../examples/block/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -D__device__ -D__global__ -D__shared__ -D__forceinline__ -D__host__ -D__device_builtin__ -D__device_builtin_texture_type__ -DTEST_ARCH=200 -D__launch_bounds__(...) -D__align__(...) -D__CUDA_ARCH__=350 -D__CUDACC__=1 -I"/include/device_launch_parameters.h" -I"/include/device_functions.h" -I"/include" -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


