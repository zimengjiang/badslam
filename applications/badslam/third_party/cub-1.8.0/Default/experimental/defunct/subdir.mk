################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../experimental/defunct/example_coo_spmv.cu \
../experimental/defunct/test_device_seg_reduce.cu 

OBJS += \
./experimental/defunct/example_coo_spmv.o \
./experimental/defunct/test_device_seg_reduce.o 

CU_DEPS += \
./experimental/defunct/example_coo_spmv.d \
./experimental/defunct/test_device_seg_reduce.d 


# Each subdirectory must supply rules for building sources it contributes
experimental/defunct/%.o: ../experimental/defunct/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -D__device__ -D__global__ -D__shared__ -D__forceinline__ -D__host__ -D__device_builtin__ -D__device_builtin_texture_type__ -DTEST_ARCH=200 -D__launch_bounds__(...) -D__align__(...) -D__CUDA_ARCH__=350 -D__CUDACC__=1 -I"/include/device_launch_parameters.h" -I"/include/device_functions.h" -I"/include" -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


