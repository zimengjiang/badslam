################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../tune/tune_device_reduce.cu 

OBJS += \
./tune/tune_device_reduce.o 

CU_DEPS += \
./tune/tune_device_reduce.d 


# Each subdirectory must supply rules for building sources it contributes
tune/%.o: ../tune/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -D__device__ -D__global__ -D__shared__ -D__forceinline__ -D__host__ -D__device_builtin__ -D__device_builtin_texture_type__ -DTEST_ARCH=200 -D__launch_bounds__(...) -D__align__(...) -D__CUDA_ARCH__=350 -D__CUDACC__=1 -I"/include/device_launch_parameters.h" -I"/include/device_functions.h" -I"/include" -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


