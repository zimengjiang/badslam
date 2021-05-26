// Copyright 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cub/cub.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <math_constants.h>

#include "badslam/cost_function.cuh"
#include "badslam/cuda_util.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/gauss_newton.cuh"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"



namespace vis {
// Macro definition
// #define CudaAssert( X ) if ( !(X) ) { return; }
  // printf( "Thread %d:%d failed assert at %s:%d! \n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); 
  
__forceinline__ __device__ void ComputeRawDepthResidualAndJacobian(
    const PixelCenterUnprojector& unprojector,
    int px,
    int py,
    float pixel_calibrated_depth,
    float depth_residual_inv_stddev,
    const float3& surfel_local_position,
    const float3& surfel_local_normal,
    float* raw_residual,
    float* jacobian) {
  float3 local_unproj;
  ComputeRawDepthResidual(unprojector, px, py, pixel_calibrated_depth,
                          depth_residual_inv_stddev,
                          surfel_local_position, surfel_local_normal,
                          &local_unproj, raw_residual);
  
  // Compute Jacobian of residual.
  
//   // Old version for exp(hat(T)) * global_T_frame:
//   jacobian[0] = surfel_global_normal.x;
//   jacobian[1] = surfel_global_normal.y;
//   jacobian[2] = surfel_global_normal.z;
//   jacobian[3] = surfel_global_position.y * surfel_global_normal.z - surfel_global_position.z * surfel_global_normal.y;
//   jacobian[4] = -surfel_global_position.x * surfel_global_normal.z + surfel_global_position.z * surfel_global_normal.x;
//   jacobian[5] = surfel_global_position.x * surfel_global_normal.y - surfel_global_position.y * surfel_global_normal.x;
  
  // New version for global_T_frame * exp(hat(T)):
//   jacobian[0] = gtf.row0.x*surfel_global_normal.x + gtf.row1.x*surfel_global_normal.y + gtf.row2.x*surfel_global_normal.z;
//   jacobian[1] = gtf.row0.y*surfel_global_normal.x + gtf.row1.y*surfel_global_normal.y + gtf.row2.y*surfel_global_normal.z;
//   jacobian[2] = gtf.row0.z*surfel_global_normal.x + gtf.row1.z*surfel_global_normal.y + gtf.row2.z*surfel_global_normal.z;
//   jacobian[3] = - surfel_global_normal.x*(gtf.row0.y*local_unproj.z - gtf.row0.z*local_unproj.y)
//                 - surfel_global_normal.y*(gtf.row1.y*local_unproj.z - gtf.row1.z*local_unproj.y)
//                 - surfel_global_normal.z*(gtf.row2.y*local_unproj.z - gtf.row2.z*local_unproj.y);
//   jacobian[4] =   surfel_global_normal.x*(gtf.row0.x*local_unproj.z - gtf.row0.z*local_unproj.x)
//                 + surfel_global_normal.y*(gtf.row1.x*local_unproj.z - gtf.row1.z*local_unproj.x)
//                 + surfel_global_normal.z*(gtf.row2.x*local_unproj.z - gtf.row2.z*local_unproj.x);
//   jacobian[5] = - surfel_global_normal.x*(gtf.row0.x*local_unproj.y - gtf.row0.y*local_unproj.x)
//                 - surfel_global_normal.y*(gtf.row1.x*local_unproj.y - gtf.row1.y*local_unproj.x)
//                 - surfel_global_normal.z*(gtf.row2.x*local_unproj.y - gtf.row2.y*local_unproj.x);
  
  // Simplified form of the new version above by rotating all the vectors into
  // the local frame (which does not change the values of the dot products),
  // i.e., multiplying by frame_tr_global from the left side:
  jacobian[0] = depth_residual_inv_stddev * surfel_local_normal.x;
  jacobian[1] = depth_residual_inv_stddev * surfel_local_normal.y;
  jacobian[2] = depth_residual_inv_stddev * surfel_local_normal.z;
  jacobian[3] = depth_residual_inv_stddev * (-surfel_local_normal.y * local_unproj.z + surfel_local_normal.z * local_unproj.y);
  jacobian[4] = depth_residual_inv_stddev * ( surfel_local_normal.x * local_unproj.z - surfel_local_normal.z * local_unproj.x);
  jacobian[5] = depth_residual_inv_stddev * (-surfel_local_normal.x * local_unproj.y + surfel_local_normal.y * local_unproj.x);
}

__forceinline__ __device__ void ComputeRawDescriptorResidualAndJacobian(
    const PixelCenterProjector& color_center_projector,
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    const float3& ls,  // surfel_local_position
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2,
    float* jacobian_1,
    float* jacobian_2) {
  ComputeRawDescriptorResidual(color_texture, pxy, t1_pxy, t2_pxy, surfel_descriptor_1, surfel_descriptor_2, raw_residual_1, raw_residual_2);
  
  float grad_x_fx_1;
  float grad_y_fy_1;
  float grad_x_fx_2;
  float grad_y_fy_2;
  DescriptorJacobianWrtProjectedPosition(
      color_texture, pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2);
  grad_x_fx_1 *= color_center_projector.fx;
  grad_x_fx_2 *= color_center_projector.fx;
  grad_y_fy_1 *= color_center_projector.fy;
  grad_y_fy_2 *= color_center_projector.fy;
  
  float inv_ls_z = 1.f / ls.z;
  float ls_z_sq = ls.z * ls.z;
  float inv_ls_z_sq = inv_ls_z * inv_ls_z;
  
  jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
  jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
  jacobian_1[2] = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq;
  
  float ls_x_y = ls.x * ls.y;
  
  jacobian_1[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
  jacobian_1[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
  jacobian_1[5] = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z;
  
  jacobian_2[0] = -grad_x_fx_2 * inv_ls_z;
  jacobian_2[1] = -grad_y_fy_2 * inv_ls_z;
  jacobian_2[2] = (ls.x * grad_x_fx_2 + ls.y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq;
  jacobian_2[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[5] = -(ls.x * grad_y_fy_2 - ls.y * grad_x_fx_2) * inv_ls_z;
}

// 3.29
__forceinline__ __device__ void ComputeRawDescriptor1PointFeatureJacobian(
  const CUDABuffer_<float>& feature_arr,
  const PixelCenterProjector& color_center_projector,
  const float2& pxy,
  const float3& ls,  // surfel_local_position
  float* jacobian_all,
  int channel) {
  /*  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index == 0){
    printf("pose_jacobian: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
  }*/

  CudaAssert(ls.x == ls.x);
  CudaAssert(ls.y == ls.y);
  CudaAssert(ls.z == ls.z);
// 11.3 jzmTODO: reuse computation. here the derivative of the projected position w.r.t. pose is the same for all the channels.
float grad_x_fx_1;
float grad_y_fy_1;
Descriptor1PointJacobianWrtProjectedPositionOnChannels(
    feature_arr, pxy, &grad_x_fx_1, &grad_y_fy_1, channel);
grad_x_fx_1 *= color_center_projector.fx;
grad_y_fy_1 *= color_center_projector.fy;
// 11.3 jzmTODO: for debugging , delete it after debugging
// unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

float inv_ls_z = 1.f / ls.z;
CudaAssert(inv_ls_z == inv_ls_z);
float ls_z_sq = ls.z * ls.z;
CudaAssert(ls_z_sq == ls_z_sq);
float inv_ls_z_sq = inv_ls_z * inv_ls_z;
// 11.20 jacobian_1, depending on channel, channel is 0-based index.
*(jacobian_all+6*channel) = -grad_x_fx_1 * inv_ls_z; // jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
*(jacobian_all+6*channel+1) = -grad_y_fy_1 * inv_ls_z; // jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
*(jacobian_all+6*channel+2) = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq; // jacobian_1[2] = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq;

float ls_x_y = ls.x * ls.y;
//CudaAssert(ls_x_y == ls_x_y);
*(jacobian_all+6*channel+3) = ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq; // jacobian_1[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
*(jacobian_all+6*channel+4) = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq; // jacobian_1[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
*(jacobian_all+6*channel+5) = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z; //jacobian_1[5] = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z;
}


__forceinline__ __device__ void TestComputeRawDescriptorFeatureJacobian(
  const CUDABuffer_<float>& feature_arr,
  const PixelCenterProjector& color_center_projector,
  const float2& pxy,
  const float2& t1_pxy,
  const float2& t2_pxy,
  const float3& ls,  // surfel_local_position
  float* jacobian_all,
  int channel) {
  /*  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index == 0){
    printf("pose_jacobian: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
  }*/

  CudaAssert(ls.x == ls.x);
  CudaAssert(ls.y == ls.y);
  CudaAssert(ls.z == ls.z);
// 11.3 jzmTODO: reuse computation. here the derivative of the projected position w.r.t. pose is the same for all the channels.
float grad_x_fx_1;
float grad_y_fy_1;
float grad_x_fx_2;
float grad_y_fy_2;
TestDescriptorJacobianWrtProjectedPositionOnChannels(
    feature_arr, pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2, channel);
grad_x_fx_1 *= color_center_projector.fx;
grad_x_fx_2 *= color_center_projector.fx;
grad_y_fy_1 *= color_center_projector.fy;
grad_y_fy_2 *= color_center_projector.fy;
// 11.3 jzmTODO: for debugging , delete it after debugging
// unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

float inv_ls_z = 1.f / ls.z;
CudaAssert(inv_ls_z == inv_ls_z);
float ls_z_sq = ls.z * ls.z;
CudaAssert(ls_z_sq == ls_z_sq);
float inv_ls_z_sq = inv_ls_z * inv_ls_z;
// 11.20 jacobian_1, depending on channel, channel is 0-based index.
*(jacobian_all+6*channel) = -grad_x_fx_1 * inv_ls_z; // jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
*(jacobian_all+6*channel+1) = -grad_y_fy_1 * inv_ls_z; // jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
*(jacobian_all+6*channel+2) = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq; // jacobian_1[2] = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq;

float ls_x_y = ls.x * ls.y;
//CudaAssert(ls_x_y == ls_x_y);
*(jacobian_all+6*channel+3) = ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq; // jacobian_1[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
*(jacobian_all+6*channel+4) = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq; // jacobian_1[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
*(jacobian_all+6*channel+5) = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z; //jacobian_1[5] = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z;

// 11.20 jacobian_2, depending on channel, channel is 0-based index.
*(jacobian_all + 6*kTotalChannels + 6*channel) = -grad_x_fx_2 * inv_ls_z; // jacobian_2[0] = -grad_x_fx_2 * inv_ls_z;
*(jacobian_all + 6*kTotalChannels + 6*channel+1) = -grad_y_fy_2 * inv_ls_z; // jacobian_2[1] = -grad_y_fy_2 * inv_ls_z;
*(jacobian_all + 6*kTotalChannels + 6*channel+2) = (ls.x * grad_x_fx_2 + ls.y * grad_y_fy_2) * inv_ls_z_sq; // jacobian_2[2] = (ls.x * grad_x_fx_2 + ls.y * grad_y_fy_2) * inv_ls_z_sq;
*(jacobian_all + 6*kTotalChannels + 6*channel+3) = ((ls.y * ls.y + ls_z_sq) * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq; // jacobian_2[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq;
*(jacobian_all + 6*kTotalChannels + 6*channel+4) = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq; // jacobian_2[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq;
*(jacobian_all + 6*kTotalChannels + 6*channel+5) = -(ls.x * grad_y_fy_2 - ls.y * grad_x_fx_2) * inv_ls_z; // jacobian_2[5] = -(ls.x * grad_y_fy_2 - ls.y * grad_x_fx_2) * inv_ls_z;
}

__forceinline__ __device__ void ComputeRawDescriptorFeatureResidualAndJacobian(
  const PixelCenterProjector& color_center_projector,
  cudaTextureObject_t color_texture,
  const float2& pxy,
  const float2& t1_pxy,
  const float2& t2_pxy,
  const float3& ls,  // surfel_local_position
  float* surfel_descriptor,
  float* raw_residual_vec,
  float* jacobian_1,
  float* jacobian_2,
  int channel) {
ComputeRawFeatureDescriptorResidual(
  color_texture, // TODO: use feature_texture
  pxy,
  t1_pxy,
  t2_pxy,
  surfel_descriptor,
  raw_residual_vec);

float grad_x_fx_1;
float grad_y_fy_1;
float grad_x_fx_2;
float grad_y_fy_2;
DescriptorJacobianWrtProjectedPosition(
    color_texture, pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2);
grad_x_fx_1 *= color_center_projector.fx;
grad_x_fx_2 *= color_center_projector.fx;
grad_y_fy_1 *= color_center_projector.fy;
grad_y_fy_2 *= color_center_projector.fy;

float inv_ls_z = 1.f / ls.z;
float ls_z_sq = ls.z * ls.z;
float inv_ls_z_sq = inv_ls_z * inv_ls_z;

jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
jacobian_1[2] = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq;

float ls_x_y = ls.x * ls.y;

jacobian_1[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
jacobian_1[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
jacobian_1[5] = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z;

jacobian_2[0] = -grad_x_fx_2 * inv_ls_z;
jacobian_2[1] = -grad_y_fy_2 * inv_ls_z;
jacobian_2[2] = (ls.x * grad_x_fx_2 + ls.y * grad_y_fy_2) * inv_ls_z_sq;
jacobian_2[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq;
jacobian_2[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq;
jacobian_2[5] = -(ls.x * grad_y_fy_2 - ls.y * grad_x_fx_2) * inv_ls_z;
}

__forceinline__ __device__ void ComputeRawDescriptorResidualAndJacobianWithFloatTexture(
    const PixelCenterProjector& color_center_projector,
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    const float3& ls,  // surfel_local_position
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2,
    float* jacobian_1,
    float* jacobian_2) {
  ComputeRawDescriptorResidualWithFloatTexture(color_texture, pxy, t1_pxy, t2_pxy, surfel_descriptor_1, surfel_descriptor_2, raw_residual_1, raw_residual_2);
  
  float grad_x_fx_1;
  float grad_y_fy_1;
  float grad_x_fx_2;
  float grad_y_fy_2;
  DescriptorJacobianWrtProjectedPositionWithFloatTexture(
      color_texture, pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2);
  grad_x_fx_1 *= color_center_projector.fx;
  grad_x_fx_2 *= color_center_projector.fx;
  grad_y_fy_1 *= color_center_projector.fy;
  grad_y_fy_2 *= color_center_projector.fy;
  
  float inv_ls_z = 1.f / ls.z;
  float ls_z_sq = ls.z * ls.z;
  float inv_ls_z_sq = inv_ls_z * inv_ls_z;
  
  jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
  jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
  jacobian_1[2] = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq;
  
  float ls_x_y = ls.x * ls.y;
  
  jacobian_1[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
  jacobian_1[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
  jacobian_1[5] = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z;
  
  jacobian_2[0] = -grad_x_fx_2 * inv_ls_z;
  jacobian_2[1] = -grad_y_fy_2 * inv_ls_z;
  jacobian_2[2] = (ls.x * grad_x_fx_2 + ls.y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq;
  jacobian_2[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[5] = -(ls.x * grad_y_fy_2 - ls.y * grad_x_fx_2) * inv_ls_z;
}

__forceinline__ __device__ void ComputeRawColorResidualAndJacobian(
    const PixelCenterProjector& color_center_projector,
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float3& ls,  // surfel_local_position
    float surfel_gradmag,
    float* raw_residual,
    float* jacobian) {
  ComputeRawColorResidual(color_texture, pxy, surfel_gradmag, raw_residual);
  
  float grad_x_fx;
  float grad_y_fy;
  ColorJacobianWrtProjectedPosition(
      color_texture, pxy, &grad_x_fx, &grad_y_fy);
  grad_x_fx *= color_center_projector.fx;
  grad_y_fy *= color_center_projector.fy;
  
  float inv_ls_z = 1.f / ls.z;
  float ls_z_sq = ls.z * ls.z;
  float inv_ls_z_sq = inv_ls_z * inv_ls_z;
  
  jacobian[0] = -grad_x_fx * inv_ls_z;
  jacobian[1] = -grad_y_fy * inv_ls_z;
  jacobian[2] = (ls.x * grad_x_fx + ls.y * grad_y_fy) * inv_ls_z_sq;
  
  float ls_x_y = ls.x * ls.y;
  
  jacobian[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy + ls_x_y * grad_x_fx) * inv_ls_z_sq;
  jacobian[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx + ls_x_y * grad_y_fy) * inv_ls_z_sq;
  jacobian[5] = -(ls.x * grad_y_fy - ls.y * grad_x_fx) * inv_ls_z;
}

template <int block_width, int block_height>
__forceinline__ __device__ void AccumulatePoseResidualAndCount(
    bool visible,
    float residual,
    CUDABuffer_<u32>& residual_count_buffer,
    CUDABuffer_<float>& residual_buffer,
    typename cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* float_storage,
    typename cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* int_storage) {
  typedef typename cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef typename cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  
  __syncthreads();  // Required before re-use of shared memory.
  int num_valid_residuals = BlockReduceInt(*int_storage).Sum(visible ? 1 : 0);
  if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
    //residual_count_buffer[blockIdx.x] = num_valid_residuals;
    atomicAdd(&residual_count_buffer(0, 0), static_cast<u32>(num_valid_residuals));
  }
  
  __syncthreads();  // Required before re-use of shared memory.
  const float residual_sum =
      BlockReduceFloat(*float_storage).Sum(visible ? residual : 0.f);
  if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
    atomicAdd(&residual_buffer(0, 0), residual_sum);
  }
}

template <int block_width, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCenterProjector color_center_projector,
    PixelCornerProjector color_corner_projector,
    PixelCenterUnprojector depth_unprojector,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

  bool visible;
  SurfelProjectionResult6 r;
  if (!AnySurfelProjectsToAssociatedPixel(&surfel_index, s, &visible, &r)) {
    return;
  }
  
  float jacobian[6];
  float raw_residual;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  // TODO: Would it be faster to do the accumulation only once, while summing
  //       both residual types at the same time?
  
  // --- Depth residual ---
  if (use_depth_residuals) {
    float3 surfel_local_normal = s.frame_T_global.Rotate(r.surfel_normal);  // TODO: Could be gotten from surfel association instead of computing it twice
    
    float depth_residual_inv_stddev =
        ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(r.px), depth_unprojector.ny(r.py), r.pixel_calibrated_depth, surfel_local_normal, s.depth_params.baseline_fx);
    
    ComputeRawDepthResidualAndJacobian(
        depth_unprojector,
        r.px,
        r.py,
        r.pixel_calibrated_depth,
        depth_residual_inv_stddev,
        r.surfel_local_position,
        surfel_local_normal,
        &raw_residual,
        jacobian);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual,
        ComputeDepthResidualWeight(raw_residual),
        jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_residual),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  // --- Descriptor residual ---
  if (use_descriptor_residuals) {
    float raw_residual_2;
    float jacobian_2[6];
    
    float2 color_pxy;
    if (TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy)) {
      // CudaAssert(visible);
      float2 t1_pxy, t2_pxy;
      ComputeTangentProjections(
          r.surfel_global_position,
          r.surfel_normal,
          SurfelGetRadiusSquared(s.surfels, surfel_index),
          s.frame_T_global,
          color_corner_projector,
          &t1_pxy,
          &t2_pxy);
      ComputeRawDescriptorResidualAndJacobian(
          color_center_projector,
          color_texture,
          color_pxy,
          t1_pxy, t2_pxy,
          r.surfel_local_position,
          s.surfels(kSurfelDescriptor1, surfel_index),
          s.surfels(kSurfelDescriptor2, surfel_index),
          &raw_residual,
          &raw_residual_2,
          jacobian,
          jacobian_2);
          /*if (surfel_index == 0){
            for (int debugi = 0; debugi < 6; ++debugi){
              printf("jacobian1 = %f, jacobian2 = %f \n", jacobian[debugi], jacobian_2[debugi]);
            }
            
            printf("residual_weight 1: %f \n",ComputeDescriptorResidualWeight(raw_residual));
            printf("residual_weight 2: %f \n",ComputeDescriptorResidualWeight(raw_residual_2));
          }*/
    } else {
      visible = false;
    }
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual,
        ComputeDescriptorResidualWeight(raw_residual),
        jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual_2,
        ComputeDescriptorResidualWeight(raw_residual_2),
        jacobian_2,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_residual),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

template <int block_width, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void TestAccumulatePoseEstimationCoeffsCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCenterProjector color_center_projector,
    PixelCornerProjector color_corner_projector,
    PixelCenterUnprojector depth_unprojector,
    /*cudaTextureObject_t color_texture,*/
    CUDABuffer_<float> feature_arr,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(surfel_index == 0){
    // printf("pose: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
    printf("hi here\n");
  }
  bool visible;
  SurfelProjectionResult6 r;
  if (!AnySurfelProjectsToAssociatedPixel(&surfel_index, s, &visible, &r)) {
    return;
  }
  // CudaAssert(visible); //should be true to be here?
  float jacobian[6] = {0};
  float depth_raw_residual = 0;
  float raw_residual_vec[kSurfelNumDescriptor] = {0}; // It's very important to initialize !!!!
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  // TODO: Would it be faster to do the accumulation only once, while summing
  //       both residual types at the same time?
  
  // --- Depth residual ---
  if (use_depth_residuals) {
    float3 surfel_local_normal = s.frame_T_global.Rotate(r.surfel_normal);  // TODO: Could be gotten from surfel association instead of computing it twice
    
    float depth_residual_inv_stddev =
        ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(r.px), depth_unprojector.ny(r.py), r.pixel_calibrated_depth, surfel_local_normal, s.depth_params.baseline_fx);
    ComputeRawDepthResidualAndJacobian(
        depth_unprojector,
        r.px,
        r.py,
        r.pixel_calibrated_depth,
        depth_residual_inv_stddev,
        r.surfel_local_position,
        surfel_local_normal,
        &depth_raw_residual,
        jacobian);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        depth_raw_residual,
        ComputeDepthResidualWeight(depth_raw_residual),
        jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(depth_raw_residual),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  // --- Descriptor residual ---
  if (use_descriptor_residuals) {
    // float raw_residual_2;
    float jacobian_all[6*kSurfelNumDescriptor] = {0}; 
    float2 color_pxy;
    float2 t1_pxy, t2_pxy;
    // 10.30 If visible, compute t1_px1, t2_pxy ( <- 11.12 This statement is false! The transformdepthtocolorpixelcorner function will execute anyway whatever visible is. 
    // I tried to save this computation by skipping doing the ComputeRawFeatureDescriptorResidual if visible = false, but I got deadlock, which might come from surfels sharing the same keyframe? )
    if (TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy)) {
      // CudaAssert(visible);
      ComputeTangentProjections(
          r.surfel_global_position,
          r.surfel_normal,
          SurfelGetRadiusSquared(s.surfels, surfel_index),
          s.frame_T_global,
          color_corner_projector,
          &t1_pxy,
          &t2_pxy);
       /*CudaAssert(t1_pxy.x < feature_arr.width()/kTotalChannels);
       CudaAssert(t2_pxy.x < feature_arr.width()/kTotalChannels);
       CudaAssert(t1_pxy.y < feature_arr.height());
       CudaAssert(t2_pxy.y < feature_arr.height());*/


      // 10.30 If visible, iterate over all the channels, accumulate H and b for each channel
      // We only need to retrieve current surfel_descriptor value once
      float surfel_descriptor[kSurfelNumDescriptor]={0}; 
      #pragma unroll
      for (int i = 0; i< kSurfelNumDescriptor; ++i){
          surfel_descriptor[i] = s.surfels(kSurfelFixedAttributeCount + i, surfel_index); // constexpr int kSurfelDescriptorArr[] = {6,7,8,9,10,11};
          // CudaAssert(surfel_descriptor[i] == surfel_descriptor[i]);
        }
      // we only need to compute the descriptor residual in vector form once. 
      // jzmTODO: maybe when we change the data structure from color_texture to feature_texture, we can learn from intensity implementation and 
      // loop over all the feature maps, for each feature map, we do exactly the same thing for intensity based approach, just to change the 
      // indices of H and b (in geometry optimization). For pose optimization, we just loop over all the feature maps and accumulate H and b.
        TestComputeRawFeatureDescriptorResidual(
          feature_arr,
          color_pxy,
          t1_pxy,
          t2_pxy,
          surfel_descriptor,
          raw_residual_vec);
        for (int channel = 0; channel < kTotalChannels; ++channel){
          TestComputeRawDescriptorFeatureJacobian(
            feature_arr,
            color_center_projector,
            color_pxy,
            t1_pxy, t2_pxy,
            r.surfel_local_position,
            jacobian_all,
            channel /* channel*/);
        }
    }
    else{
      visible = false; // nothing is done if not visible 
    }
    for (int channel = 0; channel < kTotalChannels; ++channel){
      AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual_vec[channel],
        ComputeDescriptorResidualWeight(raw_residual_vec[channel]),
        jacobian_all+6*channel, // pass the address of jacobian_c_1[0]
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
      
      AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual_vec[channel + kTotalChannels], // channel_i + N is residual_2 for each channel
        ComputeDescriptorResidualWeight(raw_residual_vec[channel + kTotalChannels]),
        jacobian_all + 6*kTotalChannels + 6*channel, // pass the address of jacobian_c_2[0]
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    }
    // 10.30 Put the debug within the for loop above?
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_residual_vec[0]),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}
void CallAccumulatePoseEstimationCoeffsCUDAKernel(
    cudaStream_t stream,
    bool debug,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCenterProjector& color_center_projector,
    const PixelCornerProjector& color_corner_projector,
    const PixelCenterUnprojector& depth_unprojector,
    /*cudaTextureObject_t color_texture,*/
    const CUDABuffer_<float>& feature_buffer,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer,
    const CUDABuffer_<float>& H_buffer,
    const CUDABuffer_<float>& b_buffer) {
  COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_1D_TEMPLATED(
          TestAccumulatePoseEstimationCoeffsCUDAKernel,
          256,
          s.surfels_size,
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, _debug, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          s,
          depth_to_color,
          color_center_projector,
          color_corner_projector,
          depth_unprojector,
          /*color_texture,*/
          feature_buffer,
          residual_count_buffer,
          residual_buffer,
          H_buffer,
          b_buffer));
  CUDA_CHECK();
}

// 3.29
template <int block_width, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffs1PointCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCenterProjector color_center_projector,
    PixelCornerProjector color_corner_projector,
    PixelCenterUnprojector depth_unprojector,
    /*cudaTextureObject_t color_texture,*/
    CUDABuffer_<float> feature_arr,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer,
    float rf_weight /*5.20*/) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  /*if(surfel_index == 0){
    printf("pose: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
  }*/
  bool visible;
  SurfelProjectionResult6 r;
  if (!AnySurfelProjectsToAssociatedPixel(&surfel_index, s, &visible, &r)) {
    return;
  }
  // CudaAssert(visible); //should be true to be here?
  float jacobian[6] = {0};
  float depth_raw_residual = 0;
  float raw_residual_vec[kSurfelNumDescriptor] = {0}; // It's very important to initialize !!!!
  // 5.26
  float wf = 1.f;
  float wg = 1.f;

  constexpr int block_height = 1;
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  // TODO: Would it be faster to do the accumulation only once, while summing
  //       both residual types at the same time?
  
  // --- Depth residual ---
  if (use_depth_residuals) {
    float3 surfel_local_normal = s.frame_T_global.Rotate(r.surfel_normal);  // TODO: Could be gotten from surfel association instead of computing it twice
    
    float depth_residual_inv_stddev =
        ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(r.px), depth_unprojector.ny(r.py), r.pixel_calibrated_depth, surfel_local_normal, s.depth_params.baseline_fx);
    ComputeRawDepthResidualAndJacobian(
        depth_unprojector,
        r.px,
        r.py,
        r.pixel_calibrated_depth,
        depth_residual_inv_stddev,
        r.surfel_local_position,
        surfel_local_normal,
        &depth_raw_residual,
        jacobian);
        // 5.26
        if(kGeomResidualChannel > 0){
          wg = feature_arr(r.py, r.px + kTotalChannels*kFeatureW);
        }
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        depth_raw_residual,
        ComputeDepthResidualWeight(depth_raw_residual)*wg, // 5.26
        jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(depth_raw_residual),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  // --- Descriptor residual ---
  if (use_descriptor_residuals) {
    // float raw_residual_2;
    float jacobian_all[6*kSurfelNumDescriptor] = {0}; 
    float2 color_pxy;
    // float2 t1_pxy, t2_pxy;
    // 10.30 If visible, compute t1_px1, t2_pxy ( <- 11.12 This statement is false! The transformdepthtocolorpixelcorner function will execute anyway whatever visible is. 
    // I tried to save this computation by skipping doing the ComputeRawFeatureDescriptorResidual if visible = false, but I got deadlock, which might come from surfels sharing the same keyframe? )
    if (TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy)) {
      // CudaAssert(visible);
      // ComputeTangentProjections(
      //     r.surfel_global_position,
      //     r.surfel_normal,
      //     SurfelGetRadiusSquared(s.surfels, surfel_index),
      //     s.frame_T_global,
      //     color_corner_projector,
      //     &t1_pxy,
      //     &t2_pxy);
       /*CudaAssert(t1_pxy.x < feature_arr.width()/kTotalChannels);
       CudaAssert(t2_pxy.x < feature_arr.width()/kTotalChannels);
       CudaAssert(t1_pxy.y < feature_arr.height());
       CudaAssert(t2_pxy.y < feature_arr.height());*/


      // 10.30 If visible, iterate over all the channels, accumulate H and b for each channel
      // We only need to retrieve current surfel_descriptor value once
      float surfel_descriptor[kSurfelNumDescriptor]={0}; 
      #pragma unroll
      for (int i = 0; i< kSurfelNumDescriptor; ++i){
          surfel_descriptor[i] = s.surfels(kSurfelFixedAttributeCount + i, surfel_index); // constexpr int kSurfelDescriptorArr[] = {6,7,8,9,10,11};
          // CudaAssert(surfel_descriptor[i] == surfel_descriptor[i]);
        }
      // we only need to compute the descriptor residual in vector form once. 
      // jzmTODO: maybe when we change the data structure from color_texture to feature_texture, we can learn from intensity implementation and 
      // loop over all the feature maps, for each feature map, we do exactly the same thing for intensity based approach, just to change the 
      // indices of H and b (in geometry optimization). For pose optimization, we just loop over all the feature maps and accumulate H and b.
        ComputeRawFeatureDescriptor1PointResidualFloatpixel(
          feature_arr,
          color_pxy,
          surfel_descriptor,
          raw_residual_vec);
        for (int channel = 0; channel < kTotalChannels; ++channel){
          ComputeRawDescriptor1PointFeatureJacobian(
            feature_arr,
            color_center_projector,
            color_pxy,
            r.surfel_local_position,
            jacobian_all,
            channel /* channel*/);
        }
        if (kFeatResidualChannel > 0){
          wf = BilinearInterpolateFeatureWeight(feature_arr, color_pxy.x, color_pxy.y);
        }
    }
    else{
      visible = false; // nothing is done if not visible 
    }
    // 5.11 compute the weight only once, compute the weight for err^2, instead of each entry. i.e. the residual weight is only pointwise, not channelwise and pointwise
    float raw_residual_squared_sum = 0;
    for (int channel = 0; channel < kTotalChannels; ++channel){
      raw_residual_squared_sum += (raw_residual_vec[channel]*raw_residual_vec[channel]);
    }
    float DescriptorResidualWeight = ComputeDescriptorResidualWeightParam(raw_residual_squared_sum, rf_weight);
    for (int channel = 0; channel < kTotalChannels; ++channel){
      AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual_vec[channel],
        // ComputeDescriptorResidualWeight(raw_residual_vec[channel]),
        DescriptorResidualWeight*wf,
        jacobian_all+6*channel, // pass the address of jacobian_c_1[0]
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    }
    // 10.30 Put the debug within the for loop above?
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidualParam(raw_residual_vec[0], rf_weight),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}
void CallAccumulatePoseEstimationCoeffs1PointCUDAKernel(
    cudaStream_t stream,
    bool debug,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    float rf_weight, // 5.20 
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCenterProjector& color_center_projector,
    const PixelCornerProjector& color_corner_projector,
    const PixelCenterUnprojector& depth_unprojector,
    /*cudaTextureObject_t color_texture,*/
    const CUDABuffer_<float>& feature_buffer,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer,
    const CUDABuffer_<float>& H_buffer,
    const CUDABuffer_<float>& b_buffer) {
  COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_1D_TEMPLATED(
          AccumulatePoseEstimationCoeffs1PointCUDAKernel,
          256,
          s.surfels_size,
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, _debug, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          s,
          depth_to_color,
          color_center_projector,
          color_corner_projector,
          depth_unprojector,
          /*color_texture,*/
          feature_buffer,
          residual_count_buffer,
          residual_buffer,
          H_buffer,
          b_buffer,
          rf_weight /*5.20 */));
  CUDA_CHECK();
}

template <int block_width, int block_height, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradientXY(
    PixelCornerProjector depth_projector,
    PixelCenterProjector color_center_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer,
    CUDABuffer_<float> debug_residual_image) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float depth_jacobian[6];
  float raw_depth_residual;
  float descriptor_jacobian_1[6];
  float descriptor_jacobian_2[6];
  float raw_descriptor_residual_1;
  float raw_descriptor_residual_2;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                ComputeRawDepthResidualAndJacobian(
                    depth_unprojector,
                    px,
                    py,
                    pixel_calibrated_depth,
                    depth_residual_inv_stddev,
                    surfel_local_position,
                    surfel_local_normal,
                    &raw_depth_residual,
                    depth_jacobian);
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {
                  // TODO: De-duplicate this with the identical code below in this file
                  // Compute descriptor in surfel image
                  const float intensity = 1 / 255.f * surfel_color(y, x);
                  const float t1_intensity = 1 / 255.f * surfel_color(y, x + 1);
                  const float t2_intensity = 1 / 255.f * surfel_color(y + 1, x);
                  
                  float surfel_descriptor_1 = (180.f * (t1_intensity - intensity));
                  float surfel_descriptor_2 = (180.f * (t2_intensity - intensity));
                  
                  // Transform the two offset points to the target / estimate frame.
                  // In order not to require depth estimates at both offset pixels,
                  // we estimate their depth using the center pixel's normal.
                  float3 surfel_normal = U16ToImageSpaceNormal(surfel_normals(y, x));
                  const float plane_d =
                      (depth_unprojector.nx(x) * surfel_calibrated_depth) * surfel_normal.x +
                      (depth_unprojector.ny(y) * surfel_calibrated_depth) * surfel_normal.y + surfel_calibrated_depth * surfel_normal.z;
                  
                  float x_plus_1_depth = plane_d / (depth_unprojector.nx(x + 1) * surfel_normal.x + depth_unprojector.ny(y) * surfel_normal.y + surfel_normal.z);
                  float3 x_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x + 1, y, x_plus_1_depth);
                  float2 pxy_t1 = depth_projector.Project(x_plus_1_local_position);
                  int t1_px = static_cast<int>(pxy_t1.x);
                  int t1_py = static_cast<int>(pxy_t1.y);
                  if (pxy_t1.x < 0 || pxy_t1.y < 0 ||
                      // t1_px < 0 || t1_py < 0 ||
                      t1_px >= frame_depth.width() || t1_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float y_plus_1_depth = plane_d / (depth_unprojector.nx(x) * surfel_normal.x + depth_unprojector.ny(y + 1) * surfel_normal.y + surfel_normal.z);
                  float3 y_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x, y + 1, y_plus_1_depth);
                  float2 pxy_t2 = depth_projector.Project(y_plus_1_local_position);
                  int t2_px = static_cast<int>(pxy_t2.x);
                  int t2_py = static_cast<int>(pxy_t2.y);
                  if (pxy_t2.x < 0 || pxy_t2.y < 0 ||
                      // t2_px < 0 || t2_py < 0 ||
                      t2_px >= frame_depth.width() || t2_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float2 color_pxy, color_pxy_t1, color_pxy_t2;
                  if (visible &&
                      x_plus_1_local_position.z > 0 &&
                      y_plus_1_local_position.z > 0 &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy) &&
                      TransformDepthToColorPixelCorner(pxy_t1, depth_to_color, &color_pxy_t1) &&
                      TransformDepthToColorPixelCorner(pxy_t2, depth_to_color, &color_pxy_t2)) {
                    ComputeRawDescriptorResidualAndJacobianWithFloatTexture(
                        color_center_projector,
                        frame_color,
                        color_pxy,
                        color_pxy_t1,
                        color_pxy_t2,
                        surfel_local_position,
                        surfel_descriptor_1,
                        surfel_descriptor_2,
                        &raw_descriptor_residual_1,
                        &raw_descriptor_residual_2,
                        descriptor_jacobian_1,
                        descriptor_jacobian_2);
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Write residual debug image?
  if (debug && x < surfel_depth.width() && y < surfel_depth.height()) {
    debug_residual_image(y, x) =
        visible ?
        ((use_depth_residuals ? ComputeWeightedDepthResidual(raw_depth_residual) : 0) +
         (use_descriptor_residuals ? ComputeWeightedDescriptorResidual(raw_descriptor_residual_1) : 0)) :  // NOTE: Using the 1st residual only
        CUDART_NAN_F;
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_depth_residual,
        ComputeDepthResidualWeight(raw_depth_residual, threshold_factor),
        depth_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  if (use_descriptor_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_1,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_1, threshold_factor),
        descriptor_jacobian_1,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_2,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_2, threshold_factor),
        descriptor_jacobian_2,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_descriptor_residual_1, threshold_factor),  // NOTE: Using the 1st residual only
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

void CallAccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradientXY(
    cudaStream_t stream,
    bool debug,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterProjector& color_center_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer,
    const CUDABuffer_<float>& H_buffer,
    const CUDABuffer_<float>& b_buffer,
    CUDABuffer_<float>* debug_residual_image) {
  COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
    CUDA_AUTO_TUNE_2D_TEMPLATED(
        AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradientXY,
        32, 32,
        surfel_depth.width(), surfel_depth.height(),
        0, stream,
        TEMPLATE_ARGUMENTS(block_width, block_height, _debug, _use_depth_residuals, _use_descriptor_residuals),
        /* kernel parameters */
        depth_projector,
        color_center_projector,
        depth_unprojector,
        baseline_fx,
        depth_to_color,
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth,
        surfel_normals,
        surfel_color,
        frame_depth,
        frame_normals,
        frame_color,
        residual_count_buffer,
        residual_buffer,
        H_buffer,
        b_buffer,
        debug_residual_image ? *debug_residual_image : CUDABuffer_<float>()));
}

template <int block_width, int block_height, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsFromFeaturesCUDAKernel(
    PixelCornerProjector depth_projector,
    PixelCenterProjector color_center_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> surfel_feature, // 2.10
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<float> frame_feature, // 2.10
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer,
    CUDABuffer_<float> debug_residual_image) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float depth_jacobian[6];
  float raw_depth_residual;
  // float descriptor_jacobian_1[6];
  // float descriptor_jacobian_2[6];
  // float raw_descriptor_residual_1;
  // float raw_descriptor_residual_2;
  // 2.10 
  float raw_descriptor_residual_vec[kSurfelNumDescriptor]={0};
  float jacobian_all[6*kSurfelNumDescriptor] = {0}; 
  
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                ComputeRawDepthResidualAndJacobian(
                    depth_unprojector,
                    px,
                    py,
                    pixel_calibrated_depth,
                    depth_residual_inv_stddev,
                    surfel_local_position,
                    surfel_local_normal,
                    &raw_depth_residual,
                    depth_jacobian);
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {
                  // TODO: De-duplicate this with the identical code below in this file
                  // Compute descriptor in surfel image
                  /*
                  const float intensity = 1 / 255.f * surfel_color(y, x);
                  const float t1_intensity = 1 / 255.f * surfel_color(y, x + 1);
                  const float t2_intensity = 1 / 255.f * surfel_color(y + 1, x);
                  
                  float surfel_descriptor_1 = (180.f * (t1_intensity - intensity));
                  float surfel_descriptor_2 = (180.f * (t2_intensity - intensity));
                  */

                  // 2.10: compute descriptor in surfel feature
                  float surfel_descriptor[kSurfelNumDescriptor] = {0};
                  float zero_descriptor[kSurfelNumDescriptor] = {0}; // Just for initialization of surfel descriptor
                  TestComputeRawFeatureDescriptorResidualIntpixel(
                    surfel_feature,
                    make_int2(x,y),
                    make_int2(x+1, y),
                    make_int2(x, y+1),
                    zero_descriptor,
                    surfel_descriptor/*the raw_residual_vec inside the function is used as surfel descriptor here*/);

                  // Transform the two offset points to the target / estimate frame.
                  // In order not to require depth estimates at both offset pixels,
                  // we estimate their depth using the center pixel's normal.
                  float3 surfel_normal = U16ToImageSpaceNormal(surfel_normals(y, x));
                  const float plane_d =
                      (depth_unprojector.nx(x) * surfel_calibrated_depth) * surfel_normal.x +
                      (depth_unprojector.ny(y) * surfel_calibrated_depth) * surfel_normal.y + surfel_calibrated_depth * surfel_normal.z;
                  
                  float x_plus_1_depth = plane_d / (depth_unprojector.nx(x + 1) * surfel_normal.x + depth_unprojector.ny(y) * surfel_normal.y + surfel_normal.z);
                  float3 x_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x + 1, y, x_plus_1_depth);
                  float2 pxy_t1 = depth_projector.Project(x_plus_1_local_position);
                  int t1_px = static_cast<int>(pxy_t1.x);
                  int t1_py = static_cast<int>(pxy_t1.y);
                  if (pxy_t1.x < 0 || pxy_t1.y < 0 ||
                      // t1_px < 0 || t1_py < 0 ||
                      t1_px >= frame_depth.width() || t1_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float y_plus_1_depth = plane_d / (depth_unprojector.nx(x) * surfel_normal.x + depth_unprojector.ny(y + 1) * surfel_normal.y + surfel_normal.z);
                  float3 y_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x, y + 1, y_plus_1_depth);
                  float2 pxy_t2 = depth_projector.Project(y_plus_1_local_position);
                  int t2_px = static_cast<int>(pxy_t2.x);
                  int t2_py = static_cast<int>(pxy_t2.y);
                  if (pxy_t2.x < 0 || pxy_t2.y < 0 ||
                      // t2_px < 0 || t2_py < 0 ||
                      t2_px >= frame_depth.width() || t2_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float2 color_pxy, color_pxy_t1, color_pxy_t2;
                  if (visible &&
                      x_plus_1_local_position.z > 0 &&
                      y_plus_1_local_position.z > 0 &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy) &&
                      TransformDepthToColorPixelCorner(pxy_t1, depth_to_color, &color_pxy_t1) &&
                      TransformDepthToColorPixelCorner(pxy_t2, depth_to_color, &color_pxy_t2)) {
                    /*ComputeRawDescriptorResidualAndJacobianWithFloatTexture(
                        color_center_projector,
                        frame_color,
                        color_pxy,
                        color_pxy_t1,
                        color_pxy_t2,
                        surfel_local_position,
                        surfel_descriptor_1,
                        surfel_descriptor_2,
                        &raw_descriptor_residual_1,
                        &raw_descriptor_residual_2,
                        descriptor_jacobian_1,
                        descriptor_jacobian_2);*/
                      TestComputeRawFeatureDescriptorResidual(
                        frame_feature,
                        color_pxy,
                        color_pxy_t1,
                        color_pxy_t2,
                        surfel_descriptor,
                        raw_descriptor_residual_vec
                      );
                      for (int channel = 0; channel < kTotalChannels; ++channel){
                          TestComputeRawDescriptorFeatureJacobian(
                            frame_feature,
                            color_center_projector,
                            color_pxy,
                            color_pxy_t1, color_pxy_t2,
                            surfel_local_position,
                            jacobian_all,
                            channel /* channel*/);
                        }
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Write residual debug image?
  if (debug && x < surfel_depth.width() && y < surfel_depth.height()) {
    debug_residual_image(y, x) =
        visible ?
        ((use_depth_residuals ? ComputeWeightedDepthResidual(raw_depth_residual) : 0) +
         (use_descriptor_residuals ? ComputeWeightedDescriptorResidual(/*raw_descriptor_residual_1*/raw_descriptor_residual_vec[0]) : 0)) :  // NOTE: Using the 1st residual only
        CUDART_NAN_F;
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_depth_residual,
        ComputeDepthResidualWeight(raw_depth_residual, threshold_factor),
        depth_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  if (use_descriptor_residuals) {/*
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_1,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_1, threshold_factor),
        descriptor_jacobian_1,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_2,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_2, threshold_factor),
        descriptor_jacobian_2,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);*/

    for (int channel = 0; channel < kTotalChannels; ++channel){
      AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_vec[channel],
        ComputeDescriptorResidualWeight(raw_descriptor_residual_vec[channel], threshold_factor),
        jacobian_all+6*channel, // pass the address of jacobian_c_1[0]
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
      
      AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_vec[channel + kTotalChannels], // channel_i + N is residual_2 for each channel
        ComputeDescriptorResidualWeight(raw_descriptor_residual_vec[channel + kTotalChannels], threshold_factor),
        jacobian_all + 6*kTotalChannels + 6*channel, // pass the address of jacobian_c_2[0]
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    }
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_descriptor_residual_vec[0], threshold_factor),  // NOTE: Using the 1st residual only
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

// 2.10
void CallAccumulatePoseEstimationCoeffsFromFeaturesCUDAKernel(
  cudaStream_t stream,
  bool debug,
  bool use_depth_residuals,
  bool use_descriptor_residuals,
  const PixelCornerProjector& depth_projector,
  const PixelCenterProjector& color_center_projector,
  const PixelCenterUnprojector& depth_unprojector,
  float baseline_fx,
  const DepthToColorPixelCorner& depth_to_color,
  float threshold_factor,
  const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
  const CUDABuffer_<float>& surfel_depth,
  const CUDABuffer_<u16>& surfel_normals,
  const CUDABuffer_<u8>& surfel_color,
  const CUDABuffer_<float>& surfel_feature, // 2.10
  const CUDABuffer_<float>& frame_depth,
  const CUDABuffer_<u16>& frame_normals,
  cudaTextureObject_t frame_color,
  const CUDABuffer_<float>& frame_feature, // 2.10
  const CUDABuffer_<u32>& residual_count_buffer,
  const CUDABuffer_<float>& residual_buffer,
  const CUDABuffer_<float>& H_buffer,
  const CUDABuffer_<float>& b_buffer,
  CUDABuffer_<float>* debug_residual_image) {
COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
  CUDA_AUTO_TUNE_2D_TEMPLATED(
      AccumulatePoseEstimationCoeffsFromFeaturesCUDAKernel,
      32, 32,
      surfel_depth.width(), surfel_depth.height(),
      0, stream,
      TEMPLATE_ARGUMENTS(block_width, block_height, _debug, _use_depth_residuals, _use_descriptor_residuals),
      /* kernel parameters */
      depth_projector,
      color_center_projector,
      depth_unprojector,
      baseline_fx,
      depth_to_color,
      threshold_factor,
      estimate_frame_T_surfel_frame,
      surfel_depth,
      surfel_normals,
      surfel_color,
      surfel_feature, // 2.10
      frame_depth,
      frame_normals,
      frame_color,
      frame_feature, // 2.10
      residual_count_buffer,
      residual_buffer,
      H_buffer,
      b_buffer,
      debug_residual_image ? *debug_residual_image : CUDABuffer_<float>()));
}

template <int block_width, int block_height, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsFromFeatures1PointCUDAKernel(
    PixelCornerProjector depth_projector,
    PixelCenterProjector color_center_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    float rf_weight, // 5.20 
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> surfel_feature, // 2.10
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<float> frame_feature, // 2.10
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer,
    CUDABuffer_<float> debug_residual_image) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float depth_jacobian[6];
  float raw_depth_residual;
  // 2.10 
  float raw_descriptor_residual_vec[kSurfelNumDescriptor]={0};
  float jacobian_all[6*kSurfelNumDescriptor] = {0}; 
  // 5.26
  float wg = 1.f;
  float wf = 1.f;
  
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                ComputeRawDepthResidualAndJacobian(
                    depth_unprojector,
                    px,
                    py,
                    pixel_calibrated_depth,
                    depth_residual_inv_stddev,
                    surfel_local_position,
                    surfel_local_normal,
                    &raw_depth_residual,
                    depth_jacobian);
                // 5.26
                if (kGeomResidualChannel > 0){
                  wg = surfel_feature(y, x + kTotalChannels*surfel_depth.width())*frame_feature(py, px + kTotalChannels*surfel_depth.width());
                }
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {

                  // 2.10: compute descriptor in surfel feature
                  float surfel_descriptor[kSurfelNumDescriptor] = {0};
                  float zero_descriptor[kSurfelNumDescriptor] = {0}; // Just for initialization of surfel descriptor
                  ComputeRawFeatureDescriptor1PointResidualIntpixel(
                    surfel_feature,
                    make_int2(x,y),
                    zero_descriptor,
                    surfel_descriptor/*the raw_residual_vec inside the function is used as surfel descriptor here*/);
                  
                  float2 color_pxy;
                  if (visible &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy)) {
                        ComputeRawFeatureDescriptor1PointResidualFloatpixel(
                        frame_feature,
                        color_pxy,
                        surfel_descriptor,
                        raw_descriptor_residual_vec
                      );
                      // 5.26
                      if (kFeatResidualChannel > 0){
                        wf = surfel_feature(y, x + (kTotalChannels+kGeomResidualChannel)*surfel_depth.width())*BilinearInterpolateFeatureWeight(frame_feature, color_pxy.x, color_pxy.y);
                      }
                      for (int channel = 0; channel < kTotalChannels; ++channel){
                        ComputeRawDescriptor1PointFeatureJacobian(
                            frame_feature,
                            color_center_projector,
                            color_pxy,
                            surfel_local_position,
                            jacobian_all,
                            channel /* channel*/);
                        }
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Write residual debug image?
  if (debug && x < surfel_depth.width() && y < surfel_depth.height()) {
    debug_residual_image(y, x) =
        visible ?
        ((use_depth_residuals ? ComputeWeightedDepthResidual(raw_depth_residual) : 0) +
         (use_descriptor_residuals ? ComputeWeightedDescriptorResidual(/*raw_descriptor_residual_1*/raw_descriptor_residual_vec[0]) : 0)) :  // NOTE: Using the 1st residual only
        CUDART_NAN_F;
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_depth_residual,
        ComputeDepthResidualWeight(raw_depth_residual, threshold_factor)*wg,
        depth_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  if (use_descriptor_residuals) {
    // 5.11 compute the weight only once, compute the weight for err^2, instead of each entry. i.e. the residual weight is only pointwise, not channelwise and pointwise
    float raw_residual_squared_sum = 0;
    for (int channel = 0; channel < kTotalChannels; ++channel){
      raw_residual_squared_sum += (raw_descriptor_residual_vec[channel]*raw_descriptor_residual_vec[channel]);
    }
    float DescriptorResidualWeight = ComputeDescriptorResidualWeightParam(raw_residual_squared_sum, rf_weight, threshold_factor);
    for (int channel = 0; channel < kTotalChannels; ++channel){
      AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_vec[channel],
        // ComputeDescriptorResidualWeight(raw_descriptor_residual_vec[channel], threshold_factor),
        DescriptorResidualWeight*wf, 
        jacobian_all+6*channel, // pass the address of jacobian_c_1[0]
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    }
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidualParam(raw_descriptor_residual_vec[0], rf_weight,threshold_factor),  // NOTE: Using the 1st residual only
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

// 3.29
void CallAccumulatePoseEstimationCoeffsFromFeatures1PointCUDAKernel(
  cudaStream_t stream,
  bool debug,
  bool use_depth_residuals,
  bool use_descriptor_residuals,
  const PixelCornerProjector& depth_projector,
  const PixelCenterProjector& color_center_projector,
  const PixelCenterUnprojector& depth_unprojector,
  float baseline_fx,
  const DepthToColorPixelCorner& depth_to_color,
  float threshold_factor,
  float rf_weight, // 5.20
  const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
  const CUDABuffer_<float>& surfel_depth,
  const CUDABuffer_<u16>& surfel_normals,
  const CUDABuffer_<u8>& surfel_color,
  const CUDABuffer_<float>& surfel_feature, // 2.10
  const CUDABuffer_<float>& frame_depth,
  const CUDABuffer_<u16>& frame_normals,
  cudaTextureObject_t frame_color,
  const CUDABuffer_<float>& frame_feature, // 2.10
  const CUDABuffer_<u32>& residual_count_buffer,
  const CUDABuffer_<float>& residual_buffer,
  const CUDABuffer_<float>& H_buffer,
  const CUDABuffer_<float>& b_buffer,
  CUDABuffer_<float>* debug_residual_image) {
COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
  CUDA_AUTO_TUNE_2D_TEMPLATED(
      AccumulatePoseEstimationCoeffsFromFeatures1PointCUDAKernel,
      32, 32,
      surfel_depth.width(), surfel_depth.height(),
      0, stream,
      TEMPLATE_ARGUMENTS(block_width, block_height, _debug, _use_depth_residuals, _use_descriptor_residuals),
      /* kernel parameters */
      depth_projector,
      color_center_projector,
      depth_unprojector,
      baseline_fx,
      depth_to_color,
      threshold_factor,
      rf_weight, // 5.20
      estimate_frame_T_surfel_frame,
      surfel_depth,
      surfel_normals,
      surfel_color,
      surfel_feature, // 2.10
      frame_depth,
      frame_normals,
      frame_color,
      frame_feature, // 2.10
      residual_count_buffer,
      residual_buffer,
      H_buffer,
      b_buffer,
      debug_residual_image ? *debug_residual_image : CUDABuffer_<float>()));
}



template <int block_width, int block_height, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradMag(
    PixelCornerProjector depth_projector,
    PixelCenterProjector color_center_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer,
    CUDABuffer_<float> debug_residual_image) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float depth_jacobian[6];
  float raw_depth_residual;
  float descriptor_jacobian[6];
  float raw_descriptor_residual;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                ComputeRawDepthResidualAndJacobian(
                    depth_unprojector,
                    px,
                    py,
                    pixel_calibrated_depth,
                    depth_residual_inv_stddev,
                    surfel_local_position,
                    surfel_local_normal,
                    &raw_depth_residual,
                    depth_jacobian);
              }
              
              if (use_descriptor_residuals) {
                float2 color_pxy;
                if (TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy)) {
                  ComputeRawColorResidualAndJacobian(
                      color_center_projector,
                      frame_color,
                      color_pxy,
                      surfel_local_position,
                      surfel_color(y, x),
                      &raw_descriptor_residual,
                      descriptor_jacobian);
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Write residual debug image?
  if (debug && x < surfel_depth.width() && y < surfel_depth.height()) {
    debug_residual_image(y, x) =
        visible ?
        ((use_depth_residuals ? ComputeWeightedDepthResidual(raw_depth_residual) : 0) +
         (use_descriptor_residuals ? ComputeWeightedDescriptorResidual(raw_descriptor_residual) : 0)) :
        CUDART_NAN_F;
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_depth_residual,
        ComputeDepthResidualWeight(raw_depth_residual, threshold_factor),
        depth_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  if (use_descriptor_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual,
        ComputeDescriptorResidualWeight(raw_descriptor_residual, threshold_factor),
        descriptor_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_descriptor_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

void CallAccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradMag(
    cudaStream_t stream,
    bool debug,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterProjector& color_center_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer,
    const CUDABuffer_<float>& H_buffer,
    const CUDABuffer_<float>& b_buffer,
    CUDABuffer_<float>* debug_residual_image) {
  COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
    CUDA_AUTO_TUNE_2D_TEMPLATED(
        AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradMag,
        32, 32,
        surfel_depth.width(), surfel_depth.height(),
        0, stream,
        TEMPLATE_ARGUMENTS(block_width, block_height, _debug, _use_depth_residuals, _use_descriptor_residuals),
        /* kernel parameters */
        depth_projector,
        color_center_projector,
        depth_unprojector,
        baseline_fx,
        depth_to_color,
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth,
        surfel_normals,
        surfel_color,
        frame_depth,
        frame_normals,
        frame_color,
        residual_count_buffer,
        residual_buffer,
        H_buffer,
        b_buffer,
        debug_residual_image ? *debug_residual_image : CUDABuffer_<float>()));
}


template <int block_width, int block_height, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void ComputeCostAndResidualCountFromImagesCUDAKernel_GradientXY(
    PixelCornerProjector depth_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float raw_depth_residual;
  float raw_descriptor_residual_1;
  float raw_descriptor_residual_2;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      // 2.8 project sufels (in its own local frame, i.e. base kf) to the estimated frame (current tracking frame)
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                float3 local_unproj;
                ComputeRawDepthResidual(depth_unprojector, px, py, pixel_calibrated_depth,
                                        depth_residual_inv_stddev,
                                        surfel_local_position, surfel_local_normal,
                                        &local_unproj, &raw_depth_residual);
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {
                  // Compute descriptor in surfel image // 2.7 jzmTODO:
                  const float intensity = 1 / 255.f * surfel_color(y, x); //2.7 TODO: use features of base kf
                  const float t1_intensity = 1 / 255.f * surfel_color(y, x + 1);
                  const float t2_intensity = 1 / 255.f * surfel_color(y + 1, x);
                  
                  float surfel_descriptor_1 = (180.f * (t1_intensity - intensity));
                  float surfel_descriptor_2 = (180.f * (t2_intensity - intensity));
                  
                  // Transform the two offset points to the target / estimate frame.
                  // In order not to require depth estimates at both offset pixels,
                  // we estimate their depth using the center pixel's normal.
                  float3 surfel_normal = U16ToImageSpaceNormal(surfel_normals(y, x));
                  const float plane_d =
                      (depth_unprojector.nx(x) * surfel_calibrated_depth) * surfel_normal.x +
                      (depth_unprojector.ny(y) * surfel_calibrated_depth) * surfel_normal.y + surfel_calibrated_depth * surfel_normal.z;
                  
                  float x_plus_1_depth = plane_d / (depth_unprojector.nx(x + 1) * surfel_normal.x + depth_unprojector.ny(y) * surfel_normal.y + surfel_normal.z);
                  float3 x_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x + 1, y, x_plus_1_depth);
                  float2 pxy_t1 = depth_projector.Project(x_plus_1_local_position);
                  int t1_px = static_cast<int>(pxy_t1.x);
                  int t1_py = static_cast<int>(pxy_t1.y);
                  if (pxy_t1.x < 0 || pxy_t1.y < 0 ||
                      // t1_px < 0 || t1_py < 0 ||
                      t1_px >= frame_depth.width() || t1_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float y_plus_1_depth = plane_d / (depth_unprojector.nx(x) * surfel_normal.x + depth_unprojector.ny(y + 1) * surfel_normal.y + surfel_normal.z);
                  float3 y_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x, y + 1, y_plus_1_depth);
                  float2 pxy_t2 = depth_projector.Project(y_plus_1_local_position);
                  int t2_px = static_cast<int>(pxy_t2.x);
                  int t2_py = static_cast<int>(pxy_t2.y);
                  if (pxy_t2.x < 0 || pxy_t2.y < 0 ||
                      // t2_px < 0 || t2_py < 0 ||
                      t2_px >= frame_depth.width() || t2_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float2 color_pxy, color_pxy_t1, color_pxy_t2;
                  if (visible &&
                      x_plus_1_local_position.z > 0 &&
                      y_plus_1_local_position.z > 0 &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy) &&
                      TransformDepthToColorPixelCorner(pxy_t1, depth_to_color, &color_pxy_t1) &&
                      TransformDepthToColorPixelCorner(pxy_t2, depth_to_color, &color_pxy_t2)) {
                    ComputeRawDescriptorResidualWithFloatTexture( // 2.8 jzmTODO: use feature descriptors
                        frame_color,
                        color_pxy,
                        color_pxy_t1,
                        color_pxy_t2,
                        surfel_descriptor_1,
                        surfel_descriptor_2,
                        &raw_descriptor_residual_1,
                        &raw_descriptor_residual_2);
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
  
  if (use_descriptor_residuals) {
    // TODO: It should be possible to merge these two calls and directly accumulate the sum (also use 2 for the residual count then).
    //       It should even be possible to merge it with the depth residual call as well in case both residual types are used.
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual_1, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual_2, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
}

void ComputeCostAndResidualCountFromImagesCUDAKernel_GradientXY(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer) {
  COMPILE_OPTION_2(use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_2D_TEMPLATED(
          ComputeCostAndResidualCountFromImagesCUDAKernel_GradientXY,
          32, 32,
          surfel_depth.width(), surfel_depth.height(),
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, block_height, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          depth_projector,
          depth_unprojector,
          baseline_fx,
          depth_to_color,
          threshold_factor,
          estimate_frame_T_surfel_frame,
          surfel_depth,
          surfel_normals,
          surfel_color,
          frame_depth,
          frame_normals,
          frame_color,
          residual_count_buffer,
          residual_buffer));
}


template <int block_width, int block_height, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void ComputeCostAndResidualCountFromImagesCUDAKernel_GradMag(
    PixelCornerProjector depth_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float raw_depth_residual;
  float raw_descriptor_residual;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                float3 local_unproj;
                ComputeRawDepthResidual(depth_unprojector, px, py, pixel_calibrated_depth,
                                        depth_residual_inv_stddev,
                                        surfel_local_position, surfel_local_normal,
                                        &local_unproj, &raw_depth_residual);
              }
              
              if (use_descriptor_residuals) {
                float2 color_pxy;
                if (TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy)) {
                  ComputeRawColorResidual(frame_color, color_pxy, surfel_color(y, x), &raw_descriptor_residual);
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
  
  if (use_descriptor_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
}

void CallComputeCostAndResidualCountFromImagesCUDAKernel_GradMag(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer) {
  COMPILE_OPTION_2(use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_2D_TEMPLATED(
          ComputeCostAndResidualCountFromImagesCUDAKernel_GradMag,
          32, 32,
          surfel_depth.width(), surfel_depth.height(),
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, block_height, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          depth_projector,
          depth_unprojector,
          baseline_fx,
          depth_to_color,
          threshold_factor,
          estimate_frame_T_surfel_frame,
          surfel_depth,
          surfel_normals,
          surfel_color,
          frame_depth,
          frame_normals,
          frame_color,
          residual_count_buffer,
          residual_buffer));
}

// 2.10
template <int block_width, int block_height, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void ComputeCostAndResidualCountFromFeaturesCUDAKernel(
    PixelCornerProjector depth_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> surfel_feature, // 2.10
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<float> frame_feature, // 2.10
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float raw_depth_residual;
  // float raw_descriptor_residual_1;
  // float raw_descriptor_residual_2;
  float raw_descriptor_residual_vec[kSurfelNumDescriptor] = {0}; //2.10
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      // 2.8 project sufels (in its own local frame, i.e. base kf) to the estimated frame (current tracking frame)
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) { // 2.25 get surfel position in the coorinate system of the frame being tracked
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) { // 2.11 px, py are ensured to be within the image bounds if the 'if' statement holds. They are casted from pxy. But px+1, py+1 can be out of bounds?
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                float3 local_unproj;
                ComputeRawDepthResidual(depth_unprojector, px, py, pixel_calibrated_depth,
                                        depth_residual_inv_stddev,
                                        surfel_local_position, surfel_local_normal,
                                        &local_unproj, &raw_depth_residual);
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {
                  // Compute descriptor in surfel image // 2.7 jzmTODO:
                  /*const float intensity = 1 / 255.f * surfel_color(y, x); //2.7 TODO: use features of base kf
                  const float t1_intensity = 1 / 255.f * surfel_color(y, x + 1);
                  const float t2_intensity = 1 / 255.f * surfel_color(y + 1, x);
                  
                  float surfel_descriptor_1 = (180.f * (t1_intensity - intensity));
                  float surfel_descriptor_2 = (180.f * (t2_intensity - intensity));*/

                  // 2.10: compute descriptor in surfel feature
                  float surfel_descriptor[kSurfelNumDescriptor] = {0};
                  float zero_descriptor[kSurfelNumDescriptor] = {0}; // Just for initialization of surfel descriptor
                  TestComputeRawFeatureDescriptorResidualIntpixel(
                    surfel_feature,
                    make_int2(x,y),
                    make_int2(x+1, y), // this is guanranteed
                    make_int2(x, y+1),
                    zero_descriptor,
                    surfel_descriptor/*the raw_residual_vec is used as surfel descriptor here*/);
                  
                  // Transform the two offset points to the target / estimate frame.
                  // In order not to require depth estimates at both offset pixels,
                  // we estimate their depth using the center pixel's normal.
                  float3 surfel_normal = U16ToImageSpaceNormal(surfel_normals(y, x));
                  const float plane_d =
                      (depth_unprojector.nx(x) * surfel_calibrated_depth) * surfel_normal.x +
                      (depth_unprojector.ny(y) * surfel_calibrated_depth) * surfel_normal.y + surfel_calibrated_depth * surfel_normal.z;
                  
                  float x_plus_1_depth = plane_d / (depth_unprojector.nx(x + 1) * surfel_normal.x + depth_unprojector.ny(y) * surfel_normal.y + surfel_normal.z);
                  float3 x_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x + 1, y, x_plus_1_depth);
                  float2 pxy_t1 = depth_projector.Project(x_plus_1_local_position);
                  int t1_px = static_cast<int>(pxy_t1.x);
                  int t1_py = static_cast<int>(pxy_t1.y);
                  if (pxy_t1.x < 0 || pxy_t1.y < 0 ||
                      // t1_px < 0 || t1_py < 0 ||
                      t1_px >= frame_depth.width() || t1_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float y_plus_1_depth = plane_d / (depth_unprojector.nx(x) * surfel_normal.x + depth_unprojector.ny(y + 1) * surfel_normal.y + surfel_normal.z);
                  float3 y_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x, y + 1, y_plus_1_depth);
                  float2 pxy_t2 = depth_projector.Project(y_plus_1_local_position);
                  int t2_px = static_cast<int>(pxy_t2.x);
                  int t2_py = static_cast<int>(pxy_t2.y);
                  if (pxy_t2.x < 0 || pxy_t2.y < 0 ||
                      // t2_px < 0 || t2_py < 0 ||
                      t2_px >= frame_depth.width() || t2_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float2 color_pxy, color_pxy_t1, color_pxy_t2;
                  if (visible &&
                      x_plus_1_local_position.z > 0 &&
                      y_plus_1_local_position.z > 0 &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy) &&
                      TransformDepthToColorPixelCorner(pxy_t1, depth_to_color, &color_pxy_t1) &&
                      TransformDepthToColorPixelCorner(pxy_t2, depth_to_color, &color_pxy_t2)) {
                    /*ComputeRawDescriptorResidualWithFloatTexture( // 2.8 jzmTODO: use feature descriptors
                        frame_color,
                        color_pxy,
                        color_pxy_t1,
                        color_pxy_t2,
                        surfel_descriptor_1,
                        surfel_descriptor_2,
                        &raw_descriptor_residual_1,
                        &raw_descriptor_residual_2);*/
                      TestComputeRawFeatureDescriptorResidual(
                        frame_feature,
                        color_pxy,
                        color_pxy_t1, // 2.11 TODO: check if this is beyound the scope of current feature scale. update: yes, some indices go out of bounds. clamped
                        color_pxy_t2, // 2.11 TODO: pass the shape of frame_feature to the fetching function? update: no, use buffer.width() and buffer.height()
                        surfel_descriptor,
                        raw_descriptor_residual_vec
                      );
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
  
  if (use_descriptor_residuals) {
    // TODO: It should be possible to merge these two calls and directly accumulate the sum (also use 2 for the residual count then).
    //       It should even be possible to merge it with the depth residual call as well in case both residual types are used.
    /*AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual_1, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual_2, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  */
  // 5.11 compute the weight only once, compute the weight for err^2, instead of each entry. i.e. the residual weight is only pointwise, not channelwise and pointwise
  float raw_residual_squared_sum = 0;
  for (int channel = 0; channel < kTotalChannels; ++channel){
    raw_residual_squared_sum += (raw_descriptor_residual_vec[channel]*raw_descriptor_residual_vec[channel]);
  }
  float DescriptorResidual = ComputeWeightedDescriptorResidual(raw_residual_squared_sum, threshold_factor);
  
  for (int channel = 0; channel < kTotalChannels; ++channel){
    AccumulatePoseResidualAndCount<block_width, block_height>(
      visible,
      // ComputeWeightedDescriptorResidual(raw_descriptor_residual_vec[channel], threshold_factor),
      DescriptorResidual,
      residual_count_buffer,
      residual_buffer,
      &temp_storage.float_storage,
      &temp_storage.int_storage);
  }
  }
}

void ComputeCostAndResidualCountFromFeaturesCUDAKernel(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& surfel_feature, // 2.10
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<float>& frame_feature, // 2.10
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer) {
  COMPILE_OPTION_2(use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_2D_TEMPLATED(
          ComputeCostAndResidualCountFromFeaturesCUDAKernel,
          32, 32,
          surfel_depth.width(), surfel_depth.height(),
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, block_height, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          depth_projector,
          depth_unprojector,
          baseline_fx,
          depth_to_color,
          threshold_factor,
          estimate_frame_T_surfel_frame,
          surfel_depth,
          surfel_normals,
          surfel_color,
          surfel_feature, // 2.10
          frame_depth,
          frame_normals,
          frame_color,
          frame_feature, // 2.10
          residual_count_buffer,
          residual_buffer));
}


// 3.29
template <int block_width, int block_height, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void ComputeCostAnd1PointResidualCountFromFeaturesCUDAKernel(
    PixelCornerProjector depth_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    float rf_weight, // 5.20
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> surfel_feature, // 2.10
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<float> frame_feature, // 2.10
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float raw_depth_residual;
  float raw_descriptor_residual_vec[kSurfelNumDescriptor] = {0}; //2.10
  float wg = 1.f; // 5.26 initialization 
  float wf = 1.f;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      // 2.8 project sufels (in its own local frame, i.e. base kf) to the estimated frame (current tracking frame)
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) { // 2.25 get surfel position in the coorinate system of the frame being tracked
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) { // 2.11 px, py are ensured to be within the image bounds if the 'if' statement holds. They are casted from pxy. But px+1, py+1 can be out of bounds?
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>( // 3.29 surfel_proejection_nvcc_only.cuh : 177
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                if (kGeomResidualChannel > 0){
                  wg = surfel_feature(y, kTotalChannels*surfel_depth.width()+x) * frame_feature(py, kTotalChannels*surfel_depth.width()+px);
               }
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                float3 local_unproj;
                ComputeRawDepthResidual(depth_unprojector, px, py, pixel_calibrated_depth,
                                        depth_residual_inv_stddev,
                                        surfel_local_position, surfel_local_normal,
                                        &local_unproj, &raw_depth_residual);
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {
                  // Compute descriptor in surfel image // 2.7 jzmTODO:
                  /*const float intensity = 1 / 255.f * surfel_color(y, x); //2.7 TODO: use features of base kf
                  const float t1_intensity = 1 / 255.f * surfel_color(y, x + 1);
                  const float t2_intensity = 1 / 255.f * surfel_color(y + 1, x);
                  
                  float surfel_descriptor_1 = (180.f * (t1_intensity - intensity));
                  float surfel_descriptor_2 = (180.f * (t2_intensity - intensity));*/

                  // 2.10: compute descriptor in surfel feature
                  float surfel_descriptor[kSurfelNumDescriptor] = {0};
                  float zero_descriptor[kSurfelNumDescriptor] = {0}; // Just for initialization of surfel descriptor
                  ComputeRawFeatureDescriptor1PointResidualIntpixel(
                    surfel_feature,
                    make_int2(x,y),
                    zero_descriptor,
                    surfel_descriptor/*the raw_residual_vec is used as surfel descriptor here*/);

                  float2 color_pxy;
                  if (visible &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy)) {
                        ComputeRawFeatureDescriptor1PointResidualFloatpixel(
                        frame_feature,
                        color_pxy,
                        surfel_descriptor,
                        raw_descriptor_residual_vec
                      );
                      // 5.26
                      if (kFeatResidualChannel > 0){
                        wf = surfel_feature(y, x + (kTotalChannels+kGeomResidualChannel)*surfel_depth.width()) * BilinearInterpolateFeatureWeight(frame_feature, color_pxy.x, color_pxy.y);
                      }          
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor)*wg,
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
  
  if (use_descriptor_residuals) {
    // TODO: It should be possible to merge these two calls and directly accumulate the sum (also use 2 for the residual count then).
    //       It should even be possible to merge it with the depth residual call as well in case both residual types are used.
  // 5.11 compute the weight only once, compute the weight for err^2, instead of each entry. i.e. the residual weight is only pointwise, not channelwise and pointwise
  float raw_residual_squared_sum = 0;
  for (int channel = 0; channel < kTotalChannels; ++channel){
    raw_residual_squared_sum += (raw_descriptor_residual_vec[channel]*raw_descriptor_residual_vec[channel]);
  }
  float DescriptorResidual = ComputeWeightedDescriptorResidualParam(raw_residual_squared_sum, rf_weight, threshold_factor);
    // for (int channel = 0; channel < kTotalChannels; ++channel){
    AccumulatePoseResidualAndCount<block_width, block_height>(
      visible,
      // ComputeWeightedDescriptorResidual(raw_descriptor_residual_vec[channel], threshold_factor),
      DescriptorResidual*wf,
      residual_count_buffer,
      residual_buffer,
      &temp_storage.float_storage,
      &temp_storage.int_storage);
  // }
  }
}

// 3.29 Computing 1-point residual instead of gradient residual
void ComputeCostAnd1PointResidualCountFromFeaturesCUDAKernel(
  cudaStream_t stream,
  bool use_depth_residuals,
  bool use_descriptor_residuals,
  const PixelCornerProjector& depth_projector,
  const PixelCenterUnprojector& depth_unprojector,
  float baseline_fx,
  const DepthToColorPixelCorner& depth_to_color,
  float threshold_factor,
  float rf_weight, // 5.20
  const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
  const CUDABuffer_<float>& surfel_depth,
  const CUDABuffer_<u16>& surfel_normals,
  const CUDABuffer_<u8>& surfel_color,
  const CUDABuffer_<float>& surfel_feature, // 2.10
  const CUDABuffer_<float>& frame_depth,
  const CUDABuffer_<u16>& frame_normals,
  cudaTextureObject_t frame_color,
  const CUDABuffer_<float>& frame_feature, // 2.10
  const CUDABuffer_<u32>& residual_count_buffer,
  const CUDABuffer_<float>& residual_buffer) {
COMPILE_OPTION_2(use_depth_residuals, use_descriptor_residuals,
    CUDA_AUTO_TUNE_2D_TEMPLATED(
        ComputeCostAnd1PointResidualCountFromFeaturesCUDAKernel,
        32, 32,
        surfel_depth.width(), surfel_depth.height(),
        0, stream,
        TEMPLATE_ARGUMENTS(block_width, block_height, _use_depth_residuals, _use_descriptor_residuals),
        /* kernel parameters */
        depth_projector,
        depth_unprojector,
        baseline_fx,
        depth_to_color,
        threshold_factor,
        rf_weight, // 5.20
        estimate_frame_T_surfel_frame,
        surfel_depth,
        surfel_normals,
        surfel_color,
        surfel_feature, // 2.10
        frame_depth,
        frame_normals,
        frame_color,
        frame_feature, // 2.10
        residual_count_buffer,
        residual_buffer));
}

}
