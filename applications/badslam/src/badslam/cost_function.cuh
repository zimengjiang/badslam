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


#pragma once

#include <cuda_runtime.h>
#include <libvis/libvis.h>

#include "badslam/cuda_util.cuh"
#include "badslam/util.cuh"
#include "badslam/robust_weighting.cuh"
#include "badslam/kernels.cuh"
namespace vis {

// --- Depth (geometric) residual ---

// Weight factor on the depth residual in the cost term.
constexpr float kDepthResidualWeight = 1.f;

// Default Tukey parameter (= factor on standard deviation at which the
// residuals have zero weight). This gets scaled for multi-res pose estimation.
constexpr float kDepthResidualDefaultTukeyParam = 10.f;

// Expected stereo matching uncertainty in pixels in the depth estimation
// process. Determines the final propagated depth uncertainty.
constexpr float kDepthUncertaintyEmpiricalFactor = 0.1f;

// 11.17 for the number of feature channels
// constexpr int kTotalChannels = 9;

// 11.18 for clamping indices
// 12.1 jzmTODO: adjust this to match the actual input size. Remember to modify this when the feature is scaled. 
constexpr int max_x = kFeatureW-1-1; // W - 1 - 1
constexpr int max_y = kFeatureH-1-1; // H - 1 - 1

// Macro definition
// #define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d! \n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }
#define CudaAssert( X ) if ( !(X) ) {return; }



// Computes the "raw" depth (geometric) residual, i.e., without any weighting.
__forceinline__ __device__ void ComputeRawDepthResidual(
    const PixelCenterUnprojector& unprojector,
    int px,
    int py,
    float pixel_calibrated_depth,
    float raw_residual_inv_stddev_estimate,
    const float3& surfel_local_position,
    const float3& surfel_local_normal,
    float3* local_unproj,
    float* raw_residual) {
  *local_unproj = unprojector.UnprojectPoint(px, py, pixel_calibrated_depth);
  *raw_residual = raw_residual_inv_stddev_estimate * Dot(surfel_local_normal, *local_unproj - surfel_local_position);
}

// Computes the "raw" depth (geometric) residual, i.e., without any weighting.
__forceinline__ __device__ void ComputeRawDepthResidual(
    float raw_residual_inv_stddev_estimate,
    const float3& surfel_local_position,
    const float3& surfel_local_normal,
    const float3& local_unproj,
    float* raw_residual) {
  *raw_residual = raw_residual_inv_stddev_estimate * Dot(surfel_local_normal, local_unproj - surfel_local_position);
}

// Computes the propagated standard deviation estimate for the depth residual.
__forceinline__ __device__ float ComputeDepthResidualStddevEstimate(float nx, float ny, float depth, const float3& surfel_local_normal, float baseline_fx) {
  return (kDepthUncertaintyEmpiricalFactor * fabs(surfel_local_normal.x * nx + surfel_local_normal.y * ny + surfel_local_normal.z) * (depth * depth)) / baseline_fx;
}

// Computes the propagated inverse standard deviation estimate for the depth residual.
__forceinline__ __device__ float ComputeDepthResidualInvStddevEstimate(float nx, float ny, float depth, const float3& surfel_local_normal, float baseline_fx) {
  return baseline_fx / (kDepthUncertaintyEmpiricalFactor * fabs(surfel_local_normal.x * nx + surfel_local_normal.y * ny + surfel_local_normal.z) * (depth * depth));
}

// Computes the weight of the depth residual in the optimization.
__forceinline__ __device__ float ComputeDepthResidualWeight(float raw_residual, float scaling = 1.f) {
  return kDepthResidualWeight * TukeyWeight(raw_residual, scaling * kDepthResidualDefaultTukeyParam);
}

// Computes the weighted depth residual for summing up the optimization cost.
__forceinline__ __device__ float ComputeWeightedDepthResidual(float raw_residual, float scaling = 1.f) {
  return kDepthResidualWeight * TukeyResidual(raw_residual, scaling * kDepthResidualDefaultTukeyParam);
}


// --- Descriptor (photometric) residual ---

// Weight factor from the cost term.
// TODO: Tune further. Make parameter?
// constexpr float kDescriptorResidualWeight = 1e-2f;
// constexpr float kDescriptorResidualWeight = 10.f; // 4.7
constexpr float kDescriptorResidualWeight = 1.f; // 4.12


// Parameter for the Huber robust loss function for photometric residuals.
// TODO: Make parameter?
// constexpr float kDescriptorResidualHuberParameter = 10.f;
constexpr float kDescriptorResidualHuberParameter =  1.f; // 4.9 1/18


// Computes the projections in an image of two (mostly) fixed points on the
// border of a surfel, whose direction to the surfel center differs by 90
// degrees. These points are used to compute the descriptor residual.
__forceinline__ __device__ void ComputeTangentProjections(
    const float3& surfel_global_position,
    const float3& surfel_global_normal,
    const float surfel_radius_squared,
    const CUDAMatrix3x4& frame_T_global,
    const PixelCornerProjector& color_corner_projector,
    float2* t1_pxy,
    float2* t2_pxy) {
  // With scaling 1, the tangent sample points are ca. 0.5 pixels away from the
  // center point when looking at the surfel from directly above.
  // TODO: Tune this! I think this has received very little tuning, if any at all.
  constexpr float kTangentScaling = 2.0f;
  
  float3 t1;
  CrossProduct(surfel_global_normal, (fabs(surfel_global_normal.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0), &t1);
  t1 = t1 * kTangentScaling * sqrtf(surfel_radius_squared / max(1e-12f, SquaredLength(t1)));
  *t1_pxy = color_corner_projector.Project(frame_T_global * (surfel_global_position + t1));
  float3 t2;
  CrossProduct(surfel_global_normal, t1, &t2);
  t2 = t2 * kTangentScaling * sqrtf(surfel_radius_squared / max(1e-12f, SquaredLength(t2)));
  *t2_pxy = color_corner_projector.Project(frame_T_global * (surfel_global_position + t2));
}

// Computes the "raw" descriptor (photometric) residual, i.e., without any
// weighting.
__forceinline__ __device__ void ComputeRawDescriptorResidual(
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2) {
  float intensity = tex2D<float4>(color_texture, pxy.x, pxy.y).w;
  float t1_intensity = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).w; // <float4> ???
  float t2_intensity = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).w;
  
  *raw_residual_1 = (180.f * (t1_intensity - intensity)) - surfel_descriptor_1;
  *raw_residual_2 = (180.f * (t2_intensity - intensity)) - surfel_descriptor_2;
}

__forceinline__ __device__ void ComputeRawDescriptorResidualWithFloatTexture(
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2) {
  float intensity = tex2D<float>(color_texture, pxy.x, pxy.y);
  
  float t1_intensity = tex2D<float>(color_texture, t1_pxy.x, t1_pxy.y);
  float t2_intensity = tex2D<float>(color_texture, t2_pxy.x, t2_pxy.y);
  
  *raw_residual_1 = (180.f * (t1_intensity - intensity)) - surfel_descriptor_1;
  *raw_residual_2 = (180.f * (t2_intensity - intensity)) - surfel_descriptor_2;
}

// Computes the weight of the descriptor residual in the optimization.
__forceinline__ __device__ float ComputeDescriptorResidualWeight(float raw_residual, float scaling = 1.f) {
  // return scaling * kDescriptorResidualWeight * HuberWeight(raw_residual, kDescriptorResidualHuberParameter);
  return scaling * kDescriptorResidualWeight * HuberWeightSquaredResidual(raw_residual, kDescriptorResidualHuberParameter);
}

// Computes the weighted descriptor residual for summing up the optimization 
// cost.
__forceinline__ __device__ float ComputeWeightedDescriptorResidual(float raw_residual, float scaling = 1.f) {
  // return scaling * kDescriptorResidualWeight * HuberResidual(raw_residual, kDescriptorResidualHuberParameter);
  return scaling * kDescriptorResidualWeight * HuberResidualSquared(raw_residual, kDescriptorResidualHuberParameter);
}

// 5.20 
// Computes the weight of the descriptor residual in the optimization.
__forceinline__ __device__ float ComputeDescriptorResidualWeightParam(float raw_residual, float rf_weight, float scaling = 1.f) {
  // return scaling * kDescriptorResidualWeight * HuberWeight(raw_residual, kDescriptorResidualHuberParameter);
  // return scaling * kDescriptorResidualWeight * HuberWeightSquaredResidual(raw_residual, kDescriptorResidualHuberParameter);
  return scaling * rf_weight * HuberWeightSquaredResidual(raw_residual, kDescriptorResidualHuberParameter);
}

// Computes the weighted descriptor residual for summing up the optimization 
// cost.
__forceinline__ __device__ float ComputeWeightedDescriptorResidualParam(float raw_residual, float rf_weight, float scaling = 1.f) {
  // return scaling * kDescriptorResidualWeight * HuberResidual(raw_residual, kDescriptorResidualHuberParameter);
  // return scaling * kDescriptorResidualWeight * HuberResidualSquared(raw_residual, kDescriptorResidualHuberParameter);
  return scaling * rf_weight * HuberResidualSquared(raw_residual, kDescriptorResidualHuberParameter);
}

// Computes the Jacobian of a surfel descriptor with regard to changes in the
// projected pixel position of the surfel. This function makes the approximation that
// the projected positions of all points on the surfel move equally. This should
// be valid since those points should all be very close together.
__forceinline__ __device__ void DescriptorJacobianWrtProjectedPosition(
    cudaTextureObject_t color_texture,
    const float2& color_pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float* grad_x_1,
    float* grad_y_1,
    float* grad_x_2,
    float* grad_y_2) {
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f));
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  float top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  float bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  float bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  
  float center_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float center_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  

  ix = static_cast<int>(::max(0.f, t1_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t1_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t1_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t1_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  
  float t1_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t1_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  
  ix = static_cast<int>(::max(0.f, t2_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t2_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t2_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t2_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  
  float t2_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t2_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  float intensity = tex2D<float4>(color_texture, color_pxy.x, color_pxy.y).w;
  float t1_intensity = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).w;
  float t2_intensity = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).w;
  
  // NOTE: It is approximate to mix all the center, t1, t2 derivatives
  //       directly since the points would move slightly differently on most
  //       pose changes. However, the approximation is possibly pretty good since
  //       the points are all close to each other.
  
  *grad_x_1 = 180.f * (t1_dx - center_dx);
  *grad_y_1 = 180.f * (t1_dy - center_dy);
  *grad_x_2 = 180.f * (t2_dx - center_dx);
  *grad_y_2 = 180.f * (t2_dy - center_dy);
}

__forceinline__ __device__ void DescriptorJacobianWrtProjectedPositionWithFloatTexture(
    cudaTextureObject_t color_texture,
    const float2& color_pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float* grad_x_fx_1,
    float* grad_y_fy_1,
    float* grad_x_fx_2,
    float* grad_y_fy_2) {
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f));
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left = tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  float top_right = tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  float bottom_left = tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  float bottom_right = tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  float center_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float center_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  

  ix = static_cast<int>(::max(0.f, t1_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t1_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t1_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t1_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  top_right = tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  bottom_left = tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  bottom_right = tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  float t1_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t1_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  
  ix = static_cast<int>(::max(0.f, t2_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t2_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t2_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t2_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  top_right = tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  bottom_left = tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  bottom_right = tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  float t2_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t2_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  float intensity = tex2D<float>(color_texture, color_pxy.x, color_pxy.y);
  float t1_intensity = tex2D<float>(color_texture, t1_pxy.x, t1_pxy.y);
  float t2_intensity = tex2D<float>(color_texture, t2_pxy.x, t2_pxy.y);
  
  // NOTE: It is approximate to mix all the center, t1, t2 derivatives
  //       directly since the points would move slightly differently on most
  //       pose changes. However, the approximation is possibly pretty good since
  //       the points are all close to each other.
  
  *grad_x_fx_1 = 180.f * (t1_dx - center_dx);
  *grad_y_fy_1 = 180.f * (t1_dy - center_dy);
  *grad_x_fx_2 = 180.f * (t2_dx - center_dx);
  *grad_y_fy_2 = 180.f * (t2_dy - center_dy);
}


// --- Color (photometric) residual for frame-to-frame tracking on precomputed gradient magnitudes ---

// Computes the "raw" color residual, i.e., without any weighting.
__forceinline__ __device__ void ComputeRawColorResidual(
    cudaTextureObject_t color_texture,
    const float2& pxy,
    float surfel_gradmag,
    float* raw_residual) {
  *raw_residual = 255.f * tex2D<float>(color_texture, pxy.x, pxy.y) - surfel_gradmag;
}

// Computes the Jacobian of the color residual with regard to changes in the
// projected position of a 3D point.
__forceinline__ __device__ void ColorJacobianWrtProjectedPosition(
    cudaTextureObject_t color_texture,
    const float2& color_pxy,
    float* grad_x_fx,
    float* grad_y_fy) {
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f));
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left = 255.f * tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  float top_right = 255.f * tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  float bottom_left = 255.f * tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  float bottom_right = 255.f * tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  *grad_x_fx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  *grad_y_fy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
}

// 5.26 bilinear interpolation of rf weight map
__forceinline__ __device__ void TestFetchFeatureArrBilinearInterpolationVec(
    const CUDABuffer_<float>& feature_arr,
    const float& px, /*pixel corner coordinates, -> x, | y*/
    const float& py,
    float* result) {
    int ix = static_cast<int>(::max(0.f, px - 0.5f));      // i = floor(px-0.5), converting corner pixel to center pixel convention
    int iy = static_cast<int>(::max(0.f, py - 0.5f));      // j = floor(py-0.5)
    // ix = ::min(ix, feature_arr.width()/kTotalChannels-1);
    ix = ::min(ix, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1);
    iy = ::min(iy, feature_arr.height()-1);
    float alpha = ::max(0.f, ::min(1.f, px - 0.5f - ix));  // alpha = frac(px-0.5) 
    float beta = ::max(0.f, ::min(1.f, py - 0.5f - iy));   // beta = frac(py-0.5)
    // unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

    int2 top_left, top_right, bottom_left, bottom_right;
    top_left = make_int2(ix,iy);
    // top_right = make_int2(::min(ix+1, feature_arr.width()/kTotalChannels-1),iy);
    top_right = make_int2(::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1),iy);
    bottom_left = make_int2(ix,::min(iy+1, feature_arr.height()-1));
    // bottom_right = make_int2(::min(ix+1, feature_arr.width()/kTotalChannels-1),::min(iy+1, feature_arr.height()-1));
    bottom_right = make_int2(::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1),::min(iy+1, feature_arr.height()-1));

    #pragma unroll
    for (int c = 0; c < kTotalChannels; ++c){
      // c does not affect the computation of the interpolated corners. 
      // Only px and py decide the corner. c is used to fetch the correct grid of the channel value corresponding to each pixel. 
      // 11.17 jzmTODO: double check indexing, (py,px), pitch index, Achtung! out of bounds buffer access
      /*if (x >= width_ || y >= height_) {
      printf("Out of bounds buffer access at (%i, %i), buffer size (%i, %i)\n",
             x, y, width_, height_);
      }*/

      *(result + c) = (1-alpha)*(1-beta)*feature_arr(top_left.y,top_left.x*kTotalChannels+c) \
                      + alpha*(1-beta)*feature_arr(top_right.y,top_right.x*kTotalChannels+c) \
                      + beta*(1-alpha)*feature_arr(bottom_left.y,bottom_left.x*kTotalChannels+c) \
                      + alpha*beta*feature_arr(bottom_right.y, bottom_right.x*kTotalChannels+c);
    }
}

/*11.17: bilinear interpolation, same effect as filtering in tex2D fetching */
/*https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching*/
__forceinline__ __device__ float BilinearInterpolateFeatureWeight(
    const CUDABuffer_<float>& feature_arr,
    const float& px, /*pixel corner coordinates, -> x, | y*/
    const float& py) {
    float result; 
    int ix = static_cast<int>(::max(0.f, px - 0.5f));      // i = floor(px-0.5), converting corner pixel to center pixel convention
    int iy = static_cast<int>(::max(0.f, py - 0.5f));      // j = floor(py-0.5)
    // ix = ::min(ix, feature_arr.width()/kTotalChannels-1);
    ix = ::min(ix, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1);
    iy = ::min(iy, feature_arr.height()-1);
    float alpha = ::max(0.f, ::min(1.f, px - 0.5f - ix));  // alpha = frac(px-0.5) 
    float beta = ::max(0.f, ::min(1.f, py - 0.5f - iy));   // beta = frac(py-0.5)
    // unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

    int2 top_left, top_right, bottom_left, bottom_right;
    int w = feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel);
    top_left = make_int2(ix,iy);
    // top_right = make_int2(::min(ix+1, feature_arr.width()/kTotalChannels-1),iy);
    top_right = make_int2(::min(ix+1, w-1),iy);
    bottom_left = make_int2(ix,::min(iy+1, feature_arr.height()-1));
    // bottom_right = make_int2(::min(ix+1, feature_arr.width()/kTotalChannels-1),::min(iy+1, feature_arr.height()-1));
    bottom_right = make_int2(::min(ix+1, w-1),::min(iy+1, feature_arr.height()-1));
    result = (1-alpha)*(1-beta)*feature_arr(top_left.y, top_left.x + (kTotalChannels+kGeomResidualChannel)*w) \
                      + alpha*(1-beta)*feature_arr(top_right.y, top_right.x + (kTotalChannels+kGeomResidualChannel)*w) \
                      + beta*(1-alpha)*feature_arr(bottom_left.y,bottom_left.x + (kTotalChannels+kGeomResidualChannel)*w) \
                      + alpha*beta*feature_arr(bottom_right.y, bottom_right.x + (kTotalChannels+kGeomResidualChannel)*w);
    return result;
}

/*11.17: bilinear interpolation, when the number of channels = 1*/
/*https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching*/
// __forceinline__ __device__ void TestFetchFeatureArrBilinearInterpolationFloat(
//     const CUDABuffer_<float>& feature_arr,
//     const float& px, /*pixel corner coordinates, -> x, | y*/
//     const float& py,
//     float* result) {
//     int ix = static_cast<int>(::max(0.f, px - 0.5f));      // i = floor(px-0.5)
//     int iy = static_cast<int>(::max(0.f, py - 0.5f));      // j = floor(py-0.5)
//     ix = ::min(ix, feature_arr.width()/kTotalChannels-1);
//     iy = ::min(iy, feature_arr.height()-1);
//     float alpha = ::max(0.f, ::min(1.f, px - 0.5f - ix));  // alpha = frac(px-0.5) 
//     float beta = ::max(0.f, ::min(1.f, py - 0.5f - iy));   // beta = frac(py-0.5)
//     int2 top_left, top_right, bottom_left, bottom_right;
//     top_left = make_int2(ix,iy);
//     top_right = make_int2(ix+1,iy);
//     bottom_left = make_int2(ix,iy+1);
//     bottom_right = make_int2(ix+1,iy+1);
//     *(result) = (1-alpha)*(1-beta)*feature_arr(top_left.y,top_left.x)  \ 
//                     + alpha*(1-beta)*feature_arr(top_right.y,top_right.x) \ 
//                     + beta*(1-alpha)*feature_arr(bottom_left.y,bottom_left.x) \ 
//                     + alpha*beta*feature_arr(bottom_right.y, bottom_right.x);
// }

__forceinline__ __device__ void TestFetchFeatureArrVec(
    const CUDABuffer_<float>& feature_arr,
    const int& px, /*pixel corner coordinates, -> x, | y*/
    const int& py,
    float* result) {
    /*unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (surfel_index == 0){
    printf("bilinear: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
  }*/
    #pragma unroll
    for (int c = 0; c < kTotalChannels; ++c){
      *(result + c) = feature_arr(py, px*kTotalChannels+c);
    }
}

__forceinline__ __device__ void TestCheckBilinearInterpolation(
  float* raw_residual_vec,
  float* raw_residual_vec_true){
    printf("check bilinear interpolation\n");
    for (int i = 0; i < kTotalChannels; ++i){
      printf("difference i: %f \n",*(raw_residual_vec+i)-*(raw_residual_vec_true+i));
    }
}

__forceinline__ __device__ void TestComputeRawFeatureDescriptorResidual(
    const CUDABuffer_<float>& feature_arr,
    const float2& pxy, /*pixel corner convention*/
    const float2& t1_pxy,/*pixel corner convention*/
    const float2& t2_pxy,
    float* surfel_descriptor_vec,
    float* raw_residual_vec) {
  
  /*unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index == 0){
    printf("residual: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
  }*/
  // feature vectors
  float f_pxy[kTotalChannels] = {0}; // initialize all to 0, memory allocation
  float f_t1[kTotalChannels] = {0};
  float f_t2[kTotalChannels] = {0};
  TestFetchFeatureArrBilinearInterpolationVec(feature_arr, pxy.x, pxy.y, f_pxy);
  TestFetchFeatureArrBilinearInterpolationVec(feature_arr, t1_pxy.x, t1_pxy.y, f_t1);
  TestFetchFeatureArrBilinearInterpolationVec(feature_arr, t2_pxy.x, t2_pxy.y, f_t2);
  #pragma unroll
  for (int c = 0; c < kTotalChannels; ++c){
    *(raw_residual_vec+c) = 180.f * (f_t1[c] - f_pxy[c]) - surfel_descriptor_vec[c];
    *(raw_residual_vec+kTotalChannels+c) = 180.f * (f_t2[c] - f_pxy[c]) - surfel_descriptor_vec[kTotalChannels+c];
  }

  // fetching texture object (filtering mode: bilinear interpolation)
  /*float pxy_feature1 = tex2D<float4>(color_texture, pxy.x, pxy.y).x;
  float pxy_feature2 = tex2D<float4>(color_texture, pxy.x, pxy.y).y;
  float pxy_feature3 = tex2D<float4>(color_texture, pxy.x, pxy.y).z;
  printf("pxy_feature1 - f_t11: %f, %f\n", pxy_feature1, f_pxy[0]);
  printf("pxy_feature2 - f_t12: %f, %f\n", pxy_feature2 ,f_pxy[1]);
  printf("pxy_feature3 - f_t13: %f, %f\n", pxy_feature3 , f_pxy[2]);
 
  float t1_feature1 = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).x;
  float t1_feature2 = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).y;
  float t1_feature3 = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).z;
  printf("t1_feature1 - f_t1: %f, %f\n", t1_feature1 , f_t1[0]);
  printf("t1_feature2 - f_t2: %f, %f\n", t1_feature2 , f_t1[1]);
  printf("t1_feature3 - f_t3: %f, %f\n", t1_feature3 , f_t1[2]);

  float t2_feature1 = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).x;
  float t2_feature2 = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).y;
  float t2_feature3 = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).z;
  printf("t2_feature1 - f_t21: %f, %f\n", t2_feature1 , f_t2[0]);
  printf("t2_feature2 - f_t22: %f, %f\n", t2_feature2 , f_t2[1]);
  printf("t2_feature3 - f_t23: %f, %f\n", t2_feature3 , f_t2[2]);*/

  /*float raw_residual_vec_true[kTotalChannels*2] = {0};
  *(raw_residual_vec_true) = (180.f * (t1_feature1 - pxy_feature1)) - surfel_descriptor_vec[0];
  *(raw_residual_vec_true+1) = (180.f * (t1_feature2 - pxy_feature2)) - surfel_descriptor_vec[1];
  *(raw_residual_vec_true+2) = (180.f * (t1_feature3 - pxy_feature3)) - surfel_descriptor_vec[2];
  *(raw_residual_vec_true+3) = (180.f * (t2_feature1 - pxy_feature1)) - surfel_descriptor_vec[3];
  *(raw_residual_vec_true+4) = (180.f * (t2_feature2 - pxy_feature2)) - surfel_descriptor_vec[4];
  *(raw_residual_vec_true+5) = (180.f * (t2_feature3 - pxy_feature3)) - surfel_descriptor_vec[5];*/

  // TestCheckBilinearInterpolation(raw_residual_vec, raw_residual_vec_true);
}

__forceinline__ __device__ void TestComputeRawFeatureDescriptorResidualIntpixel(
    const CUDABuffer_<float>& feature_arr,
    const int2& pxy, /*pixel corner convention*/
    const int2& t1_pxy,/*pixel corner convention*/
    const int2& t2_pxy,
    float* surfel_descriptor_vec,
    float* raw_residual_vec) {

  // feature vectors
  float f_pxy[kTotalChannels] = {0}; // initialize all to 0, memory allocation
  float f_t1[kTotalChannels] = {0};
  float f_t2[kTotalChannels] = {0};
  // 2.24 Must handle out of range fetching by mannually clamping. Invalid texture memory fetching is handled by CUDA, that's why original code doesn't address that.
  TestFetchFeatureArrVec(feature_arr, pxy.x, pxy.y, f_pxy);
  TestFetchFeatureArrVec(feature_arr, t1_pxy.x, t1_pxy.y, f_t1);
  TestFetchFeatureArrVec(feature_arr, t2_pxy.x, t2_pxy.y, f_t2);
  #pragma unroll
  for (int c = 0; c < kTotalChannels; ++c){
    *(raw_residual_vec+c) = 180.f * (f_t1[c] - f_pxy[c]) - surfel_descriptor_vec[c];
    *(raw_residual_vec+kTotalChannels+c) = 180.f * (f_t2[c] - f_pxy[c]) - surfel_descriptor_vec[kTotalChannels+c];
  }
}

// 3.29 1point residual, indexing int pixel location
__forceinline__ __device__ void ComputeRawFeatureDescriptor1PointResidualIntpixel(
    const CUDABuffer_<float>& feature_arr,
    const int2& pxy, /*pixel corner convention*/ /* Validity must be handled already */
    float* surfel_descriptor_vec,
    float* raw_residual_vec) {
  // feature vectors
  float f_pxy[kTotalChannels] = {0}; // initialize all to 0, memory allocation
  // 2.24 Must handle out of range fetching by mannually clamping. Invalid texture memory fetching is handled by CUDA, that's why original code doesn't address that.
  TestFetchFeatureArrVec(feature_arr, pxy.x, pxy.y, f_pxy);
  #pragma unroll
  for (int c = 0; c < kTotalChannels; ++c){
    // *(raw_residual_vec+c) = 180.f * f_pxy[c] - surfel_descriptor_vec[c]; // 3.29 180.0 is emperically set, might get rid of it.
     *(raw_residual_vec+c) = f_pxy[c] - surfel_descriptor_vec[c]; // 4.7 remove 180.
  }
}

// 3.29 1point residual, indexing float pixel location using bilinear interpolation
__forceinline__ __device__ void ComputeRawFeatureDescriptor1PointResidualFloatpixel(
    const CUDABuffer_<float>& feature_arr,
    const float2& pxy, /*pixel corner convention*/
    float* surfel_descriptor_vec,
    float* raw_residual_vec) {
  // feature vectors
  float f_pxy[kTotalChannels] = {0}; // initialize all to 0, memory allocation
  TestFetchFeatureArrBilinearInterpolationVec(feature_arr, pxy.x, pxy.y, f_pxy);
  #pragma unroll
  for (int c = 0; c < kTotalChannels; ++c){
    // *(raw_residual_vec+c) = 180.f * f_pxy[c] - surfel_descriptor_vec[c];
    *(raw_residual_vec+c) = f_pxy[c] - surfel_descriptor_vec[c]; // 4.7 remove uncertainty
  }
}

__forceinline__ __device__ void ComputeRawFeatureDescriptorResidual(
    cudaTextureObject_t feature_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float* surfel_descriptor_vec,
    float* raw_residual_vec) {
  // float intensity = tex2D<float4>(color_texture, pxy.x, pxy.y).w;
  // unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  // float t1_intensity = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).w; 
  // float t2_intensity = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).w;
  // TODO: for feature maps. make feature[N]t1_feature[N], t2_feature[N];
  
  /*for (int i = 0; i < 1; ++i){ // here i < 1, the 1 is the number of channels, TODO: make a variable
    *(raw_residual+i) = (180.f * (t1_intensity - intensity)) - surfel_descriptor[i];
    *(raw_residual+1+i) = (180.f * (t2_intensity - intensity)) - surfel_descriptor[i];
  }
  *raw_residual = (180.f * (t1_intensity - intensity)) - surfel_descriptor[0];
  *(raw_residual+1) = (180.f * (t2_intensity - intensity)) - surfel_descriptor[1];
  if (threadIdx.x == 9  && blockIdx.x == 0) { // jzm: 23/10 are you sure the threadIdx.x which you wanna check is correct? acutally it's fine. Only want to print result of an arbitrary thread. 
    printf("intensity = %f\n", intensity);
  }*/

  // ----- the following is checked in my mind ... --- //
  // since the depth residual dominates the descriptor residual, very litte difference is observed directly using color image to compute residual, even when the color residual jacobian has not been modified yet.
  float pxy_feature1 = tex2D<float4>(feature_texture, pxy.x, pxy.y).x;
  float pxy_feature2 = tex2D<float4>(feature_texture, pxy.x, pxy.y).y;
  float pxy_feature3 = tex2D<float4>(feature_texture, pxy.x, pxy.y).z;

  float t1_feature1 = tex2D<float4>(feature_texture, t1_pxy.x, t1_pxy.y).x;
  float t1_feature2 = tex2D<float4>(feature_texture, t1_pxy.x, t1_pxy.y).y;
  float t1_feature3 = tex2D<float4>(feature_texture, t1_pxy.x, t1_pxy.y).z;

  float t2_feature1 = tex2D<float4>(feature_texture, t2_pxy.x, t2_pxy.y).x;
  float t2_feature2 = tex2D<float4>(feature_texture, t2_pxy.x, t2_pxy.y).y;
  float t2_feature3 = tex2D<float4>(feature_texture, t2_pxy.x, t2_pxy.y).z;

  *(raw_residual_vec) = (180.f * (t1_feature1 - pxy_feature1)) - surfel_descriptor_vec[0];
  *(raw_residual_vec+1) = (180.f * (t1_feature2 - pxy_feature2)) - surfel_descriptor_vec[1];
  *(raw_residual_vec+2) = (180.f * (t1_feature3 - pxy_feature3)) - surfel_descriptor_vec[2];
  *(raw_residual_vec+3) = (180.f * (t2_feature1 - pxy_feature1)) - surfel_descriptor_vec[3];
  *(raw_residual_vec+4) = (180.f * (t2_feature2 - pxy_feature2)) - surfel_descriptor_vec[4];
  *(raw_residual_vec+5) = (180.f * (t2_feature3 - pxy_feature3)) - surfel_descriptor_vec[5];
}
// 10.30, instead of computing intensity gradients, compute gradients on each color channel
// jzmTODO: use templates to get values in each feature map. Also depends on the structure of feature_texture
// Computes the Jacobian of a surfel descriptor with regard to changes in the
// projected pixel position of the surfel. This function makes the approximation that
// the projected positions of all points on the surfel move equally. This should
// be valid since those points should all be very close together.
/*__forceinline__ __device__ void DescriptorJacobianWrtProjectedPositionOnChannels(
    cudaTextureObject_t color_texture, // jzmTODO: make it feature_texture
    const float2& color_pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float* grad_x_1,
    float* grad_y_1,
    float* grad_x_2,
    float* grad_y_2,
    int channel) {
  // 11.16 Texture object fetching and filtering, +-0.5 stuff.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f)); // refer to libvis/camera.h: 103, convert to pixel center convention??? easier to compute the offsets from te pixel centers -> bilinear interpolation
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z) // frac(xB), xB = x-0.5
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left, top_right, bottom_left, bottom_right;
  if (channel == 0){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).x; // 11.12 +0.5 due to texture indexing of CUDA
     top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).x;
     bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).x;
     bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).x;
  }
  else if (channel == 1){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).y;
     top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).y;
     bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).y;
     bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).y;
  }
  else if (channel == 2){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).z;
     top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).z;
     bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).z;
     bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).z;
  }
  else if (channel == 3){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  }  
  
  float center_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float center_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  

  ix = static_cast<int>(::max(0.f, t1_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t1_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t1_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t1_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  if (channel == 0){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).x;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).x;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).x;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).x;
  }
  else if (channel == 1){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).y;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).y;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).y;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).y;
  }
  else if (channel == 2){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).z;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).z;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).z;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).z;
  }
  else if (channel == 3){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  }
  
  
  float t1_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t1_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  
  ix = static_cast<int>(::max(0.f, t2_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t2_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t2_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t2_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)

  if (channel == 0){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).x;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).x;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).x;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).x;
  }
  else if (channel == 1){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).y;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).y;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).y;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).y;
  }
  else if (channel == 2){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).z;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).z;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).z;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).z;
  }
  else if (channel == 3){
    top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  }
  
  float t2_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t2_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  // 11.2 not used?
  // float intensity = tex2D<float4>(color_texture, color_pxy.x, color_pxy.y).w;
  // float t1_intensity = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).w;
  // float t2_intensity = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).w;
  
  // NOTE: It is approximate to mix all the center, t1, t2 derivatives
  //       directly since the points would move slightly differently on most
  //       pose changes. However, the approximation is possibly pretty good since
  //       the points are all close to each other.
  
  *grad_x_1 = 180.f * (t1_dx - center_dx);
  *grad_y_1 = 180.f * (t1_dy - center_dy);
  *grad_x_2 = 180.f * (t2_dx - center_dx);
  *grad_y_2 = 180.f * (t2_dy - center_dy);
}*/

__forceinline__ __device__ void TestDescriptorJacobianWrtProjectedPositionOnChannels(
    const CUDABuffer_<float>& feature_arr, 
    const float2& color_pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float* grad_x_1,
    float* grad_y_1,
    float* grad_x_2,
    float* grad_y_2,
    int c) {
  /*unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (surfel_index == 0){
    printf("jacobian: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
  }*/
  // 11.16 Texture object fetching and filtering, +-0.5 stuff.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f)); // refer to libvis/camera.h: 103, convert to pixel center convention??? easier to compute the offsets from te pixel centers -> bilinear interpolation
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  // ix = ::min(ix, feature_arr.width()/kTotalChannels-1);
  ix = ::min(ix, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1);
  iy = ::min(iy, feature_arr.height()-1);
  // 11.19 out of bounds error => clamp it, texture memory handles like this!!! always check if out of bounds when indexing something
  // see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory

  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z) // frac(xB), xB = x-0.5
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left, top_right, bottom_left, bottom_right;

  top_left = feature_arr(iy, ix*kTotalChannels+c);
  // top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);
  bottom_left = feature_arr(::min(iy+1, feature_arr.height()-1), ix*kTotalChannels+c);
  // bottom_right = feature_arr(::min(iy+1, feature_arr.height()-1), ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  bottom_right = feature_arr(::min(iy+1, feature_arr.height()-1), ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);

  float center_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float center_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  

  ix = static_cast<int>(::max(0.f, t1_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t1_pxy.y - 0.5f));
  ix = ::min(ix, feature_arr.width()/(kTotalChannels+kFeatResidualChannel+kGeomResidualChannel)-1);
  iy = ::min(iy, feature_arr.height()-1);
  tx = ::max(0.f, ::min(1.f, t1_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t1_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)

  top_left = feature_arr(iy, ix*kTotalChannels+c);
  // top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);
  bottom_left = feature_arr(::min(iy+1, feature_arr.height()-1), ix*kTotalChannels+c);
  // bottom_right = feature_arr(::min(iy+1, feature_arr.height()-1), ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  bottom_right = feature_arr(::min(iy+1, feature_arr.height()-1), ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);

  
  float t1_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t1_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  
  ix = static_cast<int>(::max(0.f, t2_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t2_pxy.y - 0.5f));
  // ix = ::min(ix, feature_arr.width()/kTotalChannels-1);
  ix = ::min(ix, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1);
  iy = ::min(iy, feature_arr.height()-1);
  tx = ::max(0.f, ::min(1.f, t2_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t2_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)


  top_left = feature_arr(iy, ix*kTotalChannels+c);
  // top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);
  bottom_left = feature_arr(::min(iy+1, feature_arr.height()-1), ix*kTotalChannels+c);
  // bottom_right = feature_arr(::min(iy+1, feature_arr.height()-1), ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  bottom_right = feature_arr(::min(iy+1, feature_arr.height()-1), ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);

  float t2_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t2_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  // NOTE: It is approximate to mix all the center, t1, t2 derivatives
  //       directly since the points would move slightly differently on most
  //       pose changes. However, the approximation is possibly pretty good since
  //       the points are all close to each other.
  
  *grad_x_1 = 180.f * (t1_dx - center_dx); // 2.24 180. is emperically set. 
  *grad_y_1 = 180.f * (t1_dy - center_dy);
  *grad_x_2 = 180.f * (t2_dx - center_dx);
  *grad_y_2 = 180.f * (t2_dy - center_dy);
}

__forceinline__ __device__ void Descriptor1PointJacobianWrtProjectedPositionOnChannels(
    const CUDABuffer_<float>& feature_arr, 
    const float2& color_pxy,
    float* grad_x_1,
    float* grad_y_1,
    int c) {
  /*unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (surfel_index == 0){
    printf("jacobian: feat(400,2000)=%f, feat(457,2216)=%f \n",feature_arr(400,2000), feature_arr(457,2216));
  }*/
  // 11.16 Texture object fetching and filtering, +-0.5 stuff.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f)); // refer to libvis/camera.h: 103, convert to pixel center convention??? easier to compute the offsets from te pixel centers -> bilinear interpolation
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  // ix = ::min(ix, feature_arr.width()/kTotalChannels-1);
  ix = ::min(ix, feature_arr.width()/(kTotalChannels+kFeatResidualChannel+kGeomResidualChannel)-1);
  iy = ::min(iy, feature_arr.height()-1);
  // 11.19 out of bounds error => clamp it, texture memory handles like this!!! always check if out of bounds when indexing something
  // see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory

  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z) // frac(xB), xB = x-0.5
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left, top_right, bottom_left, bottom_right;

  top_left = feature_arr(iy, ix*kTotalChannels+c);
  // top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  top_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);
  bottom_left = feature_arr(::min(iy+1, feature_arr.height()-1), ix*kTotalChannels+c);
  // bottom_right = feature_arr(::min(iy+1, feature_arr.height()-1), ::min(ix+1, feature_arr.width()/kTotalChannels-1)*kTotalChannels+c);
  bottom_right = feature_arr(iy, ::min(ix+1, feature_arr.width()/(kTotalChannels+kGeomResidualChannel+kFeatResidualChannel)-1)*kTotalChannels+c);

  float center_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float center_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  

  // NOTE: It is approximate to mix all the center, t1, t2 derivatives
  //       directly since the points would move slightly differently on most
  //       pose changes. However, the approximation is possibly pretty good since
  //       the points are all close to each other.
  
  // *grad_x_1 = 180.f * center_dx; // 2.24 180. is emperically set. 
  // *grad_y_1 = 180.f * center_dy; 
  *grad_x_1 = center_dx; // 2.24 180. is emperically set. 4.7 remove it 
  *grad_y_1 = center_dy; 
}

}
