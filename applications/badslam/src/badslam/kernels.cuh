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

#pragma once // 11.20 include guard

#include <cuda_runtime.h>
#include <libvis/libvis.h>

namespace vis {

// This bit is set in the depth value of pixels which should not be used (since
// for example their normal direction is unknown).
constexpr u16 kInvalidDepthBit = 1 << 15;

// This value is used as depth value for pixels with unknown depth.
constexpr u16 kUnknownDepth = 65535;  // highest 16-bit unsigned value

// This flag is set for active surfels.
constexpr u8 kSurfelActiveFlag = 1 << 0;

// The number of buffers to use for surfel merging. In effect, this is the
// number of different surfels that can be stored for one pixel. If there are
// more individual surfels in one pixel that should not be merged than this
// value, then we might overlook cases where surfels should be merged. However,
// the more buffers are used, the slower it gets.
constexpr int kMergeBufferCount = 3;

// Invalid index value used in the "supporting surfels" buffers.
constexpr u32 kInvalidIndex = 4294967295;  // = numeric_limits<u32>::max();

// Threshold on angle difference between surfels to consider them compatible.
// constexpr float normal_compatibility_threshold_deg = 40;  // TODO: make parameter?
constexpr float cos_normal_compatibility_threshold = 0.76604f;  // = cosf(M_PI / 180.f * normal_compatibility_threshold_deg);


// Scalar type used in the PCG-based Gauss-Newton optimization (float or double).
typedef float PCGScalar;

// 11.20 number of channels, cost_function.cuh also needs this const, #include this file?
constexpr int kTotalChannels = 128;


// The surfel structure is stored in large buffers. It is organized
// such that each row stores one attribute and each column stores the
// attribute values for one surfel.

constexpr int kSurfelX = 0;  // float
constexpr int kSurfelY = 1;  // float
constexpr int kSurfelZ = 2;  // float
constexpr int kSurfelNormal = 3;  // (2 bits unused, s10 z, s10 y, s10 x)
constexpr int kSurfelRadiusSquared = 4;  // float
constexpr int kSurfelColor = 5;  // (u8 r, u8 g, u8 b, 8 bits unused)
constexpr int kSurfelDescriptor1 = 6;  // float
constexpr int kSurfelDescriptor2 = 7;  // float
// jzm, 10.29: for channel size of 3, need 6 descriptors in total, append 4 more.
// constexpr int kSurfelDescriptor3 = 8;
// constexpr int kSurfelDescriptor4 = 9;
// constexpr int kSurfelDescriptor5 = 10;
// constexpr int kSurfelDescriptor6 = 11;

// jzm. 10.29: +4 for each old value, for 4 more descirptors added
// 10.30 some steps only uses few kSurfelAccum
// kernel_assign_colors: kSurfelAccum0-2N [observation count, descriptors]
// kernel_opt_geometry: kSurfelAccum0-(2N+1)*(N+1)+2N [H, b ]
/*constexpr int kSurfelAccum0 = 8+4;  // float
constexpr int kSurfelAccum1 = 9+4;  // float
constexpr int kSurfelAccum2 = 10+4;  // float
constexpr int kSurfelAccum3 = 11+4;  // float
constexpr int kSurfelAccum4 = 12+4;  // float
constexpr int kSurfelAccum5 = 13+4;  // float
constexpr int kSurfelAccum6 = 14+4;  // float
constexpr int kSurfelAccum7 = 15+4;  // float
constexpr int kSurfelAccum8 = 16+4;  // float*/


// 10.29, construct kSurfelDescriptorArr
// constexpr kSurfelAttributesArr kSurfelDescriptor = kSurfelAttributesArr<2,6>();
// constexpr kSurfelAttributesArr kSurfelAccum = kSurfelAttributesArr<9,8>();

// 10.29, use an int array to store the index of kSurfelAccum



// This first number of attributes will be copied if a surfel is copied to a
// different index.
// constexpr int kSurfelDataAttributeCount = 8;
// constexpr int kSurfelDataAttributeCount = 12; // 6+2N, 6 is for kSurfelX/Y/Z, kSurfelNormal, kSurfelRadiusSquared, kSurfelColor, N channels => 2N descriptor residuals
// Total surfel attribute count, including temporary attributes (i.e. H and b) (which are not
// preserved during copies).
// constexpr int kSurfelAttributeCount = 17;
// 11.1 we only need to keep non-zero entries of H:4N+1 and b: 2N+1, in total:6N+2
/*
H is symmetric. split it into 3 parts: NOTICE: ensure that A and D are invertable. 
[A, C^T, diag(D)], A: top left part of H, C^T: top right part of H, D: bottom right part of H.
*/
// constexpr int kSurfelAttributeCount = 32; 

/*---------Explanation for k's----------*/
/*
1. fixed attributes: 6, [0,1,2,3,4,5]
2. descriptors: 2N, [6, ..., 2N+5]
3. Hessian: (2N+1)+2N, [2N+6, ..., 6N+6]
4. b: 2N+1 [6N+7, 8N+7]
5. data attribute count (x,y,z,normal,r^2,color,descriptor): 6+2N
6. accumulate H and b count: 6N+2
7. in totol: 8N+8
*/
// 11.1 
/*constexpr int kSurfelAccumHAndBCount = 20; // kSurfelAttributeCount - kSurfelDataAttributeCount
constexpr int kSurfelAccumHCount = 13; // 4N+1
constexpr int kSurfelAccumBCount = 7; // 2N+1
constexpr int kSurfelFixedAttributeCount = 6; */
// 11.17 add feature channels
// constexpr int kNumChannels = 3; // N, already defined in cost_function.cuh
// 11.20 parametrize things
constexpr int kSurfelAccumHAndBCount = 6*kTotalChannels+2; // 6N+2
constexpr int kSurfelAccumHCount = 4*kTotalChannels+1; // 4N+1
constexpr int kSurfelAccumBCount = 2*kTotalChannels+1; // 2N+1 
constexpr int kSurfelDataAttributeCount = 6+2*kTotalChannels; // 6+2N, 6 is for kSurfelX/Y/Z, kSurfelNormal, kSurfelRadiusSquared, kSurfelColor, N channels => 2N descriptor residuals
constexpr int kSurfelAttributeCount = 8*kTotalChannels+8; 
constexpr int kSurfelFixedAttributeCount = 6;
constexpr int kSurfelNumDescriptor = 2*kTotalChannels; // 2N

constexpr int kSurfelAccum0 = kSurfelFixedAttributeCount + kSurfelNumDescriptor;// 8+4;  // float
constexpr int kSurfelAccum1 = kSurfelAccum0 + 1;  // 13 float
constexpr int kSurfelAccum2 = kSurfelAccum0 + 2;  // 14 float
constexpr int kSurfelAccum3 = kSurfelAccum0 + 3;  // 15 float
constexpr int kSurfelAccum4 = kSurfelAccum0 + 4;  // 16 float
constexpr int kSurfelAccum5 = kSurfelAccum0 + 5;  // 17 float
constexpr int kSurfelAccum6 = kSurfelAccum0 + 6;  // 18 float
constexpr int kSurfelAccum7 = kSurfelAccum0 + 7;  // 19 float
constexpr int kSurfelAccum8 = kSurfelAccum0 + 8;  // 20 float

// 12.1 paprameters related to feature shape
constexpr int kFeatureW = 739;  
constexpr int kFeatureH = 458;  
}
