///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2014, Fuwen TAN, all rights reserved.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions (licence) and the following disclaimer
//       in the documentation and/or other materials provided with the
//       distribution.
//     * The name of the author may not be used to endorse or promote products
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to
//       and keep in place any licencing terms of those libraries.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////
#ifndef _HELPER_MATH_EXT_H_
#define _HELPER_MATH_EXT_H_

#include "cuda_runtime.h"
#include <helper_math.h>

inline __host__ __device__ float dist2f2(float2 p1, float2 p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

inline __host__ __device__ float dist2f4(float4 p1, float4 p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + 
		   (p1.y - p2.y) * (p1.y - p2.y) + 
		   (p1.z - p2.z) * (p1.z - p2.z);
}

inline __host__ __device__ float4 cross_product(float4 ax, float4 ay)
{
	float4 az;
	az.x = ax.y * ay.z - ax.z * ay.y;
	az.y = ax.z * ay.x - ax.x * ay.z;
	az.z = ax.x * ay.y - ax.y * ay.x;
	az.w = 0;

	return az;
}

inline __host__ __device__ float dot_product(float4 ax, float4 ay)
{
	return ax.x * ay.x + ax.y * ay.y + ax.z * ay.z;
}

#endif//_HELPER_MATH_EXT_H_