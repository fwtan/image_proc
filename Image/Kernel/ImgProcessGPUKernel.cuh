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
#ifndef _IMAGE_PROCESS_KERNEL_CUH_
#define _IMAGE_PROCESS_KERNEL_CUH_

inline __device__ unsigned int Float4ToUint(float4 rgba)
{
	rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return ((unsigned int)(rgba.w * 255.0f) << 24) | ((unsigned int)(rgba.z * 255.0f) << 16) | ((unsigned int)(rgba.y * 255.0f) << 8) | (unsigned int)(rgba.x * 255.0f);
}

/////////////////////////////////////////////////////////////////////////////////////
// resize image

__global__ void resizeImageKernelUchar1(int width, int height, int texWidth, int texHeight, float scale, unsigned char* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height)
	{
		return;
	}

	float tx = (float)(ix + 1) * (float)(texWidth) / (float)(width) - 1;
	float ty = (float)(iy + 1) * (float)(texHeight)/ (float)(height) - 1;
	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	float fo = (1-dx)*(1-dy)* float(tex2D(iproctexc1,tx,ty))   +
			   dx    *(1-dy)* float(tex2D(iproctexc1,tx+1,ty)) + 
			   (1-dx)*dy    * float(tex2D(iproctexc1,tx,ty+1)) + 
			   dx*dy        * float(tex2D(iproctexc1,tx+1,ty+1));

	fo = scale * fo;

	unsigned char* ptr = (unsigned char*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = clamp(fo,0.f,255.f);
}

__global__ void resizeImageKernelUchar2(int width, int height, int texWidth, int texHeight, float scale, uchar2* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height)
	{
		return;
	}

	float tx = (float)(ix + 1) * (float)(texWidth) / (float)(width) - 1;
	float ty = (float)(iy + 1) * (float)(texHeight)/ (float)(height) - 1;
	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	uchar2 c  = tex2D(iproctexc2,tx,  ty);
	uchar2 r  = tex2D(iproctexc2,tx+1,ty);
	uchar2 b  = tex2D(iproctexc2,tx,  ty+1);
	uchar2 rb = tex2D(iproctexc2,tx+1,ty+1);

	float fox = (1-dx)*(1-dy)*c.x + dx*(1-dy)*r.x + (1-dx)*dy*b.x + dx*dy*rb.x;
	float foy = (1-dx)*(1-dy)*c.y + dx*(1-dy)*r.y + (1-dx)*dy*b.y + dx*dy*rb.y;

	fox *= scale; foy *= scale;
	uchar2 co;
	co.x = clamp(fox,0.f,255.f); co.y = clamp(foy,0.f,255.f);

	uchar2* ptr = (uchar2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = co;
}

__global__ void resizeImageKernelUchar4(int width, int height, int texWidth, int texHeight, float scale, uchar4* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height)
	{
		return;
	}

	float tx = (float)(ix + 1) * (float)(texWidth) / (float)(width) - 1;
	float ty = (float)(iy + 1) * (float)(texHeight)/ (float)(height) - 1;
	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	uchar4 c  = tex2D(iproctexc4,tx,  ty);
	uchar4 r  = tex2D(iproctexc4,tx+1,ty);
	uchar4 b  = tex2D(iproctexc4,tx,  ty+1);
	uchar4 rb = tex2D(iproctexc4,tx+1,ty+1);

	float fox = (1-dx)*(1-dy)*c.x + dx*(1-dy)*r.x + (1-dx)*dy*b.x + dx*dy*rb.x;
	float foy = (1-dx)*(1-dy)*c.y + dx*(1-dy)*r.y + (1-dx)*dy*b.y + dx*dy*rb.y;
	float foz = (1-dx)*(1-dy)*c.z + dx*(1-dy)*r.z + (1-dx)*dy*b.z + dx*dy*rb.z;
	float fow = (1-dx)*(1-dy)*c.w + dx*(1-dy)*r.w + (1-dx)*dy*b.w + dx*dy*rb.w;

	fox *= scale; foy *= scale; foz *= scale; fow *= scale;
	uchar4 co;
	co.x = clamp(fox,0.f,255.f); co.y = clamp(foy,0.f,255.f);
	co.z = clamp(foz,0.f,255.f); co.w = clamp(fow,0.f,255.f);

	uchar4* ptr = (uchar4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = co;
}

__global__ void resizeImageKernelFloat1(int width, int height, int texWidth, int texHeight, float scale, float* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height)
	{
		return;
	}

	float tx = (float)(ix + 1) * (float)(texWidth) / (float)(width) - 1;
	float ty = (float)(iy + 1) * (float)(texHeight)/ (float)(height) - 1;
	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	
    /*// for image
	float fo = (1-dx)*(1-dy)*tex2D(iproctexf1,tx,ty)   +
			   dx    *(1-dy)*tex2D(iproctexf1,tx+1,ty) + 
			   (1-dx)*dy    *tex2D(iproctexf1,tx,ty+1) + 
			   dx*dy        *tex2D(iproctexf1,tx+1,ty+1);*/

	// for depth
	float fo = tex2D(iproctexf1,tx,ty);

	fo *= scale;

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = fo;
}

__global__ void resizeImageKernelFloat2(int width, int height, int texWidth, int texHeight, float scale, float2* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height)
	{
		return;
	}

	float tx = (float)(ix + 1) * (float)(texWidth) / (float)(width) - 1;
	float ty = (float)(iy + 1) * (float)(texHeight)/ (float)(height) - 1;
	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	/* // for image
	float2 fo = (1-dx)*(1-dy)*tex2D(iproctexf2,tx,ty)   +
		        dx    *(1-dy)*tex2D(iproctexf2,tx+1,ty) + 
		        (1-dx)*dy    *tex2D(iproctexf2,tx,ty+1) + 
		        dx*dy        *tex2D(iproctexf2,tx+1,ty+1);*/
	
	float2 fo = tex2D(iproctexf2,tx,ty);

	fo *= scale;

	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = fo;
}

__global__ void resizeImageKernelFloat4(int width, int height, int texWidth, int texHeight, float scale, float4* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height)
	{
		return;
	}

	float tx = (float)(ix + 1) * (float)(texWidth) / (float)(width) - 1;
	float ty = (float)(iy + 1) * (float)(texHeight)/ (float)(height) - 1;
	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	/* // for image
	float4 fo = (1-dx)*(1-dy)*tex2D(iproctexf4,tx,ty)   +
				dx    *(1-dy)*tex2D(iproctexf4,tx+1,ty) + 
				(1-dx)*dy    *tex2D(iproctexf4,tx,ty+1) + 
				dx*dy        *tex2D(iproctexf4,tx+1,ty+1);*/

	
	float4 fo = tex2D(iproctexf4,tx,ty);

	fo *= scale;

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = fo;
}



__global__ void bilateralFilterKernelFloat1(int width, int height, int radius, float sigma_i, float sigma_s, float* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float s = 0;
	float w = 0;

	float c = tex2D(iproctexf1,ix,iy);

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float n = tex2D(iproctexf1,ix+nx,iy+ny);
			float di = (n - c)*(n - c);
			di = expf(-0.5f*di/sigma_i);
			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

            float f = di * ds;
            w += f;
            s += f * n;
		}
	}

	c = s/w;
//	c = __saturatef(fabs(c));

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = c;
}

__global__ void bilateralFilterKernelFloat2(int width, int height, int radius, float sigma_i, float sigma_s, float2* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 s = {0,0};
    float  w = 0;

	float2 c = tex2D(iproctexf2,ix,iy);

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float2 n = tex2D(iproctexf2,ix+nx,iy+ny);
			float di = dist2f2(c,n);
			di = expf(-0.5f*di/sigma_i);
			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

            float f = di * ds;
            w += f;
            s += f * n;
		}
	}

	c = s/w;
//	c.x = __saturatef(fabs(c.x));
//	c.y = __saturatef(fabs(c.y));


	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = c;
}

__global__ void bilateralFilterKernelFloat4(int width, int height, int radius, float sigma_i, float sigma_s, float4* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float4 s = {0,0,0,0};
    float  w = 0;

	float4 c = tex2D(iproctexf4,ix,iy);

	// temporary modification for broken normal map
	if (c.x == 0 && c.y == 0 && c.z == 0)
	{
		return;
	}

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float4 n = tex2D(iproctexf4,ix+nx,iy+ny);
			// temporary modification for broken normal map
			if (n.x == 0 && n.y == 0 && n.z == 0)	continue;
			float di = dist2f4(c,n);
			di = expf(-0.5f*di/sigma_i);
			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			w += di * ds;
			s += di * ds * n;
		}
	}

	c = s/w;
	/*c.x = __saturatef(fabs(c.x));
	c.y = __saturatef(fabs(c.y));
	c.z = __saturatef(fabs(c.z));
	c.w = __saturatef(fabs(c.w));*/

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = c;
}

__global__ void bilateralFilterKernelUchar1(int width, int height, int radius, float sigma_i, float sigma_s, unsigned char* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float s = 0;
	float w = 0;

	float c = tex2D(iproctexc1n,ix,iy);

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float n = tex2D(iproctexc1n,ix+nx,iy+ny);
			float di = (n - c)*(n - c);
			di = expf(-0.5f*di/sigma_i);
			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			w += di * ds;
			s += di * ds * n;
		}
	}

	c = s/w;
	c = __saturatef(fabs(c));
	unsigned char o = c * 255;

	unsigned char* ptr = (unsigned char*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = o;
}

__global__ void bilateralFilterKernelUchar2(int width, int height, int radius, float sigma_i, float sigma_s, uchar2* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 s = {0,0};
	float w = 0;

	float2 c = tex2D(iproctexc2n,ix,iy);

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float2 n = tex2D(iproctexc2n,ix+nx,iy+ny);
			float di = dist2f2(c,n);
			di = expf(-0.5f*di/sigma_i);
			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			w += di * ds;
			s += di * ds * n;
		}
	}

	s = s/w * 255.f;
	uchar2  co; co.x = clamp(s.x,0.f,255.f); co.y = clamp(s.y,0.f,255.f);
	uchar2* ptr = (uchar2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = co;
}

__global__ void bilateralFilterKernelUchar4(int width, int height, int radius, float sigma_i, float sigma_s, uchar4* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float4 s = {0,0,0,0};
	float w = 0;

	float4 c = tex2D(iproctexc4n,ix,iy);

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float4 n = tex2D(iproctexc4n,ix+nx,iy+ny);
			float di = dist2f4(c,n);
			di = expf(-0.5f*di/sigma_i);
			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			w += di * ds;
			s += di * ds * n;
		}
	}

	uint* ptr = (uint*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = Float4ToUint(s/w);
}

__global__ void copyFromFloatToUcharKernel(int width, int height, int channel, 
	const float*   dIn,  size_t dInPitch,
	unsigned char* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	const float*	ptri = (const float*)((char*)dIn + iy * dInPitch) + ix * channel;
	unsigned char*	ptro = (unsigned char*)((char*)dOut + iy * dOutPitch) + ix * channel;

	for (int i = 0; i < channel; ++i)
	{
		ptro[i] = 255 * __saturatef(fabs(ptri[i]));
	}
}

__global__ void copyFromUcharToFloatKernel(int width, int height, int channel, 
	const unsigned char*  dIn,  size_t dInPitch,
	float*		  dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	const unsigned char*	ptri = (const unsigned char*)((char*)dIn + iy * dInPitch) + ix * channel;
	float*			ptro = (float*)((char*)dOut + iy * dOutPitch) + ix * channel;

	for (int i = 0; i < channel; ++i)
	{
		ptro[i] = float(ptri[i])/255.f;
	}
}

template <class T1,class T2>
__global__ void cropImageKernel(int width, int height, int channel, int left, int top,
	const T1* dIn,  size_t dInPitch,
	T2*		  dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	int tx = ix + left;
	int ty = iy + top;

	const T1* ptri = (const T1*)((char*)dIn + ty * dInPitch) + tx * channel;
	T2*	      ptro = (T2*)((char*)dOut + iy * dOutPitch) + ix * channel;

	for (int i = 0; i < channel; ++i)
	{
		ptro[i] = ptri[i];
	}
}

__global__ void hFilteringKernelFloat1(int width, int height, float* dFilter1D, int radius, float* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter1D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y == 0)
	{
		sFilter1D[threadIdx.x] = dFilter1D[threadIdx.x];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float s = 0;
//	float w = 0;
	for (int i = -radius; i <= radius; ++i)
	{
		s += sFilter1D[i+radius] * tex2D(iproctexf1,ix+i,iy);
//		w += sFilter1D[i+radius];
	}

	// if (w > 0)
	// {
	// 	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void hFilteringKernelFloat2(int width, int height, float* dFilter1D, int radius, float2* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter1D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y == 0)
	{
		sFilter1D[threadIdx.x] = dFilter1D[threadIdx.x];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 s = {0,0};
//	float  w = 0;
	for (int i = -radius; i <= radius; ++i)
	{
		s += sFilter1D[i+radius] * tex2D(iproctexf2,ix+i,iy);
//		w += sFilter1D[i+radius];
	}

	// if (w > 0)
	// {
	// 	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void hFilteringKernelFloat4(int width, int height, float* dFilter1D, int radius, float4* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter1D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y == 0)
	{
		sFilter1D[threadIdx.x] = dFilter1D[threadIdx.x];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float4 s = {0,0,0,0};
//	float  w = 0;
	for (int i = -radius; i <= radius; ++i)
	{
		s += sFilter1D[i+radius] * tex2D(iproctexf4,ix+i,iy);
//		w += sFilter1D[i+radius];
	}

	// if (w > 0)
	// {
	// 	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void vFilteringKernelFloat1(int width, int height, float* dFilter1D, int radius, float* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter1D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y == 0)
	{
		sFilter1D[threadIdx.x] = dFilter1D[threadIdx.x];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float s = 0;
//	float w = 0;
	for (int i = -radius; i <= radius; ++i)
	{
		s += sFilter1D[i+radius] * tex2D(iproctexf1,ix,iy+i);
//		w += sFilter1D[i+radius];
	}

	// if (w > 0)
	// {
	// 	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void vFilteringKernelFloat2(int width, int height, float* dFilter1D, int radius, float2* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter1D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y == 0)
	{
		sFilter1D[threadIdx.x] = dFilter1D[threadIdx.x];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 s = {0,0};
//	float  w = 0;
	for (int i = -radius; i <= radius; ++i)
	{
		s += sFilter1D[i+radius] * tex2D(iproctexf2,ix,iy+i);
//		w += sFilter1D[i+radius];
	}

	// if (w > 0)
	// {
	// 	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void vFilteringKernelFloat4(int width, int height, float* dFilter1D, int radius, float4* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter1D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y == 0)
	{
		sFilter1D[threadIdx.x] = dFilter1D[threadIdx.x];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float4 s = {0,0,0,0};
//	float  w = 0;
	for (int i = -radius; i <= radius; ++i)
	{
		s += sFilter1D[i+radius] * tex2D(iproctexf4,ix,iy+i);
//		w += sFilter1D[i+radius];
	}

	// if (w > 0)
	// {
	// 	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void filteringKernelFloat1(int width, int height, float* dFilter2D, int radius, float* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter2D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y < length)
	{
		int fid = threadIdx.x + threadIdx.y * length;
		sFilter2D[fid] = dFilter2D[fid];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float s = 0;
//	float w = 0;
	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			int fid = nx + radius + (ny + radius)*length;
			float ni = tex2D(iproctexf1,ix+nx,iy+ny);
			// temporary modification for broken depth map
//			if (ni < 0.5f || ni > 1.f) continue;
			s += sFilter2D[fid] * ni;
//			w += sFilter2D[fid];
		}
		
	}

	// if (w > 0)
	// {
	// 	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void filteringKernelFloat2(int width, int height, float* dFilter2D, int radius, float2* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter2D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y < length)
	{
		int fid = threadIdx.x + threadIdx.y * length;
		sFilter2D[fid] = dFilter2D[fid];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 s = {0,0};
//	float  w = 0;
	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			int fid = nx + radius + (ny + radius)*length;
			s += sFilter2D[fid] * tex2D(iproctexf2,ix+nx,iy+ny);
//			w += sFilter2D[fid];
		}
	}

	// if (w > 0)
	// {
	// 	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void filteringKernelFloat4(int width, int height, float* dFilter2D, int radius, float4* dOut, size_t dOutPitch)
{
	extern __shared__ float sFilter2D[];

	int length = 2 * radius + 1;
	if (threadIdx.x < length && threadIdx.y < length)
	{
		int fid = threadIdx.x + threadIdx.y * length;
		sFilter2D[fid] = dFilter2D[fid];
	}
	__syncthreads();

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float4 s = {0,0,0,0};
//	float  w = 0;
	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			int fid = nx + radius + (ny + radius)*length;
			float4 nn = tex2D(iproctexf4,ix+nx,iy+ny);
			// temporary modification for broken normal map
//			if (nn.x == 0 && nn.y == 0 && nn.z == 0) continue;
			s += sFilter2D[fid] * nn;
//			w += sFilter2D[fid];
		}
	}

	// if (w > 0)
	// {
	// 	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	// 	(*ptr) = s/w;
	// }

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = s;
}

__global__ void warpImageKernelFloat1(int width, int height, float* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 uv = tex2D(iprocauxf2,ix,iy);
	float tx = ix + uv.x;
	float ty = iy + uv.y;

	if (tx >= width || ty >= height)
	{
		return;
	}

	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	float fo = (1-dx)*(1-dy)*tex2D(iproctexf1,tx,ty)   +
		       dx    *(1-dy)*tex2D(iproctexf1,tx+1,ty) + 
		       (1-dx)*dy    *tex2D(iproctexf1,tx,ty+1) + 
		       dx*dy        *tex2D(iproctexf1,tx+1,ty+1);

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = fo;
}

__global__ void warpImageKernelFloat2(int width, int height, float2* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 uv = tex2D(iprocauxf2,ix,iy);
	float tx = ix + uv.x;
	float ty = iy + uv.y;

	if (tx >= width || ty >= height)
	{
		return;
	}

	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	float2 fo = (1-dx)*(1-dy)*tex2D(iproctexf2,tx,ty)   +
		dx    *(1-dy)*tex2D(iproctexf2,tx+1,ty) + 
		(1-dx)*dy    *tex2D(iproctexf2,tx,ty+1) + 
		dx*dy        *tex2D(iproctexf2,tx+1,ty+1);

	float2* ptr = (float2*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = fo;
}

__global__ void warpImageKernelFloat4(int width, int height, float4* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float2 uv = tex2D(iprocauxf2,ix,iy);
	float tx = ix + uv.x;
	float ty = iy + uv.y;

	if (tx >= width || ty >= height)
	{
		return;
	}

	float dx = tx - int(tx); dx = fmaxf(fminf(dx,1.f),0.f);
	float dy = ty - int(ty); dy = fmaxf(fminf(dy,1.f),0.f);

	float4 fo = (1-dx)*(1-dy)*tex2D(iproctexf4,tx,ty)   +
		dx    *(1-dy)*tex2D(iproctexf4,tx+1,ty) + 
		(1-dx)*dy    *tex2D(iproctexf4,tx,ty+1) + 
		dx*dy        *tex2D(iproctexf4,tx+1,ty+1);

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = fo;
}

template <typename FT>
__global__ void vFlipImageKernel(int width, int height, int channel, const FT* dIn, size_t dInPitch, FT* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	const FT* ptri = (const FT*)((char*)dIn + (height - 1 - iy) * dInPitch) + ix * channel;
	FT*	      ptro = (FT*)((char*)dOut + iy * dOutPitch) + ix * channel;

	for (int i = 0; i < channel; ++i)
	{
		ptro[i] = ptri[i];
	}
}

template <typename FT>
__global__ void hFlipImageKernel(int width, int height, int channel, const FT* dIn, size_t dInPitch, FT* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	const FT* ptri = (const FT*)((char*)dIn + iy * dInPitch) + (width - 1 - ix) * channel;
	FT*	      ptro = (FT*)((char*)dOut + iy * dOutPitch) + ix * channel;

	for (int i = 0; i < channel; ++i)
	{
		ptro[i] = ptri[i];
	}
}

//////////////////////////////////////////////////////////////
// deprecated
__global__ void jointBilateralFilterKernelFloat1(int width, int height, int radius, float sigma_i, float sigma_s, float sigma_r, float* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float fx = ((float)ix + 0.5f) / (float)width;
	float fy = ((float)iy + 0.5f) / (float)height;

	float  s = 0;
	float  w = 0;
	float  c = tex2D(iproctexf1,ix,iy);
	float4 r = tex2D(iprocauxf4,fx,fy);
	

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float ni = tex2D(iproctexf1,ix+nx,iy+ny);
			float di = (ni - c) * (ni - c);
			di = expf(-0.5f*di/sigma_i);

			float4 nr = tex2D(iprocauxf4,(ix + nx + 0.5f) / (float)width,(iy + ny + 0.5f) / (float)height);
			float dr = dist2f4(r,nr);
			dr = expf(-0.5f*dr/sigma_r);

			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			// temporary modification for snakes
//			float f = di * ds * dr;
			float f = ds * dr;
			w += f;
			s += f * ni;
		}
	}

	c = s/w;
	// // temporary modification for snakes
	// if (c < -0.2) c = -1;
	// else if (c < 0.2) c = 0;
	// else c = 1;

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = c;
}

__global__ void jointBilateralFilterKernelFloat4(
	int width, 
	int height, 
	int radius, 
	float sigma_i, 
	float sigma_s, 
	float sigma_r, 
	float4* dOut, 
	size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float fx = ((float)ix + 0.5f) / (float)width;
	float fy = ((float)iy + 0.5f) / (float)height;

	float4 s = {0,0,0,0};
	float  w = 0;

	float4 r = tex2D(iprocauxf4,fx,fy);
	float4 c = tex2D(iproctexf4,ix,iy);

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float4 ni = tex2D(iproctexf4,ix+nx,iy+ny);
			float di = dist2f4(c,ni);
			di = expf(-0.5f*di/sigma_i);

			float4 nr = tex2D(iprocauxf4,(ix + nx + 0.5f) / (float)width,(iy + ny + 0.5f) / (float)height);
			float dr = dist2f4(r,nr);
			dr = expf(-0.5f*dr/sigma_r);

			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			//float f = di * ds * dr;
			float f = ds * dr;
			w += f;
			s += f * ni;
		}
	}

	c = s/w;

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = c;
}

__global__ void jointBilateralFilterKernelUchar1(int width, int height, int radius, float sigma_i, float sigma_s, float sigma_r, float* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float fx = ((float)ix + 0.5f) / (float)width;
	float fy = ((float)iy + 0.5f) / (float)height;

	float  s = 0;
	float  w = 0;
	float  c = tex2D(iproctexf1,ix,iy);
	float4 r = tex2D(iprocauxc4n,fx,fy);


	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float ni = tex2D(iproctexf1,ix+nx,iy+ny);
			float di = (ni - c) * (ni - c);
			di = expf(-0.5f*di/sigma_i);

			float4 nr = tex2D(iprocauxc4n,(ix + nx + 0.5f) / (float)width,(iy + ny + 0.5f) / (float)height);
			float dr = dist2f4(r,nr);
			dr = expf(-0.5f*dr/sigma_r);

			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			float f = di * ds * dr;
			w += f;
			s += f * ni;
		}
	}

	c = s/w;

	float* ptr = (float*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = c;
}

__global__ void jointBilateralFilterKernelUchar4(int width, int height, int radius, float sigma_i, float sigma_s, float sigma_r, float4* dOut, size_t dOutPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float fx = ((float)ix + 0.5f) / (float)width;
	float fy = ((float)iy + 0.5f) / (float)height;

	float4 s = {0,0,0,0};
	float  w = 0;

	float4 r = tex2D(iprocauxf4,fx,fy);
	float4 c = tex2D(iprocauxc4n,ix,iy);

	for (int ny = -radius; ny <= radius; ++ny)
	{
		for (int nx = -radius; nx <= radius; ++nx)
		{
			float4 ni = tex2D(iproctexf4,ix+nx,iy+ny);
			float di = dist2f4(c,ni);
			di = expf(-0.5f*di/sigma_i);

			float4 nr = tex2D(iprocauxc4n,(ix + nx + 0.5f) / (float)width,(iy + ny + 0.5f) / (float)height);
			float dr = dist2f4(r,nr);
			dr = expf(-0.5f*dr/sigma_r);

			float ds = nx * nx + ny * ny;
			ds = expf(-0.5f*ds/sigma_s);

			float f = di * ds * dr;
			w += f;
			s += f * ni;
		}
	}

	c = s/w;

	float4* ptr = (float4*)((char*)dOut + iy * dOutPitch) + ix;
	(*ptr) = c;
}

#endif//_IMAGE_PROCESS_KERNEL_CUH_
