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

#include "ImgProcessGPU.h"
#include <helper_math.h>
#include <rendercheck_gl.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda_gl.h>

texture<float, 2,cudaReadModeElementType>               iproctexf1;
texture<float2,2,cudaReadModeElementType>               iproctexf2;
texture<float4,2,cudaReadModeElementType>               iproctexf4;
texture<unsigned char, 2,cudaReadModeElementType>       iproctexc1;
texture<uchar2,2,cudaReadModeElementType>               iproctexc2;
texture<uchar4,2,cudaReadModeElementType>               iproctexc4;
texture<unsigned char, 2, cudaReadModeNormalizedFloat>  iproctexc1n;
texture<uchar2,2,cudaReadModeNormalizedFloat>           iproctexc2n;
texture<uchar4,2,cudaReadModeNormalizedFloat>           iproctexc4n;
texture<float2,2,cudaReadModeElementType>               iprocauxf2;
texture<float4,2,cudaReadModeElementType>               iprocauxf4;
texture<uchar4,2,cudaReadModeNormalizedFloat>           iprocauxc4n;

#include "Kernel/ImgProcessGPUKernel.cuh"

void ImgProcessGPU::resizeImageUchar(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg, float ratio, float scale)
{
    assert(srcimg.data());
	if (!srcimg.data())
	{
		return;
	}

	int sw = srcimg.width();
	int sh = srcimg.height();

	int tw = sw * ratio + 0.5;
	int th = sh * ratio + 0.5;

	resizeImageUchar(srcimg,dstimg,tw,th,scale);
}

void ImgProcessGPU::resizeImageUchar(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg, int tw, int th, float scale)
{
    assert(srcimg.data());
	if (!srcimg.data())
	{
		return;
	}

	int sw = srcimg.width();
	int sh = srcimg.height();
	int c  = srcimg.channel();

	if (tw != dstimg.width() || th != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(tw,th,c);
	}

	dim3 gridSize((tw + 16 - 1) / 16, (th + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<unsigned char>();
		iproctexc1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexc1,srcimg.data(),desc,sw,sh,srcimg.pitch()));
		resizeImageKernelUchar1<<<gridSize, blockSize>>>(tw,th,sw,sh,scale,(unsigned char*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 2:
		desc = cudaCreateChannelDesc<uchar2>();
		iproctexc2.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexc2,srcimg.data(),desc,sw,sh,srcimg.pitch()));
		resizeImageKernelUchar2<<<gridSize, blockSize>>>(tw,th,sw,sh,scale,(uchar2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 4:
		desc = cudaCreateChannelDesc<uchar4>();
		iproctexc4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexc4,srcimg.data(),desc,sw,sh,srcimg.pitch()));
		resizeImageKernelUchar4<<<gridSize, blockSize>>>(tw,th,sw,sh,scale,(uchar4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}

void ImgProcessGPU::resizeImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float ratio, float scale)
{
    assert(srcimg.data());
	if (!srcimg.data())
	{
		return;
	}

	int sw = srcimg.width();
	int sh = srcimg.height();

	int tw = sw * ratio + 0.5;
	int th = sh * ratio + 0.5;

	resizeImageFloat(srcimg,dstimg,tw,th,scale);
}

void ImgProcessGPU::resizeImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int tw, int th, float scale)
{
    assert(srcimg.data());
	if (!srcimg.data())
	{
		return;
	}

	int sw = srcimg.width();
	int sh = srcimg.height();
	int c  = srcimg.channel();

	if (tw != dstimg.width() || th != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(tw,th,c);
	}

	dim3 gridSize((tw + 16 - 1) / 16, (th + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf1,srcimg.data(),desc,sw,sh,srcimg.pitch()));
		resizeImageKernelFloat1<<<gridSize, blockSize>>>(tw,th,sw,sh,scale,(float*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 2:
		desc = cudaCreateChannelDesc<float2>();
		iproctexf2.filterMode = cudaFilterModePoint;
		iproctexf2.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf2,srcimg.data(),desc,sw,sh,srcimg.pitch()));
		resizeImageKernelFloat2<<<gridSize, blockSize>>>(tw,th,sw,sh,scale,(float2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf4,srcimg.data(),desc,sw,sh,srcimg.pitch()));
		resizeImageKernelFloat4<<<gridSize, blockSize>>>(tw,th,sw,sh,scale,(float4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}


void ImgProcessGPU::bilateralFilterUchar(const BaseImgGPU<unsigned char>& srcimg,BaseImgGPU<unsigned char>& dstimg, int radius, float sigmaintensity, float sigmaspace)
{
    assert(srcimg.data());
	if (!srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<unsigned char>();
		iproctexc1n.filterMode = cudaFilterModePoint;
		iproctexc1n.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexc1n,srcimg.data(),desc,w,h,srcimg.pitch()));
		bilateralFilterKernelUchar1<<<gridSize, blockSize>>>(w,h,radius,sigmaintensity,sigmaspace,(unsigned char*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 2:
		desc = cudaCreateChannelDesc<uchar2>();
        iproctexc2n.filterMode = cudaFilterModePoint;
        iproctexc2n.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexc2n,srcimg.data(),desc,w,h,srcimg.pitch()));
		bilateralFilterKernelUchar2<<<gridSize, blockSize>>>(w,h,radius,sigmaintensity,sigmaspace,(uchar2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 4:
		desc = cudaCreateChannelDesc<uchar4>();
		iproctexc4n.filterMode = cudaFilterModePoint;
		iproctexc4n.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexc4n,srcimg.data(),desc,w,h,srcimg.pitch()));
		bilateralFilterKernelUchar4<<<gridSize, blockSize>>>(w,h,radius,sigmaintensity,sigmaspace,(uchar4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}

void ImgProcessGPU::bilateralFilterFloat(const BaseImgGPU<float>& srcimg,BaseImgGPU<float>& dstimg, int radius, float sigmaintensity, float sigmaspace)
{
    assert(srcimg.data());
	if (!srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf1,srcimg.data(),desc,w,h,srcimg.pitch()));
		bilateralFilterKernelFloat1<<<gridSize, blockSize>>>(w,h,radius,sigmaintensity,sigmaspace,(float*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 2:
		desc = cudaCreateChannelDesc<float2>();
		iproctexf2.filterMode = cudaFilterModePoint;
		iproctexf2.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf2,srcimg.data(),desc,w,h,srcimg.pitch()));
		bilateralFilterKernelFloat2<<<gridSize, blockSize>>>(w,h,radius,sigmaintensity,sigmaspace,(float2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf4,srcimg.data(),desc,w,h,srcimg.pitch()));
		bilateralFilterKernelFloat4<<<gridSize, blockSize>>>(w,h,radius,sigmaintensity,sigmaspace,(float4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}


void ImgProcessGPU::copyFromFloatToUchar(const BaseImgGPU<float>& srcimg, BaseImgGPU<unsigned char>& dstimg)
{
	if (!srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);

	copyFromFloatToUcharKernel<<<gridSize, blockSize>>>(w,h,c,
		srcimg.data(),srcimg.pitch(),
		dstimg.data(),dstimg.pitch());

	cudaDeviceSynchronize();
}

void ImgProcessGPU::copyFromUcharToFloat(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<float>& dstimg)
{
	if (!srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);

	copyFromUcharToFloatKernel<<<gridSize, blockSize>>>(w,h,c,
		srcimg.data(),srcimg.pitch(),
		dstimg.data(),dstimg.pitch());

	cudaDeviceSynchronize();
}

template <class T1, class T2>
void ImgProcessGPU::cropImage(const BaseImgGPU<T1>& srcimg, BaseImgGPU<T2>& dstimg, int left, int top, int tw, int th)
{
	if (!srcimg.data())
	{
		return;
	}
	int sw = srcimg.width();
	int sh = srcimg.height();
	int c = srcimg.channel();
	if (left + tw > sw || top + th > sh)
	{
		return;
	}
	if (tw != dstimg.width() || th != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(tw, th, c);
	}

	dim3 gridSize((tw + 16 - 1) / 16, (th + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cropImageKernel<T1,T2><<<gridSize,blockSize>>>(tw, th, c, left, top, srcimg.data(), srcimg.pitch(), dstimg.data(), dstimg.pitch());
	cudaDeviceSynchronize();
}

void ImgProcessGPU::cropImageUchar(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg, int left, int top, int tw, int th)
{
	if (!srcimg.data())
	{
		return;
	}
	int sw = srcimg.width();
	int sh = srcimg.height();
	int c  = srcimg.channel();
	if (left + tw > sw || top + th > sh)
	{
		return;
	}
	if (tw != dstimg.width() || th != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(tw,th,c);
	}

	dim3 gridSize((tw + 16 - 1) / 16, (th + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cropImageKernel<unsigned char,unsigned char><<<gridSize,blockSize>>>(tw,th,c,left,top,srcimg.data(),srcimg.pitch(),dstimg.data(),dstimg.pitch());
	cudaDeviceSynchronize();
}

void ImgProcessGPU::cropImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int left, int top, int tw, int th)
{
	if (!srcimg.data())
	{
		return;
	}
	int sw = srcimg.width();
	int sh = srcimg.height();
	int c  = srcimg.channel();
	if (left + tw > sw || top + th > sh)
	{
		return;
	}
	if (tw != dstimg.width() || th != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(tw,th,c);
	}

	dim3 gridSize((tw + 16 - 1) / 16, (th + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cropImageKernel<float,float><<<gridSize,blockSize>>>(tw,th,c,left,top,srcimg.data(),srcimg.pitch(),dstimg.data(),dstimg.pitch());
	cudaDeviceSynchronize();
}

void ImgProcessGPU::hFilteringFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float* dFilter1D, int radius)
{
	if (!srcimg.data() || !dFilter1D)
	{
		return;
	}

	int w  = srcimg.width();
	int h  = srcimg.height();
	int c  = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	int len = 2 * radius + 1;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf1,srcimg.data(),desc,w,h,srcimg.pitch()));
		hFilteringKernelFloat1<<<gridSize, blockSize, sizeof(float)*len>>>(w,h,dFilter1D,radius,(float*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		break;

	case 2:
		desc = cudaCreateChannelDesc<float2>();
		iproctexf2.filterMode = cudaFilterModePoint;
		iproctexf2.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf2,srcimg.data(),desc,w,h,srcimg.pitch()));
		hFilteringKernelFloat2<<<gridSize, blockSize, sizeof(float)*len>>>(w,h,dFilter1D,radius,(float2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf4,srcimg.data(),desc,w,h,srcimg.pitch()));
		hFilteringKernelFloat4<<<gridSize, blockSize, sizeof(float)*len>>>(w,h,dFilter1D,radius,(float4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		break;
	}
}

void ImgProcessGPU::vFilteringFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float* dFilter1D, int radius)
{
	if (!srcimg.data() || !dFilter1D)
	{
		return;
	}

	int w  = srcimg.width();
	int h  = srcimg.height();
	int c  = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	int len = 2 * radius + 1;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf1,srcimg.data(),desc,w,h,srcimg.pitch()));
		vFilteringKernelFloat1<<<gridSize, blockSize, sizeof(float)*len>>>(w,h,dFilter1D,radius,(float*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		break;

	case 2:
		desc = cudaCreateChannelDesc<float2>();
		iproctexf2.filterMode = cudaFilterModePoint;
		iproctexf2.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf2,srcimg.data(),desc,w,h,srcimg.pitch()));
		vFilteringKernelFloat2<<<gridSize, blockSize, sizeof(float)*len>>>(w,h,dFilter1D,radius,(float2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf4,srcimg.data(),desc,w,h,srcimg.pitch()));
		vFilteringKernelFloat4<<<gridSize, blockSize, sizeof(float)*len>>>(w,h,dFilter1D,radius,(float4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		break;
	}
}

void ImgProcessGPU::filteringFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float* dFilter2D, int radius)
{
	if (!srcimg.data() || !dFilter2D)
	{
		return;
	}

	int w  = srcimg.width();
	int h  = srcimg.height();
	int c  = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	int len = 2 * radius + 1;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf1,srcimg.data(),desc,w,h,srcimg.pitch()));
		filteringKernelFloat1<<<gridSize, blockSize, sizeof(float)*len*len>>>(w,h,dFilter2D,radius,dstimg.data(),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 2:
		desc = cudaCreateChannelDesc<float2>();
		iproctexf2.filterMode = cudaFilterModePoint;
		iproctexf2.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf2,srcimg.data(),desc,w,h,srcimg.pitch()));
		filteringKernelFloat2<<<gridSize, blockSize, sizeof(float)*len*len>>>(w,h,dFilter2D,radius,(float2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf4,srcimg.data(),desc,w,h,srcimg.pitch()));
		filteringKernelFloat4<<<gridSize, blockSize, sizeof(float)*len*len>>>(w,h,dFilter2D,radius,(float4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}

void ImgProcessGPU::gaussianFilterFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int radius, float sigma)
{
	int len = 2 * radius + 1;
	float* dFilter2D = 0;
	checkCudaErrors(cudaMalloc(&dFilter2D,sizeof(float)*len*len));
	generate2DGaussianGPU(dFilter2D,radius,sigma);
	filteringFloat(srcimg,dstimg,dFilter2D,radius);
	cudaFree(dFilter2D);
}

void ImgProcessGPU::warpImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, const BaseImgGPU<float>& flow)
{
	if (!srcimg.data() || !flow.data())
	{
		return;
	}

	int w  = srcimg.width();
	int h  = srcimg.height();
	int c  = srcimg.channel();

	if (w != flow.width() || h != flow.height())
	{
		return;
	}

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w,h,c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	cudaChannelFormatDesc desc;
	cudaChannelFormatDesc desc_uv = cudaCreateChannelDesc<float2>();
	iprocauxf2.filterMode = cudaFilterModePoint;
	iprocauxf2.normalized = false;
	checkCudaErrors(cudaBindTexture2D(0,iprocauxf2,flow.data(),desc_uv,w,h,flow.pitch()));

	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf1,srcimg.data(),desc,w,h,srcimg.pitch()));
		warpImageKernelFloat1<<<gridSize, blockSize>>>(w,h,(float*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 2:
		desc = cudaCreateChannelDesc<float2>();
		iproctexf2.filterMode = cudaFilterModePoint;
		iproctexf2.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf2,srcimg.data(),desc,w,h,srcimg.pitch()));
		warpImageKernelFloat2<<<gridSize, blockSize>>>(w,h,(float2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0,iproctexf4,srcimg.data(),desc,w,h,srcimg.pitch()));
		warpImageKernelFloat4<<<gridSize, blockSize>>>(w,h,(float4*)(dstimg.data()),(size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}

template <typename FT>
void ImgProcessGPU::vFlipImage(const BaseImgGPU<FT>& srcimg, BaseImgGPU<FT>& dstimg)
{
	if (!srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	vFlipImageKernel<FT><<<gridSize,blockSize>>>(w, h, c, srcimg.data(), srcimg.pitch(), dstimg.data(), dstimg.pitch());
	cudaDeviceSynchronize();
}

template <typename FT>
void ImgProcessGPU::hFlipImage(const BaseImgGPU<FT>& srcimg, BaseImgGPU<FT>& dstimg)
{
	if (!srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	hFlipImageKernel<FT><<<gridSize,blockSize>>>(w, h, c, srcimg.data(), srcimg.pitch(), dstimg.data(), dstimg.pitch());
	cudaDeviceSynchronize();
}

template void ImgProcessGPU::cropImage<float, float>(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int left, int top, int tw, int th);
template void ImgProcessGPU::cropImage<float, unsigned char>(const BaseImgGPU<float>& srcimg, BaseImgGPU<unsigned char>& dstimg, int left, int top, int tw, int th);
template void ImgProcessGPU::cropImage<unsigned char, unsigned char>(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg, int left, int top, int tw, int th);
template void ImgProcessGPU::cropImage<unsigned char, float>(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<float>& dstimg, int left, int top, int tw, int th);

template void ImgProcessGPU::vFlipImage<float>(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg);
template void ImgProcessGPU::hFlipImage<float>(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg);
template void ImgProcessGPU::vFlipImage<unsigned char>(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg);
template void ImgProcessGPU::hFlipImage<unsigned char>(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg);

void ImgProcessGPU::jointBilateralFilterUchar(const BaseImgGPU<unsigned char>& srcref, const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int radius, float sigmaintensity, float sigmaref, float sigmaspace)
{
	if (!srcref.data() || !srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);

	// bind the reference frame
	cudaChannelFormatDesc descc4 = cudaCreateChannelDesc<uchar4>();
	iprocauxc4n.filterMode = cudaFilterModePoint;
	iprocauxc4n.normalized = true;
	checkCudaErrors(cudaBindTexture2D(0, iprocauxc4n, srcref.data(), descc4, srcref.width(), srcref.height(), srcref.pitch()));

	cudaChannelFormatDesc desc;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0, iproctexf1, srcimg.data(), desc, w, h, srcimg.pitch()));
		jointBilateralFilterKernelUchar1<<<gridSize,blockSize>>>(w, h, radius, sigmaintensity, sigmaspace, sigmaref, (float*)(dstimg.data()), (size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0, iproctexf4, srcimg.data(), desc, w, h, srcimg.pitch()));
		jointBilateralFilterKernelUchar4<<<gridSize,blockSize>>>(w, h, radius, sigmaintensity, sigmaspace, sigmaref, (float4*)(dstimg.data()), (size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}

void ImgProcessGPU::jointBilateralFilterFloat(
		const BaseImgGPU<float>& srcref, 
		const BaseImgGPU<float>& srcimg, 
		BaseImgGPU<float>& dstimg, 
		int radius, 
		float sigmaintensity, 
		float sigmaref, 
		float sigmaspace)
{
	if (!srcref.data() || !srcimg.data())
	{
		return;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	dim3 gridSize((w + 16 - 1) / 16, (h + 16 - 1) / 16);
	dim3 blockSize(16, 16);

	// bind the reference frame
	cudaChannelFormatDesc descf4 = cudaCreateChannelDesc<float4>();
	iprocauxf4.filterMode = cudaFilterModePoint;
	iprocauxf4.normalized = true;
	checkCudaErrors(cudaBindTexture2D(0, iprocauxf4, srcref.data(), descf4, srcref.width(), srcref.height(), srcref.pitch()));

	cudaChannelFormatDesc desc;
	switch (c)
	{
	case 1:
		desc = cudaCreateChannelDesc<float>();
		iproctexf1.filterMode = cudaFilterModePoint;
		iproctexf1.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0, iproctexf1, srcimg.data(), desc, w, h, srcimg.pitch()));
		jointBilateralFilterKernelFloat1<<<gridSize,blockSize>>>(w, h, radius, sigmaintensity, sigmaspace, sigmaref, (float*)(dstimg.data()), (size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;

		//	case 2:
		//		desc = cudaCreateChannelDesc<float2>();
		//		iproctexf2.filterMode = cudaFilterModePoint;
		//		iproctexf2.normalized = false;
		//		checkCudaErrors(cudaBindTexture2D(0,iproctexf2,srcimg.data(),desc,w,h,srcimg.pitch()));
		////		bilateralFilterKernelFloat2<<<gridSize, blockSize>>>(w,h,radius,sigmaintensity,sigmaspace,(float2*)(dstimg.data()),(size_t)(dstimg.pitch()));
		//		cudaDeviceSynchronize();
		//		break;

	case 4:
		desc = cudaCreateChannelDesc<float4>();
		iproctexf4.filterMode = cudaFilterModePoint;
		iproctexf4.normalized = false;
		checkCudaErrors(cudaBindTexture2D(0, iproctexf4, srcimg.data(), desc, w, h, srcimg.pitch()));
		jointBilateralFilterKernelFloat4<<<gridSize,blockSize>>>(w, h, radius, sigmaintensity, sigmaspace, sigmaref, (float4*)(dstimg.data()), (size_t)(dstimg.pitch()));
		cudaDeviceSynchronize();
		break;
	}
}
