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
#ifndef _IMG_PROCESS_GPU_H_
#define _IMG_PROCESS_GPU_H_

#include <assert.h>
#include "BaseImgGPU.h"
#include "HelperMathExt.h"

class ImgProcessGPU
{
public:
	explicit ImgProcessGPU(){}
	virtual ~ImgProcessGPU(){}

public:
	// image resize
	void resizeImageUchar(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg, float ratio,   float scale);
	void resizeImageUchar(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg, int tw, int th, float scale);
	void resizeImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float ratio,   float scale);
	void resizeImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int tw, int th, float scale);

	// bilateral filtering
	void bilateralFilterUchar(const BaseImgGPU<unsigned char>& srcimg,BaseImgGPU<unsigned char>& dstimg, int radius, float sigmaintensity, float sigmaspace);
	void bilateralFilterFloat(const BaseImgGPU<float>& srcimg,BaseImgGPU<float>& dstimg, int radius, float sigmaintensity, float sigmaspace);

	// format conversion
	void copyFromFloatToUchar(const BaseImgGPU<float>& srcimg, BaseImgGPU<unsigned char>& dstimg);
	void copyFromUcharToFloat(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<float>& dstimg);

	// general filtering
	void hFilteringFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float* dFilter1D, int radius);
	void vFilteringFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float* dFilter1D, int radius);
	void filteringFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, float* dFilter2D,  int radius);
	void gaussianFilterFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int radius, float sigma = -1);
	void laplacianFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg);
	// warping
	void warpImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, const BaseImgGPU<float>& flow);

	// crop
	template <class T1, class T2>
	void cropImage(const BaseImgGPU<T1>& srcimg, BaseImgGPU<T2>& dstimg, int left, int top, int tw, int th);
	
	// flip
	template <typename FT>
	void vFlipImage(const BaseImgGPU<FT>& srcimg, BaseImgGPU<FT>& dstimg);
	template <typename FT>
	void hFlipImage(const BaseImgGPU<FT>& srcimg, BaseImgGPU<FT>& dstimg);


	template <class T>
	static void generate1DGaussianCPU(T*& hFilter1D, int radius, double sigma=-1);
	template <class T>
	static void generate2DGaussianCPU(T*& hFilter2D, int radius, double sigma=-1);
	template <class T>
	static void generate1DGaussianGPU(T*& dFilter1D, int radius, double sigma=-1);
	template <class T>
	static void generate2DGaussianGPU(T*& dFilter2D, int radius, double sigma=-1);
	

	void jointBilateralFilterUchar(const BaseImgGPU<unsigned char>& srcref,
								   const BaseImgGPU<float>& srcimg,
								   BaseImgGPU<float>& dstimg, 
								   int radius, 
								   float sigmaintensity, 
								   float sigmaref, 
								   float sigmaspace);
	void jointBilateralFilterFloat(const BaseImgGPU<float>& srcref,
								   const BaseImgGPU<float>& srcimg,
								   BaseImgGPU<float>& dstimg, 
								   int radius, 
								   float sigmaintensity, 
								   float sigmaref, 
								   float sigmaspace);

private:
	// crop
	void cropImageUchar(const BaseImgGPU<unsigned char>& srcimg, BaseImgGPU<unsigned char>& dstimg, int left, int top, int tw, int th);
	void cropImageFloat(const BaseImgGPU<float>& srcimg, BaseImgGPU<float>& dstimg, int left, int top, int tw, int th);
};

//template <class T1,class T2>
//void ImgProcessGPU::ResizeImage(const BaseImgGPU<T1>* pSrcimg, BaseImgGPU<T2>* pDstimg, float ratio)
//{
//	if (!pSrcimg || !pSrcimg->data() || !pDstimg)
//	{
//		return;
//	}
//
//	int sw = pSrcimg->width();
//	int sh = pSrcimg->height();
//
//	int tw = sw * ratio + 0.5;
//	int th = sh * ratio + 0.5;
//
//	ResizeImage<T1,T2>(pSrcimg,pDstimg,tw,th);
//}
//
//template <class T1,class T2>
//void ImgProcessGPU::ResizeImage(const BaseImgGPU<T1>* pSrcimg, BaseImgGPU<T2>* pDstimg, int tw, int th)
//{
//	if (!pSrcimg || !pSrcimg->data() || !pDstimg)
//	{
//		return;
//	}
//
//	if (typeid(T1) == typeid(float) && typeid(T2) == typeid(float))
//	{
//		ResizeImageFloat((*pSrcimg),(*pDstimg),tw,th);
//	}
//	else
//	{
//		ResizeImageUchar((*pSrcimg),(*pDstimg),tw,th);
//	}
//}

//------------------------------------------------------------------------------------------------------------
// function to generate a 2D Gaussian image
// pImage must be allocated before calling the function
//------------------------------------------------------------------------------------------------------------
template <class T>
void ImgProcessGPU::generate2DGaussianCPU(T*& hFilter2D, int radius, double sigma)
{
	if(sigma==-1)	sigma=radius/2;
	double alpha=1/(2*sigma*sigma);
	int winlength=radius*2+1;
	if(hFilter2D==NULL)
        hFilter2D=new T[winlength*winlength];
	double total = 0;
	for(int i=-radius;i<=radius;i++)
		for(int j=-radius;j<=radius;j++)
		{
			hFilter2D[(i+radius)*winlength+j+radius]=exp(-(double)(i*i+j*j)*alpha);
			total += hFilter2D[(i+radius)*winlength+j+radius];
		}
		for(int i = 0;i<winlength*winlength;i++)
			hFilter2D[i]/=total;
}

//------------------------------------------------------------------------------------------------------------
// function to generate a 1D Gaussian image
// pImage must be allocated before calling the function
//------------------------------------------------------------------------------------------------------------
template <class T>
void ImgProcessGPU::generate1DGaussianCPU(T*& hFilter1D, int radius, double sigma)
{
	if(sigma==-1)	sigma=radius/2;
	double alpha=1/(2*sigma*sigma);
	int winlength=radius*2+1;
	if(hFilter1D==NULL)
        hFilter1D=new T[winlength];
	double total = 0;
	for(int i=-radius;i<=radius;i++)
	{
		hFilter1D[i+radius]=exp(-(double)(i*i)*alpha);
		total += hFilter1D[i+radius];
	}
	for(int i = 0;i<winlength;i++)
		hFilter1D[i]/=total;
}

template <class T>
void ImgProcessGPU::generate1DGaussianGPU(T*& dFilter1D, int radius, double sigma)
{
	int length = 2 * radius + 1;
	int numOfElements = length;
	if (dFilter1D == NULL)
	{
		cudaMalloc(&dFilter1D,numOfElements*sizeof(T));
	}
    T* hFilter1D = new T[numOfElements];
	generate1DGaussianCPU(hFilter1D,radius,sigma);
	cudaMemcpy(dFilter1D,hFilter1D,numOfElements*sizeof(T),cudaMemcpyHostToDevice);
	delete hFilter1D;
	hFilter1D = 0;
}

template <class T>
void ImgProcessGPU::generate2DGaussianGPU(T*& dFilter2D, int radius, double sigma)
{
	int length = 2 * radius + 1;
	int numOfElements = length * length;
	if (dFilter2D == NULL)
	{
		cudaMalloc(&dFilter2D,numOfElements*sizeof(T));
	}
    T* hFilter2D = new T[numOfElements];
	generate2DGaussianCPU(hFilter2D,radius,sigma);
	cudaMemcpy(dFilter2D,hFilter2D,numOfElements*sizeof(T),cudaMemcpyHostToDevice);
	delete hFilter2D;
	hFilter2D = 0;
}

#endif
