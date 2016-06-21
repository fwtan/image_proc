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
#ifndef _BASE_IMG_GPU_H_
#define _BASE_IMG_GPU_H_

#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "BaseImgCPU.h"

template <typename FT>
class BaseImgGPU
{
public:
	typedef FT value_type;

    BaseImgGPU(): dPtr_(NULL), dPitch_(0), width_(0), height_(0), channel_(0){}
	~BaseImgGPU(){ clear(); }

	BaseImgGPU(const BaseImgGPU& dImg)
	{
		copy_from(dImg);
	}

	BaseImgGPU& operator=(const BaseImgGPU& dImg)
	{
		copy_from(dImg);
		return *this;
	}

    void init(int w, int h, int c)
	{
        assert(w * h * c > 0);
		if (w * h * c <= 0)	
            return;

		clear();

		width_   = w;
		height_  = h;
		channel_ = c;

		checkCudaErrors(cudaMallocPitch(&(dPtr_), &(dPitch_),  sizeof(value_type) * channel_ * width_, height_));
		checkCudaErrors(cudaMemset2D(dPtr_, dPitch_, 0, sizeof(value_type) * channel_ * width_, height_));

        return;
	}

	virtual void clear() 
	{ 
		if (dPtr_ != NULL)
		{
			checkCudaErrors(cudaFree(dPtr_));
			dPtr_ = NULL;
		}
		width_ = 0;
		height_ = 0;
		channel_ = 0;
	}
	
	void reset()
	{
        assert(dPtr_);
		if (dPtr_ == NULL) return;
		checkCudaErrors(cudaMemset2D(dPtr_, dPitch_, 0, sizeof(value_type) * channel_ * width_, height_));
	}

	FT*  data() { return dPtr_; }
	const FT* data() const { return dPtr_; }

	bool copy_from(const BaseImgCPU<FT>& hImg) 
	{
		if (!hImg.data())
		{
			return false;
		}

		if (hImg.width() != width_ || hImg.height() != height_ || hImg.channel() != channel_)
		{
			init(hImg.width(), hImg.height(), hImg.channel());
		}

		checkCudaErrors(cudaMemcpy2D(dPtr_, dPitch_, hImg.data(), sizeof(value_type) * channel_ * width_,
			sizeof(value_type) * channel_ * width_, height_,
			cudaMemcpyHostToDevice));

		return true;
	}

	bool copy_from(const BaseImgGPU<FT>& dImg) 
	{
		if (!dImg.data())
		{
			return false;
		}

		if (dImg.width() != width_ || dImg.height() != height_ || dImg.channel() != channel_)
		{
			init(dImg.width(), dImg.height(), dImg.channel());
		}

		checkCudaErrors(cudaMemcpy2D(dPtr_, dPitch_, dImg.data(), dImg.pitch(),
			sizeof(value_type) * channel_ * width_, height_,
			cudaMemcpyDeviceToDevice));

		return true;
	}

	bool copy_to(BaseImgCPU<FT>& hImg) const
	{
		if (!dPtr_)
		{
			return false;
		}

		if (hImg.width() != width_ || hImg.height() != height_ || hImg.channel() != channel_)
		{
			hImg.init(width_, height_, channel_);
		}

		checkCudaErrors(cudaMemcpy2D(hImg.data(), sizeof(value_type) * channel_ * width_, dPtr_, dPitch_,
			sizeof(value_type) * channel_ * width_, height_,
			cudaMemcpyDeviceToHost));

		return true;
	}
	bool copy_to(BaseImgGPU<FT>& dImg) const
	{
		if (!dPtr_)
		{
			return false;
		}

		if (dImg.width() != width_ || dImg.height() != height_ || dImg.channel() != channel_)
		{
			dImg.init(width_, height_, channel_);
		}

		checkCudaErrors(cudaMemcpy2D(dImg.data(), dImg.pitch(), dPtr_, dPitch_,
			sizeof(value_type) * channel_ * width_, height_,
			cudaMemcpyDeviceToDevice));

		return true;
	}

	int width() const { return width_; }
	int height() const { return height_; }
	int channel() const { return channel_; }
	int pitch() const { return dPitch_; }

private:
	FT*    dPtr_;
	size_t dPitch_;

	int width_;
	int height_;
	int channel_;
};

#endif
