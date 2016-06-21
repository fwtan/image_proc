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


#ifndef _BASE_IMG_CPU_H_
#define _BASE_IMG_CPU_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

template <typename FT>
class BaseImgCPU
{
public:
	typedef FT value_type;

	BaseImgCPU() : width_(0), height_(0), channel_(0), array_(0) {}
	~BaseImgCPU() { clear(); }

	BaseImgCPU(const BaseImgCPU& hImg)
	{
		copy_from(hImg);
	}

	BaseImgCPU& operator=(const BaseImgCPU& hImg)
	{
		copy_from(hImg);
		return *this;
	}

    void init(int w, int h, int c)
	{
        assert(w * h * c > 0);
        if (w * h * c <= 0)
            return;

        clear();

        width_ = w;
        height_ = h;
        channel_ = c;
        array_  = (FT*)malloc(width_ * height_ * channel_ * sizeof(FT));
        if (!array_)
        {
            printf("Memory Limit.\n");
            return;
        }
        memset(array_, 0, width_ * height_ * channel_ * sizeof(FT));
        return;
	}

	virtual void clear()
	{
		if (array_ != 0)
		{
			free(array_);
			array_ = 0;
		}

		width_ = 0;
		height_ = 0;
		channel_ = 0;
	}

	void reset()
	{
        assert(array_);
		if (!array_) return;
		memset(array_, 0, sizeof(value_type) * width_ * height_ * channel_);
	}

	FT*	     data()		  { return array_; }
	const FT* data() const { return array_; }
	int	width()		const { return width_; }
	int	height()	const { return height_; }
	int channel()	const { return channel_; }

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

		memcpy(array_, hImg.data(), sizeof(value_type) * channel_ * width_ * height_);

		return true;
	}

	bool copy_to(BaseImgCPU<FT>& hImg) const
	{
		if (!array_)
		{
			return false;
		}

		if (hImg.width() != width_ || hImg.height() != height_ || hImg.channel() != channel_)
		{
			hImg.init(width_, height_, channel_);
		}

		memcpy(hImg.data(), array_, sizeof(value_type) * channel_ * width_ * height_);

		return true;
	}

	FT* operator()(int x, int y)  
	{
		return &array_[(x + y * width_)*channel_];
	}
	const FT* operator()(int x, int y) const
	{
		return &array_[(x + y * width_)*channel_];
	}

private:
	int width_;
	int height_;
	int channel_;
	FT*  array_;
};

#endif//_BASE_IMG_CPU_H_
