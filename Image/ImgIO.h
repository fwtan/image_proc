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

#ifndef _IMG_IO_H_
#define _IMG_IO_H_

#include <assert.h>
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/highgui.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>

#include "BaseImgCPU.h"
#include "BaseImgGPU.h"

class ImgIO
{
public:
	enum ImageType{standard, derivative, normalized};
	explicit ImgIO(){}
	virtual ~ImgIO(){}
public:
	template <class T>
	static bool loadImageCPU(const char* filename, BaseImgCPU<T>& img);
	template <class T>
	static bool loadImageGPU(const char* filename, BaseImgGPU<T>& img);
	template <class T>
	static bool saveImageCPU(const char* filename, const BaseImgCPU<T>& img, ImageType imtype = standard);
	template <class T>
	static bool saveImageGPU(const char* filename, BaseImgGPU<T>& img, ImageType imtype = standard);
	template <class T>
	static bool loadArrayCPU(const char* filename, BaseImgCPU<T>& img);
	template <class T>
	static bool loadArrayGPU(const char* filename, BaseImgGPU<T>& img);
	template <class T>
	static bool saveArrayCPU(const char* filename, BaseImgCPU<T>& img);
	template <class T>
	static bool saveArrayGPU(const char* filename, BaseImgGPU<T>& img);


	template <class T>
	static bool base_to_cv(const BaseImgCPU<T>& hImg, cv::Mat& cvImg);

	template <class T>
	static bool base_to_cv(const BaseImgGPU<T>& dImg, cv::Mat& cvImg);

	template <class T>
	static bool cv_to_base(const cv::Mat& cvImg, BaseImgCPU<T>& hImg, bool bNeedAlpha = true);

	template <class T>
	static bool cv_to_base(const cv::Mat& cvImg, BaseImgGPU<T>& dImg, bool bNeedAlpha = true);
};

template <class T>
bool ImgIO::cv_to_base(const cv::Mat& cvImg, BaseImgCPU<T>& hImg, bool bNeedAlpha)
{
	if (cvImg.data == NULL)
	{
		return false;
	}

	bool isBaseFloat = false;
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(long double))
		isBaseFloat = true;

	int width  = cvImg.cols;
	int height = cvImg.rows;
	int channel = cvImg.channels(); 

	if (channel > 1 && bNeedAlpha)
	{
		hImg.init(width, height, 4);
	}
	else
	{
		hImg.init(width, height, channel);
	}

	switch (cvImg.depth())
	{
	case CV_8U:
		for (int y = 0; y < height; ++y)
		{
			const uchar* p = cvImg.ptr<uchar>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					if (isBaseFloat)
					{
						hImg(x,y)[k] = static_cast<T>(p[offset + k])/255;
					}
					else
					{
						hImg(x,y)[k] = p[offset + k];
					}
				}
			}
		}
		break;
	case CV_8S:
		for (int y = 0; y < height; ++y)
		{
			const char* p = cvImg.ptr<char>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					if (isBaseFloat)
					{
						hImg(x,y)[k] = static_cast<T>(p[offset + k])/255;
					}
					else
					{
						hImg(x,y)[k] = p[offset + k];
					}
				}
			}
		}
		break;
	case CV_16U:
		for (int y = 0; y < height; ++y)
		{
			const ushort* p = cvImg.ptr<ushort>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					if (isBaseFloat)
					{
						hImg(x,y)[k] = static_cast<T>(p[offset + k])/255;
					}
					else
					{
						hImg(x,y)[k] = p[offset + k];
					}
				}
			}
		}
		break;
	case CV_16S:
		for (int y = 0; y < height; ++y)
		{
			const short* p = cvImg.ptr<short>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					if (isBaseFloat)
					{
						hImg(x,y)[k] = static_cast<T>(p[offset + k])/255;
					}
					else
					{
						hImg(x,y)[k] = p[offset + k];
					}
				}
			}
		}
		break;
	case CV_32S:
		for (int y = 0; y < height; ++y)
		{
			const int* p = cvImg.ptr<int>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					if (isBaseFloat)
					{
						hImg(x,y)[k] = static_cast<T>(p[offset + k])/255;
					}
					else
					{
						hImg(x,y)[k] = p[offset + k];
					}
				}
			}
		}
		break;
	case CV_32F:
		for (int y = 0; y < height; ++y)
		{
			const float* p = cvImg.ptr<float>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					if (isBaseFloat)
					{
						hImg(x,y)[k] = p[offset + k];
					}
					else
					{
						hImg(x,y)[k] = std::min(std::max(static_cast<T>(p[offset + k] * 255), T(0)), T(255));
					}
				}
			}
		}
		break;
	case CV_64F:
		for (int y = 0; y < height; ++y)
		{
			const double* p = cvImg.ptr<double>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					if (isBaseFloat)
					{
						hImg(x,y)[k] = p[offset + k];
					}
					else
					{
						hImg(x,y)[k] = std::min(std::max(static_cast<T>(p[offset + k] * 255), T(0)), T(255));
					}
				}
			}
		}
		break;
	default:
		printf("Unsupported cv image type.\n");
	}

	return true;
}

template <class T>
bool ImgIO::cv_to_base(const cv::Mat& cvImg, BaseImgGPU<T>& dImg, bool bNeedAlpha)
{
	BaseImgCPU<T> hImg;
	return ( cv_to_base<T>(cvImg, hImg, bNeedAlpha) && dImg.copy_from(hImg) );
}

template <class T>
bool ImgIO::base_to_cv(const BaseImgCPU<T>& hImg, cv::Mat& cvImg)
{
	if (!hImg.data())
	{
		return false;
	}

	int width   = hImg.width();
	int height  = hImg.height();
	int channel = hImg.channel();

	if (channel == 4)
	{
		channel = 3;
	}

	bool isBaseFloat = false;
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(long double))
		isBaseFloat = true;

	if (isBaseFloat)
	{
		cvImg.create(height, width, CV_32FC(channel));
		for (int y = 0; y < height; ++y)
		{
			float* p = cvImg.ptr<float>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					p[offset + k] = hImg(x,y)[k];
				}
			}
		}
	}
	else
	{
		cvImg.create(height, width, CV_8UC(channel));
		for (int y = 0; y < height; ++y)
		{
			uchar* p = cvImg.ptr<uchar>(y);
			for (int x = 0; x < width; ++x)
			{
				int offset = x * channel;
				for (int k = 0; k < channel; ++k)
				{
					p[offset + k] = hImg(x,y)[k];
				}
			}
		}
	}

	return true;
}

template <class T>
bool ImgIO::base_to_cv(const BaseImgGPU<T>& dImg, cv::Mat& cvImg)
{
	BaseImgCPU<T> hImg;
	return ( dImg.copy_to(hImg) && base_to_cv<T>(hImg, cvImg) );
}


template <class T>
bool ImgIO::loadImageCPU(const char* filename, BaseImgCPU<T>& img)
{
	cv::Mat im = cv::imread(filename, cv::IMREAD_UNCHANGED);
	
	if (im.data == NULL)
	{
		return false;
	} 
		
	if(im.depth() != CV_8U)
	{
		return false;
	}

	int w = im.cols;
	int h = im.rows;
	int c = im.channels();

	if (c == 3) img.init(w, h, 4);
	else		img.init(w, h, c);

	bool IsFloat = false;
	if( typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(long double))
		IsFloat = true;

	for(int y = 0; y < h; ++y)
	{
		int offset1 = y * im.step;
		for (int x = 0; x < w; ++x)
		{
			int offset2 = x * c;
			for (int k = 0; k < c; ++k)
			{
				if (IsFloat)
					img(x,y)[k] = static_cast<T>(im.data[offset1 + offset2 + k])/255.f;
				else
					img(x,y)[k] = im.data[offset1 + offset2 + k];
			}
		}
	}

	return true;
}

template <class T>
bool ImgIO::loadImageGPU(const char* filename, BaseImgGPU<T>& img)
{
	BaseImgCPU<T> hImg;
	return ( loadImageCPU<T>(filename, hImg) && img.copy_from(hImg) );
}

template <class T>
bool ImgIO::saveImageCPU(const char* filename, const BaseImgCPU<T>& img, ImageType imtype)
{
    assert(img.data());
    if (!img.data())
        return false;

	int w = img.width();
	int h = img.height();
	int c = img.channel();

	cv::Mat im;
	switch(c)
	{
	case 1:
		im.create(h, w, CV_8UC1);
		break;
	case 4:
		im.create(h, w, CV_8UC3);
		break;
	default:
		return false;
	}

	c = im.channels();

	// check whether the type is float point
	bool IsFloat = false;
	if(typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(long double))
		IsFloat = true;

	T Max, Min;
	switch(imtype)
	{
	case standard:
		break;
	case derivative:
		// find the max of the absolute value
		Max = std::numeric_limits<T>::min();
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				for (int k = 0; k < c; ++k)
				{
                    Max = std::max(Max, static_cast<T>(std::abs(img(x,y)[k])));
				}
			}
		}
		Max *= 2;
		break;
	case normalized:
		Max = std::numeric_limits<T>::min();
		Min = std::numeric_limits<T>::max();
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				for (int k = 0; k < c; ++k)
				{
                    Max = std::max(Max, static_cast<T>(img(x,y)[k]));
                    Min = std::min(Min, static_cast<T>(img(x,y)[k]));
				}
			}
		}
		break;
	}

	for(int y = 0; y < h; ++y)
	{
		int offset1 = y * im.step;
		for (int x = 0; x < w; ++x)
		{
			int offset2 = x * c;
			for (int k = 0; k < c; ++k)
			{
				switch (imtype)
				{
				case standard:
					if (IsFloat)
						im.data[offset1 + offset2 + k] = std::min(std::max(img(x,y)[k] * 255, T(0)), T(255));
					else
                        im.data[offset1 + offset2 + k] = std::min(std::max(img(x,y)[k], T(0)), T(255));
					break;
				case ImgIO::derivative:
					im.data[offset1 + offset2 + k] = std::min(std::max((img(x,y)[k]/double(Max) + 0.5) * 255, 0.0), 255.0);
					break;
				case ImgIO::normalized:
					im.data[offset1 + offset2 + k] = std::min(std::max((img(x,y)[k] - Min)/double(Max - Min) * 255, 0.0), 255.0); 
					break;
				}
			}
		}
	}

	return cv::imwrite(filename,im);
}

template <class T>
bool ImgIO::saveImageGPU(const char* filename, BaseImgGPU<T>& img, ImageType imtype)
{
	BaseImgCPU<T> hImg;
	return (img.copy_to(hImg) && saveImageCPU(filename, hImg, imtype));
}

template <class T>
bool ImgIO::loadArrayCPU(const char* filename, BaseImgCPU<T>& img)
{
    assert(img.data());
    if (!img.data())
        return false;

	std::ifstream arrayFile;
	arrayFile.open(filename, std::ios::in | std::ios::binary);
	if (!arrayFile)	return false;
	arrayFile.read((char*) img.data(), img.width()*img.height()*img.channel()*sizeof(T));
	arrayFile.close();
	return true;
}

template <class T>
bool ImgIO::loadArrayGPU(const char* filename, BaseImgGPU<T>& img)
{
	BaseImgCPU<T> hImg;
	hImg.init(img.width(), img.height(), img.channel());

	return (loadArrayCPU<T>(filename, hImg) && img.copy_from(hImg));
}

template <class T>
bool ImgIO::saveArrayCPU(const char* filename, BaseImgCPU<T>& img)
{
    assert(img.data());
    if (!img.data())
        return false;

	std::ofstream arrayFile;
	arrayFile.open(filename, std::ios::out | std::ios::binary);
	if (!arrayFile)	return false;
	arrayFile.write((char*) img.data(), img.width()*img.height()*img.channel()*sizeof(T));
	arrayFile.close();
	return true;
}

template <class T>
bool ImgIO::saveArrayGPU(const char* filename, BaseImgGPU<T>& img)
{
	BaseImgCPU<T> hImg;
	return (img.copy_to(hImg) && saveArrayCPU(filename, hImg));
}

#endif//_IMG_IO_H_
