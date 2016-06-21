
#ifndef _GL_IMAGE_H_
#define _GL_IMAGE_H_

#include <iostream>
#include "../GLToolkit/GLTools.h"

template <typename FT>
class GLImage
{
public:
	typedef FT value_type;

	GLImage() :width_(0), height_(0), channel_(0), dTexID_(0){}
	~GLImage(){ clear(); }

	void init(int w, int h, int c);
	void clear();

	unsigned int data() const { return dTexID_;  }

	int width() const { return width_; }
	int height() const { return height_; }
	int channel() const { return channel_; }

private:
	unsigned int dTexID_;
	int width_;
	int height_;
	int channel_;
};

template <typename FT>
void GLImage<FT>::init(int w, int h, int c)
{
	if (w * h * c <= 0 || c > 4)
	{
		return;
	}

	if (typeid(FT) != typeid(unsigned char) && typeid(FT) != typeid(float)) return;

	clear();

	width_ = w;
	height_ = h;
	channel_ = c;

	glGenTextures(1, &dTexID_);
	glBindTexture(GL_TEXTURE_2D, dTexID_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	if (typeid(FT) == typeid(unsigned char))
	{
		switch (c)
		{
		case 1:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
			break;
		case 2:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, w, h, 0, GL_RG, GL_UNSIGNED_BYTE, 0);
			break;
		case 3:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
			break;
		case 4:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			break;
		default:
			break;
		}
	}
	else if (typeid(FT) == typeid(float))
	{
		switch (c)
		{
		case 1:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, 0);
			break;
		case 2:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, w, h, 0, GL_RG, GL_FLOAT, 0);
			break;
		case 3:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, 0);
			break;
		case 4:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
			break;
		default:
			break;
		}
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	CHECKGLERRORS;
}

template <typename FT>
void GLImage<FT>::clear()
{
	if (glIsTexture(dTexID_) == GL_TRUE)
	{
		glDeleteTextures(1, &(dTexID_));
	}

	width_ = 0;
	height_ = 0;
	channel_ = 0;
	CHECKGLERRORS;
}

#endif//_GL_IMAGE_H_