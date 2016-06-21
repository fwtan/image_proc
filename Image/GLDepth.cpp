
#include "stdafx.h"
#include "GLDepth.h"

void GLDepth::init(int w, int h)
{
	if (w * h <= 0)	return;

	clear();

	width_ = w;
	height_ = h;

	glGenTextures(1, &dTexID_);
	glBindTexture(GL_TEXTURE_2D, dTexID_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void GLDepth::clear()
{
	if (glIsTexture(dTexID_) == GL_TRUE)
	{
		glDeleteTextures(1, &(dTexID_));
	}

	width_ = 0;
	height_ = 0;
}