
#ifndef _GL_DEPTH_H_
#define _GL_DEPTH_H_

#include <GL/glew.h>

class GLDepth
{
public:
	GLDepth() :width_(0), height_(0), dTexID_(0){}
	~GLDepth() { clear(); }

	void init(int w, int h);
	void clear();

	unsigned int data() const { return dTexID_; }

	int width()  const { return width_; }
	int height() const { return height_; }

private:
	unsigned int dTexID_;
	int width_;
	int height_;
};

#endif//_GL_DEPTH_H_