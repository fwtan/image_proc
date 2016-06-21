
#ifndef _GL_INTEROP_H_
#define _GL_INTEROP_H_

#include "BaseImgGPU.h"
#include "GLImage.h"
#include <cuda_runtime.h>
#include <rendercheck_gl.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>


static struct cudaGraphicsResource* cuda_gl_res;

class GLInterop
{
public:
	template <typename FT>
	static bool cuda_to_gl(const FT* dData, size_t dPitch, int width, int height, int channel, unsigned int gltexid);

	template <typename FT>
	static bool gl_to_cuda(unsigned int gltexid, FT* dData, size_t dPitch, int width, int height, int channel);

	template <typename FT>
	static bool cuda_to_gl(const BaseImgGPU<FT>& dImage, unsigned int gltexid);

	template <typename FT>
	static bool gl_to_cuda(unsigned int gltexid, BaseImgGPU<FT>& dImage);

	template <typename FT>
	static bool cuda_to_gl(const BaseImgGPU<FT>& srcimg, GLImage<FT>& dstimg);

	template <typename FT>
	static bool gl_to_cuda(const GLImage<FT>& srcimg, BaseImgGPU<FT>& dstimg);

	template <typename FT>
	static bool cpu_to_gl(const BaseImgCPU<FT>& srcimg, GLImage<FT>& dstimg);

	template <typename FT>
	static bool gl_to_cpu(const GLImage<FT>& srcimg, BaseImgCPU<FT>& dstimg);
};

template <typename FT>
bool GLInterop::cuda_to_gl(const FT* dData, size_t dPitch, int width, int height, int channel, unsigned int gltexid)
{
	if (dData == NULL) return false;

	// register cuda resource and gl texture
	glBindTexture(GL_TEXTURE_2D, gltexid);
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_gl_res, gltexid, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

	// get array pointer of the gl texture
	cudaArray* dArray = NULL;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_gl_res, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&dArray, cuda_gl_res, 0, 0));
	checkCudaErrors(cudaMemcpy2DToArray(dArray, 0, 0,
		dData, dPitch,
		sizeof(FT)*width*channel, height,
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_gl_res, 0));

	// unregister cuda resource and gl texture
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_gl_res));
	glBindTexture(GL_TEXTURE_2D, 0);

	return true;
}

template <typename FT>
bool GLInterop::gl_to_cuda(unsigned int gltexid, FT* dData, size_t dPitch, int width, int height, int channel)
{
	if (dData == NULL) return false;

	// register cuda resource and gl texture
	glBindTexture(GL_TEXTURE_2D, gltexid);
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_gl_res, gltexid, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

	// get array pointer of the gl texture
	cudaArray* dArray = NULL;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_gl_res, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&dArray, cuda_gl_res, 0, 0));
	checkCudaErrors(cudaMemcpy2DFromArray(dData, dPitch,
		dArray, 0, 0,
		sizeof(FT)*width*channel, height,
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_gl_res, 0));

	// unregister cuda resource and gl texture
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_gl_res));
	glBindTexture(GL_TEXTURE_2D, 0);

	return true;
}

template <typename FT>
bool GLInterop::cuda_to_gl(const BaseImgGPU<FT>& dImage, unsigned int gltexid)
{
	return cuda_to_gl<FT>(dImage.data(), dImage.pitch(), dImage.width(), dImage.height(), dImage.channel(), gltexid);
}

template <typename FT>
bool GLInterop::gl_to_cuda(unsigned int gltexid, BaseImgGPU<FT>& dImage)
{
	return gl_to_cuda<FT>(gltexid, dImage.data(), dImage.pitch(), dImage.width(), dImage.height(), dImage.channel());
}

template <typename FT>
bool GLInterop::cuda_to_gl(const BaseImgGPU<FT>& srcimg, GLImage<FT>& dstimg)
{
	if (!srcimg.data())
	{
		return false;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	return cuda_to_gl<FT>(srcimg, dstimg.data());
}

template <typename FT>
bool GLInterop::gl_to_cuda(const GLImage<FT>& srcimg, BaseImgGPU<FT>& dstimg)
{
	if (!srcimg.data())
	{
		return false;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	return gl_to_cuda<FT>(srcimg.data(), dstimg);
}

template <typename FT>
bool GLInterop::cpu_to_gl(const BaseImgCPU<FT>& srcimg, GLImage<FT>& dstimg)
{
	if (!srcimg.data())
	{
		return false;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	glBindTexture(GL_TEXTURE_2D, dstimg.data());

	if (typeid(FT) == typeid(unsigned char))
	{
		switch (c)
		{
		case 1:
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, srcimg.data());
			break;
		case 2:
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RG, GL_UNSIGNED_BYTE, srcimg.data());
			break;
		case 3:
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, srcimg.data());
			break;
		case 4:
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, srcimg.data());
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
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, srcimg.data());
			break;
		case 2:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, w, h, 0, GL_RG, GL_FLOAT, srcimg.data());
			break;
		case 3:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, srcimg.data());
			break;
		case 4:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, srcimg.data());
			break;
		default:
			break;
		}
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	CHECKGLERRORS;

	return true;
}

template <typename FT>
bool GLInterop::gl_to_cpu(const GLImage<FT>& srcimg, BaseImgCPU<FT>& dstimg)
{
	if (!srcimg.data())
	{
		return false;
	}

	int w = srcimg.width();
	int h = srcimg.height();
	int c = srcimg.channel();

	if (w != dstimg.width() || h != dstimg.height() || c != dstimg.channel())
	{
		dstimg.init(w, h, c);
	}

	glBindTexture(GL_TEXTURE_2D, srcimg.data());

	if (typeid(FT) == typeid(unsigned char))
	{
		switch (c)
		{
		case 1:
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, dstimg.data());
			break;
		case 2:
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, dstimg.data());;
			break;
		case 3:
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, dstimg.data());
			break;
		case 4:
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, dstimg.data());
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
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, dstimg.data());
			break;
		case 2:
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, dstimg.data());;
			break;
		case 3:
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, dstimg.data());
			break;
		case 4:
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, dstimg.data());
			break;
		default:
			break;
		}
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	CHECKGLERRORS;

	return true;
}

#endif