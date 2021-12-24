#pragma once
#include "CudaKernel.hpp"

class GrayscaleKernel : public CudaKernel {
public:
	GrayscaleKernel(json& j) : CudaKernel(j) {};
	~GrayscaleKernel();

	void execute();
};