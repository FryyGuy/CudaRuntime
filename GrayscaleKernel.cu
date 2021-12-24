#pragma once

#include "GrayscaleKernel.hpp"

#define R 0
#define G 1
#define B 2

GrayscaleKernel::~GrayscaleKernel() {

}

__global__ void rgb_to_grayscale(float* input, 
								 float* output) {

	// Data is laid out like [R, G, B, R, G, B, ...]
	// So start position is every third element...
	int inputIdx = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	float r = input[inputIdx + R];
	float g = input[inputIdx + G];
	float b = input[inputIdx + B];

	int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;

	output[outputIdx] = 0.2989 * r + 0.5870 * g + 0.1140 * b;
}

void GrayscaleKernel::execute() {
	cudaError_t cudaStatus;

	auto inputDims = CudaKernel::getInputDims(0);
	auto outputDims = CudaKernel::getOutputDims();

	auto arraySize = outputDims[0] * outputDims[1] * outputDims[2];

	auto input = CudaKernel::getInput(0);
	auto output = CudaKernel::getOutput();
	
	// TODO: store exeuction time in m_opConfig
	rgb_to_grayscale <<< arraySize / 256, 256 >>> (input, output);

	// Mandatory call in order to properly save output data
	CudaKernel::storeOutput();
}