#pragma once
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "OpConfig.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using json = nlohmann::json;

class CudaKernel {
private:
	std::unique_ptr<OpConfig> m_opConfig;
	std::vector<float*> m_inputs;
	float* m_output;

	void allocateGpuBuffers();
public:
	CudaKernel(json& j);
	~CudaKernel();

	float* getInput(int idx) { return m_inputs.at(idx); }
	float* getOutput() { return m_output; }

	std::vector<size_t> getInputDims(int idx) { return m_opConfig->getInputDims(idx); }
	std::vector<size_t> getOutputDims() { return m_opConfig->getOutputDims(); }

	void assignOutputData(float* output) { m_opConfig->assignOutputData(output); }

	void storeOutput();
	void compareOutput();
	void writeOutput(std::string filePath, std::string name, bool imageType);

	// execute() should be pure virtual because only derived
	// kernel classes should be implementing the function.
	// The thinking is that each kernel will end up having
	// a different process for their execution.
	// We want to be able call kernel->execute() in main()
	// and have zero kernel dependencies present in main.cpp
	virtual void execute()=0;
};
