#pragma once
#include <assert.h>

#include <vector>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#include "Tensor.hpp"

class OpConfig {
private:
	std::string m_name;
	std::vector<std::shared_ptr<Tensor>> m_inputs;
	std::shared_ptr<Tensor> m_output;
	std::shared_ptr<Tensor> m_referenceOutput;
	std::vector<size_t> m_globalDims;
	std::vector<size_t> m_localDims;
	std::vector<size_t> m_outputDims;
	std::chrono::time_point<std::chrono::system_clock> m_executionTime;
public:
	OpConfig(json& j);
	~OpConfig();

	std::vector<size_t> getInputDims(int idx) { return m_inputs.at(idx)->getDims(); }
	std::vector<size_t> getOutputDims() { return m_outputDims; }
	std::vector<size_t> getGlobalDims() { return m_globalDims; }
	std::vector<size_t> getLocalDims() { return m_localDims; }

	std::shared_ptr<Tensor> getInputTensor(int idx);
	std::shared_ptr<Tensor> getOuptutTensor() { return m_output; }
	std::shared_ptr<Tensor> getRefOutputTensor() { return m_referenceOutput; }
	int inputListSize();
	int inputTensorSize(int idx);
	int refOutputTensorSize();

	void assignOutputData(float* data) { m_output->assignData(data); }
};