#include "CudaKernel.hpp"

CudaKernel::CudaKernel(json& j) {
	m_opConfig = std::unique_ptr<OpConfig>(new OpConfig(j));
	allocateGpuBuffers();
}

CudaKernel::~CudaKernel() {
    for (int i = 0; i < m_inputs.size(); i++) {
        cudaFree(m_inputs.at(i));
    }

    cudaFree(m_output);
}

void CudaKernel::allocateGpuBuffers() {
    cudaError_t cudaStatus;
	for (int i = 0; i < m_opConfig->inputListSize(); i++) {
		float* gpuInput = 0;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            exit(1);
        }

        // Allocate GPU buffers for input vector.
        cudaStatus = cudaMalloc((void**)&gpuInput, m_opConfig->inputTensorSize(i) * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            exit(1);
        }

        // Copy input vectors from host memory to GPU buffers.
        auto tensor = m_opConfig->getInputTensor(i);
        cudaStatus = cudaMemcpy(gpuInput, tensor->getPointerToInput(), tensor->size() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            exit(1);
        }

        m_inputs.push_back(gpuInput);
	}

    float* gpuOutput;

    // Allocate GPU buffers for output vector.
    cudaStatus = cudaMalloc((void**)&gpuOutput, m_opConfig->refOutputTensorSize() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(1);
    }

    m_output = gpuOutput;
}

void CudaKernel::storeOutput() {
    cudaError_t cudaStatus;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        exit(1);
    }

    // Copy output vector from GPU buffer to host memory.
    // TODO: The Tensor/OpConfig class may need to be re-thought with this code in mind.
    // We may want to have Tensor::m_input be a straight float* to make this piece of code
    // easier to work with. Currently, my thinking is that we'd have to create a temp pointer
    // to store the gpu data and then copy that data into a vector later.
    auto outputDims = getOutputDims();
    //auto outputTensor = m_opConfig->getOuptutTensor();
    auto size = outputDims[0] * outputDims[1] * outputDims[2];
    float* gpuResult = (float*)malloc(size * sizeof(float));
    cudaStatus = cudaMemcpy(gpuResult, m_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        exit(1);
    }

    // copy gpuResult data into the m_opConfig::m_output Tensor...
    assignOutputData(gpuResult);
    free(gpuResult);
}

void CudaKernel::compareOutput() {
    auto outputTensor = m_opConfig->getOuptutTensor();
    auto refOutputTensor = m_opConfig->getRefOutputTensor();

    assert(outputTensor->size() == refOutputTensor->size());
    assert(outputTensor->getDims() == refOutputTensor->getDims());

    auto outputData = outputTensor->getPointerToInput();
    auto refOutputData = refOutputTensor->getPointerToInput();
    
    float epsilon = 0.3;

    // create an incorrect map for int (idx) : float* (list of inocrrect floats based on epsilon comparison)
    std::map<int, float*> incorrectValueMap;
    for (int i = 0; i < refOutputTensor->size(); i++) {
        auto outElem = (int)outputData[i];
        auto refOutElem = (int)refOutputData[i];

        bool approxEqual = fabs(outElem - refOutElem) <= ((fabs(outElem) < fabs(refOutElem) ? fabs(refOutElem) : fabs(outElem)) * epsilon);
        if (!approxEqual) {
            float incorrectValues[2];
            incorrectValues[0] = outElem;
            incorrectValues[1] = refOutElem;
            incorrectValueMap.insert(std::pair<int, float*>(i, incorrectValues));
        }
    }

    if (incorrectValueMap.empty()) {
        std::cout << "All values are correct!" << std::endl;
    } else {
        std::map<int, float*>::iterator itr;
        for (itr = incorrectValueMap.begin(); itr != incorrectValueMap.end(); itr++) {
            std::cout << "Incorrect value found at position " << itr->first << ": " << itr->second[0] << " != " << itr->second[1] << "..." << std::endl;
        }
    }
}

void CudaKernel::writeOutput(std::string filePath, std::string name, bool imageType) {
    std::string fullPath = filePath + "\\" + name;
    auto outputDims = getOutputDims();
    auto size = outputDims[0] * outputDims[1] * outputDims[2];

    if (imageType) {
        bool multiChannel = outputDims[2] == 1 ? false : true;
        auto imageFormat = multiChannel ? CV_32FC3 : CV_32FC1;

        // TODO: Not working properly, access violation of some sort. 
        //auto output = m_output;
        //auto mat = cv::Mat(outputDims[0], outputDims[1], imageFormat, output);
        //cv::imshow("Result", mat);

    } else {
        std::ofstream outFile(fullPath, std::ios_base::binary);
        outFile.write((char*)m_output, size * sizeof(float));
    }
}