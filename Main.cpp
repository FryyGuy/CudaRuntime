#include <iostream>
#include <fstream>

#include "GrayscaleKernel.hpp"

int main(int argc, char** argv) {

	/*
	  Procedure
	  1) Arg[1] is a file path to a kernel_config.json file.
	  Load the contents of that into a json object to be used
	  for kernel configuration construction.
	  2) Create a CudaKernel object based on the config file.
	     - CudaKernel is a base class, so we should have a number of
		 SpecificKernel classes deidcated for each kernel, where execute()
		 is a pure virtual function to be overriden.
		 - We execute that kernel, and store output inside of the configuration.
		 We do this so we can compare against already stored output later.
	  3) Write output data to an image file if we want to, or just as a file.
	  4) Display exeuction speed of Kernel vs. Tensorflow.
	*/

	std::string configFilePath = argv[1];
	std::ifstream file(configFilePath);
	json j = json::parse(file);

	std::cout << j << std::endl;

	std::unique_ptr<CudaKernel> kernel;
	if (j["name"] == "rgb_to_grayscale") {
		kernel = std::unique_ptr<GrayscaleKernel>(new GrayscaleKernel(j));
	} else {
		std::cout << "Main:: Kernel not found. Aborting..." << std::endl;
		exit(1);
	}

	// exeucte the kernel function...
	kernel->execute();

	// compare output against reference output...
	kernel->compareOutput();

	//write output
	std::string outputPath = "C:\\Users\\fryma\\source\\repos\\GPUKernelRuntime\\CudaRuntime\\Output";
	std::string outputFileName = "";
	if (j["name"] == "rgb_to_grayscale") {
		outputFileName = "rgb_to_grayscale_output.raw";
	} else {
		std::cout << "Main:: kernel not found, aborting..." << std::endl;
		exit(1);
	}

	kernel->writeOutput(outputPath, outputFileName, false);
}