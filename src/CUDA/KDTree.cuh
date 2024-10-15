#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace Algorithm
{
	class KDTreeNode
	{
	public:
		KDTreeNode();
		~KDTreeNode();

	private:

	};

	class KDTree
	{
	public:
		KDTree();
		~KDTree();

	private:

	};
}

namespace CUDA
{
	void Test();

	std::vector<Eigen::Vector3f> DoFilter(Eigen::Vector3f* points);

	std::vector<Eigen::Vector3f> BitonicSort(Eigen::Vector3f* points, int numberOfPoints);
}