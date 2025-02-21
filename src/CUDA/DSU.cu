#include "SVO.cuh"

#include <cufft.h>

#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
    namespace DSU
    {
#define BLOCK_SIZE 256

        void TestDSU()
        {
            auto t = Time::Now();

            PLYFormat ply;
            ply.Deserialize(ResourceIO::GetPath("3d/compound.ply").string());

            unsigned int numberOfPoints = ply.GetPoints().size() / 3;

            t = Time::End(t, "Load ply");

            for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
            {
                auto x = ply.GetPoints()[i * 3];
                auto y = ply.GetPoints()[i * 3 + 1];
                auto z = ply.GetPoints()[i * 3 + 2];

                VD::AddSphere("Points", { x,y,z }, 0.05f);
            }

            t = Time::End(t, "Visualize ply");
        }
    }
}
