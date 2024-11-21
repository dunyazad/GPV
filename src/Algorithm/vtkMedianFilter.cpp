#include <Algorithm/vtkMedianFilter.h>

vtkStandardNewMacro(vtkMedianFilter);

int vtkMedianFilter::RequestData(vtkInformation* request,
    vtkInformationVector** inputVector,
    vtkInformationVector* outputVector)
{
    vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
    vtkPolyData* input = vtkPolyData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkSmartPointer<vtkKdTreePointLocator> kdTree = vtkSmartPointer<vtkKdTreePointLocator>::New();
    kdTree->SetDataSet(input);
    kdTree->BuildLocator();

    auto points = input->GetPoints();
    auto nop = (size_t)input->GetNumberOfPoints();
    for (size_t i = 0; i < nop; i++)
    {
        auto point = points->GetPoint(i);
        vtkSmartPointer<vtkIdList> result = vtkSmartPointer<vtkIdList>::New();
        kdTree->FindPointsWithinRadius(0.5f, point, result);

        std::vector<double> distances;
        for (vtkIdType j = 0; j < result->GetNumberOfIds(); j++)
        {
            vtkIdType neighborId = result->GetId(j);
            double neighborPoint[3];
            points->GetPoint(neighborId, neighborPoint);

            double distance = sqrt(vtkMath::Distance2BetweenPoints(point, neighborPoint));
            distances.push_back(distance);
        }

        std::sort(distances.begin(), distances.end());
        double medianDistance = distances[distances.size() / 2];

        //std::cout << "Point ID: " << i << " Median Distance: " << medianDistance << std::endl;

        for (vtkIdType j = 0; j < result->GetNumberOfIds(); j++)
        {
            vtkIdType neighborId = result->GetId(j);
            double neighborPoint[3];
            points->GetPoint(neighborId, neighborPoint);

            double distance = sqrt(vtkMath::Distance2BetweenPoints(point, neighborPoint));

            //// If the distance exceeds the median, replace the point's coordinates with the query point's coordinates
            //if (distance > medianDistance)
            //{
            //    points->SetPoint(neighborId, point); // Replace the point exceeding the median
            //    //std::cout << "Replaced Point ID: " << neighborId << " Exceeds Median Distance." << std::endl;
            //}
        }
    }

    vtkInformation* outInfo = outputVector->GetInformationObject(0);
    vtkPolyData* output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    output->ShallowCopy(input);

    return 1;
}
