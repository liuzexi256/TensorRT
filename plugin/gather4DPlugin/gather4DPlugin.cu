#include <algorithm>
#include <cuda_fp16.h>

#include "gather4DPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::Gather4DPlugin;

///// Gather4D Enqueue start
template <typename Data>
__global__
void pillar_map_kernel(
            int batchSize,
            Data const* idata1,
            Data const* idata2,
            Data*       odata,
            int nDims,
            int nChans,
            int _size_h,
            int _size_w
            )
{
  Data const *feat_array = &idata1[nChans * _size_h * _size_w * blockIdx.z];
  Data const *index_array = &idata2[nDims * blockIdx.z];
  Data *output_array = &odata[nDims * nChans * blockIdx.z];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int index = tid; index < nDims; index += stride)
  {
    for (int c = 0; c < nChans; c++)
    {
      int output_index = index * nChans + c;
      int x = (int)index_array[index * 4 + 2];
      int y = (int)index_array[index * 4 + 3];
      int feat_index = c * _size_h * _size_w + x * _size_w + y;
      output_array[output_index] = feat_array[feat_index];
    }
  }
}

int Gather4DPlugin::enqueue(
     int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
 {
  //std::cout << "start enqueue gather 4d" << std::endl;

  int in_feature_dims = 10000;
  int in_channel = 96;
  int _size_h = 96;
  int _size_w = 96;
  const dim3 phnetDim3(512, 1, batchSize);

  // if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
  if (1)
  {
    cudaMemsetAsync(outputs[0], 0, sizeof(float) * in_feature_dims * in_channel, stream);
    pillar_map_kernel<<<2, phnetDim3, 0, stream>>>
    (
        batchSize,
        static_cast<float const *>(inputs[0]),
        static_cast<float const *>(inputs[1]),
        static_cast<float *>(outputs[0]),
        in_feature_dims,
        in_channel,
        _size_h,
        _size_w
    );
  }
  else
  {
    cudaMemsetAsync(outputs[0], 0, sizeof(__half) * in_feature_dims * in_channel, stream);
    pillar_map_kernel<<<2, phnetDim3, 0, stream>>>
    (
      batchSize,
      static_cast<__half const *>(inputs[0]),
      static_cast<__half const *>(inputs[1]),
      static_cast<__half *>(outputs[0]),
      in_feature_dims,
      in_channel,
      _size_h,
      _size_w
    );
  }
  return cudaGetLastError() != cudaSuccess;
}
///// Gather4D Enqueue end