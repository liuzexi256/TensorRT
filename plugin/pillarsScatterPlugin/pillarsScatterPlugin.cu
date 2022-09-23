#include <cuda_fp16.h>
#include "pillarsScatterPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::PillarsScatterPlugin;

template <typename Data>
__global__
void pillar_scatter_kernel(
            int batchSize,
            Data const *idata1,
            Data const *idata2,
            Data *odata,
            const int nDims,
            const int nChans,
            const int _size_h,
            const int _size_w)
{
  const int output_size = _size_h * _size_w * nChans;
  Data const *feat_array = &idata1[nDims * nChans * blockIdx.z];
  Data const *index_array = &idata2[nDims * 4 * blockIdx.z];
  Data *output_array = &odata[output_size * blockIdx.z];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int index = tid; index < nDims; index += stride)
  {
    for (int c = 0; c < nChans; c++)
    {
      int feature_index =  index * nChans + c;
      int x = (int)index_array[index * 4 + 2];
      int y = (int)index_array[index * 4 + 3];
      int odata_index = c * _size_h * _size_w + x * _size_w + y;
      output_array[odata_index] = feat_array[feature_index];
    }
  }
}

int PillarsScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                         const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
  // std::cout << "start enqueue pillars scatter" << std::endl;
  auto const& input0_dims = inputDesc[0].dims;

  const int in_feature_dims = input0_dims.d[1];
  const int in_channel = input0_dims.d[2];
  const int batchSize = inputDesc[0].dims.d[0];
  const dim3 phnetDim3(512, 1, batchSize);

  if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
  {
    cudaMemsetAsync(outputs[0], 0, sizeof(float) * in_feature_dims * in_channel, stream);
    pillar_scatter_kernel<<<2, phnetDim3, 0, stream>>>
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
    cudaMemsetAsync(outputs[0], 0xFF, sizeof(__half) * in_feature_dims * in_channel, stream);
    pillar_scatter_kernel<<<2, phnetDim3, 0, stream>>>
    (
      batchSize,
      static_cast<__half const *>( inputs[0]),
      static_cast<__half const *>( inputs[1]),
      static_cast<__half *>(outputs[0]),
      in_feature_dims,
      in_channel,
      _size_h,
      _size_w
    );
  }
  return cudaGetLastError() != cudaSuccess;
}