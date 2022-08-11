#include <algorithm>
#include <cuda_fp16.h>

#include "gather4DPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::Gather4DPlugin;

__device__ __forceinline__
float atomicMaxFloat(float * addr, float value)
{
    float old = (value >= 0) ?
        __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}


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

int Gather4DPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                         const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
  std::cout << "start enqueue gather 4d" << std::endl;
  std::cout << "input0.shape0 = " << inputDesc[0].dims.d[0] << std::endl;
  std::cout << "input0.shape1 = " << inputDesc[0].dims.d[1] << std::endl;
  std::cout << "input0.shape2 = " << inputDesc[0].dims.d[2] << std::endl;
  std::cout << "input0.shape2 = " << inputDesc[0].dims.d[3] << std::endl;
  std::cout << "input1.shape0 = " << inputDesc[1].dims.d[0] << std::endl;
  std::cout << "input1.shape1 = " << inputDesc[1].dims.d[1] << std::endl;
  std::cout << "input1.shape2 = " << inputDesc[1].dims.d[2] << std::endl;
  std::cout << "input1.shape2 = " << inputDesc[1].dims.d[3] << std::endl;

  auto const& input0_dims = inputDesc[0].dims;
  auto const& input1_dims = inputDesc[1].dims;

  int in_feature_dims = input1_dims.d[1];
  int in_channel = input0_dims.d[1];
  int _size_h = input0_dims.d[2];
  int _size_w = input0_dims.d[3];
  const dim3 phnetDim3(512, 1, inputDesc[0].dims.d[0]);

  if (nvinfer1::DataType::kFLOAT == nvinfer1::DataType::kFLOAT)
  {
    cudaMemsetAsync(outputs[0], 0, sizeof(float) * in_feature_dims * in_channel, stream);
    pillar_map_kernel<<<2, phnetDim3, 0, stream>>>
    (
        inputDesc[0].dims.d[0],
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
      inputDesc[0].dims.d[0],
      static_cast<__half const *>(inputs[0]),
      static_cast<__half const *>(inputs[1]),
      static_cast<__half *>(outputs[0]),
      in_feature_dims,
      in_channel,
      _size_h,
      _size_w
    );
  }
  std::cout << "cuda error gather 4d: " << cudaGetLastError() << std::endl;
  std::cout << "end enqueue gather 4d" << std::endl;
  return cudaGetLastError() != cudaSuccess;
}
///// Gather4D Enqueue end