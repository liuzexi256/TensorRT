#include <cuda_fp16.h>
#include "scatterMaxPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::ScatterMaxPlugin;

__device__ __forceinline__
float atomicMaxFloat(float * addr, float value)
{
  float old = (value >= 0) ?
      __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
      __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

template <typename Data>
__global__
void scatter_max_kernel_f16(
            int batchSize,
            Data const* idata1,
            const int nDims,
            Data const* idata2,
            const int nChans,
            Data* odata)
{
  Data const *feat_array = &idata1[nDims * nChans * blockIdx.z];
  Data const *index_array = &idata2[nDims * blockIdx.z];
  Data *scatter_array = &odata[nDims * nChans * blockIdx.z];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int index = tid; index < nDims; index += stride)
  {
      for (int c = 0; c < nChans; c++)
      {
        int feat_idx = index * nChans + c;
        int scatter_idx = (int)index_array[index] * nChans + c;
        Data feat = feat_array[feat_idx];
        if (__hgt(feat, scatter_array[scatter_idx]))
        {
          scatter_array[scatter_idx] = feat;
        }
      }
  }
}

template <typename Data>
__global__
void scatter_max_kernel_f32(
            int batchSize,
            Data const* idata1,
            const int nDims,
            Data const* idata2,
            const int nChans,
            Data* odata)
{
  Data const *feat_array = &idata1[nDims * nChans * blockIdx.z];
  Data const *index_array = &idata2[nDims * blockIdx.z];
  Data *scatter_array = &odata[nDims * nChans * blockIdx.z];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int index = tid; index < nDims; index += stride)
  {
      for (int c = 0; c < nChans; c++)
      {
        int feat_idx = index * nChans + c;
        int scatter_idx = (int)index_array[index] * nChans + c;
        Data feat = feat_array[feat_idx];
        atomicMaxFloat(&scatter_array[scatter_idx], feat);
      }
  }
}

int ScatterMaxPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                         const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
  // std::cout << "start enqueue scatter max" << std::endl;
  auto const& input0_dims = inputDesc[0].dims;

  const int nChans = input0_dims.d[2];
  const int nDims = input0_dims.d[1];
  const int batchSize = input0_dims.d[0];
  const dim3 phnetDim3(512, 1, batchSize);

  if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
  {
    cudaMemsetAsync(outputs[0], 0xFF, sizeof(float) * nChans * _size_w, stream);
    scatter_max_kernel_f32<<<2, phnetDim3, 0, stream>>>
    (
      batchSize,
      static_cast<float const *>(inputs[0]),
      nDims,
      static_cast<float const *>(inputs[1]),
      nChans,
      static_cast<float *>(outputs[0])
    );
  }
  else
  {
    cudaMemsetAsync(outputs[0], 0xFF, sizeof(__half) * nChans * _size_w, stream);
    scatter_max_kernel_f16<<<2, phnetDim3, 0, stream>>>
    (
      batchSize,
      static_cast<__half const *>(inputs[0]),
      nDims,
      static_cast<__half const *>(inputs[1]),
      nChans,
      static_cast<__half *>(outputs[0])
    );
  }
  return cudaGetLastError() != cudaSuccess;
}