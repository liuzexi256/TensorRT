/*
 * @Author: Zexi Liu
 * @Date: 2022-08-11 13:49:31
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-09-20 17:07:43
 * @FilePath: /TensorRT/plugin/gather4DPlugin/gather4DPlugin.h
 * @Description: 
 * 
 * Copyright (c) 2022 by Uisee, All Rights Reserved. 
 */
#ifndef TRT_GATHER_D_PLUGIN_H
#define TRT_GATHER_D_PLUGIN_H

#include "NvInferPlugin.h"
#include "serialize.hpp"
#include "kernel.h"
#include "plugin.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Gather4DPlugin : public IPluginV2Ext
{
public:
    Gather4DPlugin();

    Gather4DPlugin(const void* data, size_t length);

    ~Gather4DPlugin() override = default;

    int getNbOutputs() const noexcept override; 
    
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

private:
    const char* mPluginNamespace;
    std::string mNameSpace;
};

class Gather4DPluginCreator : public BaseCreator
{
public:
    Gather4DPluginCreator();

    ~Gather4DPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion()const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;

protected:
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GATHER_4D_PLUGIN_H