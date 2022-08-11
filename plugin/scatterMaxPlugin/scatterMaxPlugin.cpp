/*
 * @Author: Zexi Liu
 * @Date: 2022-07-29 11:45:47
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-08-10 15:03:38
 * @FilePath: /TensorRT/plugin/scatterMaxPlugin/scatterMaxPlugin.cpp
 * @Description: 
 * 
 * Copyright (c) 2022 by Uisee, All Rights Reserved. 
 */

#include "scatterMaxPlugin.h"
#include "half.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ScatterMaxPlugin;
using nvinfer1::plugin::ScatterMaxPluginCreator;

namespace
{
const char* SCATTER_MAX_PLUGIN_VERSION{"1"};
const char* SCATTER_MAX_PLUGIN_NAME{"scatter_max"};
} // namespace

PluginFieldCollection ScatterMaxPluginCreator::mFC{};
std::vector<PluginField> ScatterMaxPluginCreator::mPluginAttributes;

//REGISTER_TENSORRT_PLUGIN(ScatterMaxPluginCreator);

//ScatterMaxPlugin::ScatterMaxPlugin() {}

ScatterMaxPlugin::ScatterMaxPlugin(int w)
    : _size_w(w)
{

}

void ScatterMaxPlugin::deserialize(void const* serialData, size_t serialLength) noexcept
{
    // deserialize_value(&serialData, &serialLength, &_input_dims);
    // deserialize_value(&serialData, &serialLength, &_max_batch_size);
    // deserialize_value(&serialData, &serialLength, &_data_type);
    // deserialize_value(&serialData, &serialLength, &_data_format);
    //deserialize_value(&serialData, &serialLength, &_op_type);
    deserialize_value(&serialData, &serialLength, &_size_w);

}

ScatterMaxPlugin::ScatterMaxPlugin(void const* serialData, size_t serialLength)
{
    this->deserialize(serialData, serialLength);
    // const char *d = reinterpret_cast<const char*>(serialData), *a = d;
    // w = read<int>(d);
    // mInputDims = Dims3();
    // mInputDims.d[0] = read<int>(d);
    // mInputDims.d[1] = read<int>(d);
    // mInputDims.d[2] = read<int>(d);
    // mOutputDims = Dims3();
    // mOutputDims.d[0] = read<int>(d);
    // mOutputDims.d[1] = read<int>(d);
    // mOutputDims.d[2] = read<int>(d);
    // ASSERT(d == a + serialLength);
}

size_t ScatterMaxPlugin::getSerializationSize() const noexcept
{
    //size_t ret_size = serialized_size(_op_type);
                    // + serialized_size(_input_dims) 
                    // + serialized_size(_max_batch_size) 
                    // + serialized_size(_data_type) 
                    // + serialized_size(_data_format);

    //ret_size = ret_size + serialized_size(_size_w);
    size_t ret_size = serialized_size(_size_w);
    return ret_size;
}

void ScatterMaxPlugin::serialize(void *buffer) const noexcept
{
    // serialize_value(&buffer, _input_dims);
    // serialize_value(&buffer, _max_batch_size);
    // serialize_value(&buffer, _data_type);
    // serialize_value(&buffer, _data_format);

    //serialize_value(&buffer, (int)_op_type);
    serialize_value(&buffer, (int)_size_w);
}

int ScatterMaxPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs ScatterMaxPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    //std::cout << "wwwwwwwwww" << _size_w << std::endl;
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = exprBuilder.constant(_size_w);
    output.d[2] = inputs[0].d[2];
    return output;
}

int ScatterMaxPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void ScatterMaxPlugin::terminate() noexcept {}

size_t ScatterMaxPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void ScatterMaxPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
//   assert(nbOutputs == 1);
//   assert(nbInputs == 3);
//   assert(mType == inputs[0].desc.type);
}

// bool ScatterMaxPlugin::supportsFormat(DataType type,
//                                       PluginFormat format) const noexcept{
//   return (type == DataType::kFLOAT ||
//                   type == DataType::kHALF);
// }

bool ScatterMaxPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

const char* ScatterMaxPlugin::getPluginType() const noexcept
{
    return SCATTER_MAX_PLUGIN_NAME;
}

const char* ScatterMaxPlugin::getPluginVersion() const noexcept
{
    return SCATTER_MAX_PLUGIN_VERSION;
}

void ScatterMaxPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* ScatterMaxPlugin::clone() const noexcept
{
    auto* plugin = new ScatterMaxPlugin(_size_w);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

nvinfer1::DataType ScatterMaxPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

void ScatterMaxPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* ScatterMaxPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

ScatterMaxPluginCreator::ScatterMaxPluginCreator()
{
    mFC.nbFields = 0;
}

const char* ScatterMaxPluginCreator::getPluginName() const noexcept
{
    return SCATTER_MAX_PLUGIN_NAME;
}

const char* ScatterMaxPluginCreator::getPluginVersion() const noexcept
{
    return SCATTER_MAX_PLUGIN_VERSION;
}

const PluginFieldCollection* ScatterMaxPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* ScatterMaxPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;

        for (int i = 0; i < nbFields; ++i)
        {
            if (!strcmp(fields[i].name, "w"))
            {
                w = *(reinterpret_cast<const int*>(fields[i].data));
                std::cout << "w: " << w << std::endl;
            }
        }
        ScatterMaxPlugin* plugin = new ScatterMaxPlugin(w);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* ScatterMaxPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    ScatterMaxPlugin* plugin = new ScatterMaxPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
