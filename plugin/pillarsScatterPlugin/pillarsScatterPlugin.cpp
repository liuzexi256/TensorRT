/*
 * @Author: Zexi Liu
 * @Date: 2022-07-29 11:45:47
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-08-08 17:14:04
 * @FilePath: /TensorRT/plugin/pillarsScatterPlugin/pillarsScatterPlugin.cpp
 * @Description: 
 * 
 * Copyright (c) 2022 by Uisee, All Rights Reserved. 
 */

#include "pillarsScatterPlugin.h"
#include "half.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::PillarsScatterPlugin;
using nvinfer1::plugin::PillarsScatterPluginCreator;

namespace
{
const char* PILLARS_SCATTER_PLUGIN_VERSION{"1"};
const char* PILLARS_SCATTER_PLUGIN_NAME{"pillars_scatter"};
} // namespace

PluginFieldCollection PillarsScatterPluginCreator::mFC{};
std::vector<PluginField> PillarsScatterPluginCreator::mPluginAttributes;

//REGISTER_TENSORRT_PLUGIN(PillarsScatterPluginCreator);

PillarsScatterPlugin::PillarsScatterPlugin(int w, int h)
    : _size_w(w)
    , _size_h(h)
{

}

void PillarsScatterPlugin::deserialize(void const* serialData, size_t serialLength) noexcept
{
    // deserialize_value(&serialData, &serialLength, &_input_dims);
    // deserialize_value(&serialData, &serialLength, &_max_batch_size);
    // deserialize_value(&serialData, &serialLength, &_data_type);
    // deserialize_value(&serialData, &serialLength, &_data_format);
    //deserialize_value(&serialData, &serialLength, &_op_type);
    deserialize_value(&serialData, &serialLength, &_size_w);
    deserialize_value(&serialData, &serialLength, &_size_h);

}

PillarsScatterPlugin::PillarsScatterPlugin(void const* serialData, size_t serialLength)
{
    this->deserialize(serialData, serialLength);
}

size_t PillarsScatterPlugin::getSerializationSize() const noexcept
{
    //size_t ret_size = serialized_size(_op_type);
                    // + serialized_size(_input_dims) 
                    // + serialized_size(_max_batch_size) 
                    // + serialized_size(_data_type) 
                    // + serialized_size(_data_format);

    //ret_size = ret_size + serialized_size(_size_w);
    size_t ret_size = serialized_size(_size_w) + serialized_size(_size_h);
    return ret_size;
}

void PillarsScatterPlugin::serialize(void *buffer) const noexcept
{
    // serialize_value(&buffer, _input_dims);
    // serialize_value(&buffer, _max_batch_size);
    // serialize_value(&buffer, _data_type);
    // serialize_value(&buffer, _data_format);

    //serialize_value(&buffer, (int)_op_type);
    serialize_value(&buffer, (int)_size_w);
    serialize_value(&buffer, (int)_size_h);
}

int PillarsScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs PillarsScatterPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // nvinfer1::DimsExprs output;

    // nvinfer1::DimsExprs const& input = inputs[0];
    // output.nbDims = 2;
    // output.d[0] = _size_w;
    // output.d[1] = input.d[1];
    // nvinfer1::DimsExprs output(inputs[0]);
    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[2];
    output.d[2] = exprBuilder.constant(_size_h);
    output.d[3] = exprBuilder.constant(_size_w);
    return output;
}

int PillarsScatterPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void PillarsScatterPlugin::terminate() noexcept {}

size_t PillarsScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void PillarsScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
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

bool PillarsScatterPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

const char* PillarsScatterPlugin::getPluginType() const noexcept
{
    return PILLARS_SCATTER_PLUGIN_NAME;
}

const char* PillarsScatterPlugin::getPluginVersion() const noexcept
{
    return PILLARS_SCATTER_PLUGIN_VERSION;
}

void PillarsScatterPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* PillarsScatterPlugin::clone() const noexcept
{
    auto* plugin = new PillarsScatterPlugin(_size_w, _size_h);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

nvinfer1::DataType PillarsScatterPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

void PillarsScatterPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* PillarsScatterPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

PillarsScatterPluginCreator::PillarsScatterPluginCreator()
{
    mFC.nbFields = 0;
}

const char* PillarsScatterPluginCreator::getPluginName() const noexcept
{
    return PILLARS_SCATTER_PLUGIN_NAME;
}

const char* PillarsScatterPluginCreator::getPluginVersion() const noexcept
{
    return PILLARS_SCATTER_PLUGIN_VERSION;
}

const PluginFieldCollection* PillarsScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* PillarsScatterPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;

        for (int i = 0; i < nbFields; ++i)
        {
            if (!strcmp(fields[i].name, "_size_w"))
            {
                w = *(reinterpret_cast<const int*>(fields[i].data));
                std::cout << "w_pillar: " << w << std::endl;
            }
            if (!strcmp(fields[i].name, "_size_h"))
            {
                h = *(reinterpret_cast<const int*>(fields[i].data));
                std::cout << "h_pillar: " << h << std::endl;
            }
        }
        PillarsScatterPlugin* plugin = new PillarsScatterPlugin(w, h);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* PillarsScatterPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    PillarsScatterPlugin* plugin = new PillarsScatterPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
