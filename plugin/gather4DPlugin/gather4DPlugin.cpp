/*
 * @Author: Zexi Liu
 * @Date: 2022-07-29 11:45:47
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-08-10 13:47:50
 * @FilePath: /TensorRT/plugin/gather4DPlugin/gather4DPlugin.cpp
 * @Description: 
 * 
 * Copyright (c) 2022 by Uisee, All Rights Reserved. 
 */

#include "gather4DPlugin.h"
#include "half.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::Gather4DPlugin;
using nvinfer1::plugin::Gather4DPluginCreator;

namespace
{
const char* GATHER_4D_PLUGIN_VERSION{"1"};
const char* GATHER_4D_PLUGIN_NAME{"gather_4d"};
} // namespace

PluginFieldCollection Gather4DPluginCreator::mFC{};
std::vector<PluginField> Gather4DPluginCreator::mPluginAttributes;

//REGISTER_TENSORRT_PLUGIN(PillarsScatterPluginCreator);

Gather4DPlugin::Gather4DPlugin()
    // : _size_w(w)
    // , _size_h(h)
{

}

void Gather4DPlugin::deserialize(void const* serialData, size_t serialLength) noexcept
{
    // deserialize_value(&serialData, &serialLength, &_input_dims);
    // deserialize_value(&serialData, &serialLength, &_max_batch_size);
    // deserialize_value(&serialData, &serialLength, &_data_type);
    // deserialize_value(&serialData, &serialLength, &_data_format);
    //deserialize_value(&serialData, &serialLength, &_op_type);
    // deserialize_value(&serialData, &serialLength, &_size_w);
    // deserialize_value(&serialData, &serialLength, &_size_h);

}

Gather4DPlugin::Gather4DPlugin(void const* serialData, size_t serialLength)
{
    this->deserialize(serialData, serialLength);
}

size_t Gather4DPlugin::getSerializationSize() const noexcept
{
    //size_t ret_size = serialized_size(_op_type);
                    // + serialized_size(_input_dims) 
                    // + serialized_size(_max_batch_size) 
                    // + serialized_size(_data_type) 
                    // + serialized_size(_data_format);

    //ret_size = ret_size + serialized_size(_size_w);
    // size_t ret_size = serialized_size(_size_w) + serialized_size(_size_h);
    // return ret_size;
}

void Gather4DPlugin::serialize(void *buffer) const noexcept
{
    // serialize_value(&buffer, _input_dims);
    // serialize_value(&buffer, _max_batch_size);
    // serialize_value(&buffer, _data_type);
    // serialize_value(&buffer, _data_format);

    //serialize_value(&buffer, (int)_op_type);
    // serialize_value(&buffer, (int)_size_w);
    // serialize_value(&buffer, (int)_size_h);
}

int Gather4DPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs Gather4DPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // nvinfer1::DimsExprs output;

    // nvinfer1::DimsExprs const& input = inputs[0];
    // output.nbDims = 2;
    // output.d[0] = _size_w;
    // output.d[1] = input.d[1];
    // nvinfer1::DimsExprs output(inputs[0]);
    DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[1].d[1];
    output.d[2] = inputs[0].d[1];
    return output;
}

int Gather4DPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void Gather4DPlugin::terminate() noexcept {}

size_t Gather4DPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void Gather4DPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
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

bool Gather4DPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

const char* Gather4DPlugin::getPluginType() const noexcept
{
    return GATHER_4D_PLUGIN_NAME;
}

const char* Gather4DPlugin::getPluginVersion() const noexcept
{
    return GATHER_4D_PLUGIN_VERSION;
}

void Gather4DPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* Gather4DPlugin::clone() const noexcept
{
    auto* plugin = new Gather4DPlugin();
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

nvinfer1::DataType Gather4DPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

void Gather4DPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* Gather4DPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

Gather4DPluginCreator::Gather4DPluginCreator()
{
    mFC.nbFields = 0;
}

const char* Gather4DPluginCreator::getPluginName() const noexcept
{
    return GATHER_4D_PLUGIN_NAME;
}

const char* Gather4DPluginCreator::getPluginVersion() const noexcept
{
    return GATHER_4D_PLUGIN_VERSION;
}

const PluginFieldCollection* Gather4DPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* Gather4DPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;

        Gather4DPlugin* plugin = new Gather4DPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* Gather4DPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    Gather4DPlugin* plugin = new Gather4DPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
