/*
 * @Author: Zexi Liu
 * @Date: 2022-07-29 11:45:47
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-08-16 18:38:42
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
{

}

void Gather4DPlugin::deserialize(void const* serialData, size_t serialLength) noexcept
{

}

Gather4DPlugin::Gather4DPlugin(void const* serialData, size_t serialLength)
{
    this->deserialize(serialData, serialLength);
}

size_t Gather4DPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void Gather4DPlugin::serialize(void *buffer) const noexcept
{

}

int Gather4DPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs Gather4DPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
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
