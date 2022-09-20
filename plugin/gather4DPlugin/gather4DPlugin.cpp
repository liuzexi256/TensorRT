/*
 * @Author: Zexi Liu
 * @Date: 2022-07-29 11:45:47
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-09-20 17:08:32
 * @FilePath: /TensorRT/plugin/gather4DPlugin/gather4DPlugin.cpp
 * @Description: 
 * 
 * Copyright (c) 2022 by Uisee, All Rights Reserved. 
 */

#include "gather4DPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

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

//REGISTER_TENSORRT_PLUGIN(Gather4DPluginCreator);

Gather4DPlugin::Gather4DPlugin()
{
}

Gather4DPlugin::Gather4DPlugin(void const* data, size_t length)
{
}

int Gather4DPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int Gather4DPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void Gather4DPlugin::terminate() noexcept {}

Dims Gather4DPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    Dims output;
    output.nbDims = 2;
    output.d[0] = inputs[1].d[0];
    output.d[1] = inputs[0].d[0];
    return output;
}

size_t Gather4DPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}

size_t Gather4DPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void Gather4DPlugin::serialize(void* buffer) const noexcept
{
}

void Gather4DPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize) noexcept
{
}

bool Gather4DPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kHALF || type == DataType::kFLOAT) && format == PluginFormat::kLINEAR);
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

IPluginV2Ext* Gather4DPlugin::clone() const noexcept
{
    Gather4DPlugin* plugin = new Gather4DPlugin();
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

void Gather4DPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
    mPluginNamespace = "";
}

const char* Gather4DPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

nvinfer1::DataType Gather4DPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

bool Gather4DPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool Gather4DPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
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
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    return new Gather4DPlugin();
}

IPluginV2Ext* Gather4DPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    Gather4DPlugin* plugin =  new Gather4DPlugin(data, length);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
