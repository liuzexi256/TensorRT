/*
 * @Author: Zexi Liu
 * @Date: 2022-07-29 11:45:47
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-09-20 17:06:18
 * @FilePath: /TensorRT/plugin/scatterMaxPlugin/scatterMaxPlugin.cpp
 * @Description: 
 * 
 * Copyright (c) 2022 by Uisee, All Rights Reserved. 
 */

#include "scatterMaxPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

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

ScatterMaxPlugin::ScatterMaxPlugin() {}

ScatterMaxPlugin::ScatterMaxPlugin(int w, int nChans, int nDims)
    : w(w)
    , nChans(nChans)
    , nDims(nDims)
{
}

ScatterMaxPlugin::ScatterMaxPlugin(void const* data, size_t length)
{
    deserialize_value(&data, &length, &w);
    deserialize_value(&data, &length, &nChans);
    deserialize_value(&data, &length, &nDims);
}

int ScatterMaxPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int ScatterMaxPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void ScatterMaxPlugin::terminate() noexcept {}

Dims ScatterMaxPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    Dims output;
    output.nbDims = 2;
    output.d[0] = w;
    output.d[1] = inputs[0].d[1];
    return output;
}

size_t ScatterMaxPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}



size_t ScatterMaxPlugin::getSerializationSize() const noexcept
{
    return 3 * sizeof(int);
}

void ScatterMaxPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, w);
    serialize_value(&buffer, nChans);
    serialize_value(&buffer, nDims);
}

void ScatterMaxPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize) noexcept
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);

//   assert(mType == inputs[0].desc.type);
}

bool ScatterMaxPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kHALF || type == DataType::kFLOAT) && format == PluginFormat::kLINEAR);
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

IPluginV2Ext* ScatterMaxPlugin::clone() const noexcept
{
    ScatterMaxPlugin* plugin = new ScatterMaxPlugin(w, nChans, nDims);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

void ScatterMaxPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
    mPluginNamespace = "";
}

const char* ScatterMaxPlugin::getPluginNamespace() const noexcept
{
    std::cout << "mPluginNamespace: " << mPluginNamespace << std::endl;
    return mPluginNamespace;
}

nvinfer1::DataType ScatterMaxPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

bool ScatterMaxPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool ScatterMaxPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

ScatterMaxPluginCreator::ScatterMaxPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("w", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
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
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int w = 0;
    int nChans = 0;
    int nDims = 0;
    for (int i = 0; i < nbFields; ++i)
    {
        if (!strcmp(fields[i].name, "w"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            w = *(static_cast<const int*>(fields[i].data));
            std::cout << "w: " << w << std::endl;
        }
        if (!strcmp(fields[i].name, "nChans"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            nChans = *(static_cast<const int*>(fields[i].data));
            std::cout << "nChans: " << nChans << std::endl;
        }
        if (!strcmp(fields[i].name, "nDims"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            nDims = *(static_cast<const int*>(fields[i].data));
            std::cout << "nDims: " << nDims << std::endl;
        }
    }
    return new ScatterMaxPlugin(w, nChans, nDims);
}

IPluginV2Ext* ScatterMaxPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    ScatterMaxPlugin* plugin =  new ScatterMaxPlugin(data, length);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
