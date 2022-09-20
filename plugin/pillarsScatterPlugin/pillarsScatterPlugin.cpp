/*
 * @Author: Zexi Liu
 * @Date: 2022-07-29 11:45:47
 * @LastEditors: Zexi Liu
 * @LastEditTime: 2022-09-20 18:55:37
 * @FilePath: /TensorRT/plugin/pillarsScatterPlugin/pillarsScatterPlugin.cpp
 * @Description: 
 * 
 * Copyright (c) 2022 by Uisee, All Rights Reserved. 
 */

#include "pillarsScatterPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

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
    : w(w)
    , h(h)
{
}

PillarsScatterPlugin::PillarsScatterPlugin(void const* data, size_t length)
{
    deserialize_value(&data, &length, &h);
    deserialize_value(&data, &length, &w);
}

int PillarsScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int PillarsScatterPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void PillarsScatterPlugin::terminate() noexcept {}

Dims PillarsScatterPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    Dims output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[1];
    output.d[1] = h;
    output.d[2] = w;
    return output;
}

size_t PillarsScatterPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}

size_t PillarsScatterPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * 2;
}

void PillarsScatterPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, h);
    serialize_value(&buffer, w);
}

void PillarsScatterPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize) noexcept
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);
//   assert(mType == inputs[0].desc.type);
}

bool PillarsScatterPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kHALF || type == DataType::kFLOAT) && format == PluginFormat::kLINEAR);
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

IPluginV2Ext* PillarsScatterPlugin::clone() const noexcept
{
    PillarsScatterPlugin* plugin = new PillarsScatterPlugin(w, h);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

void PillarsScatterPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
    mPluginNamespace = "";
}

const char* PillarsScatterPlugin::getPluginNamespace() const noexcept
{
    std::cout << "mPluginNamespace: " << mPluginNamespace << std::endl;
    return mPluginNamespace;
}

nvinfer1::DataType PillarsScatterPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

bool PillarsScatterPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool PillarsScatterPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

PillarsScatterPluginCreator::PillarsScatterPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("w", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("h", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
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
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int w = 0;
    int h = 0;
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

    return new PillarsScatterPlugin(w, h);
}

IPluginV2Ext* PillarsScatterPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    PillarsScatterPlugin* plugin =  new PillarsScatterPlugin(data, length);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
