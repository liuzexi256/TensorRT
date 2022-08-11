#ifndef TRT_GATHER_D_PLUGIN_H
#define TRT_GATHER_D_PLUGIN_H

#include "NvInferPlugin.h"
#include "serialize.hpp"
#include "cudnn.h"
#include "plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Gather4DPlugin : public IPluginV2DynamicExt
{
public:

    const char* mPluginNamespace;

    Gather4DPlugin();

    Gather4DPlugin(const void* serialData, size_t serialLength);

    ~Gather4DPlugin() override = default;

    void deserialize(const void* data, size_t length) noexcept;
    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    int getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
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
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GATHER_4D_PLUGIN_H