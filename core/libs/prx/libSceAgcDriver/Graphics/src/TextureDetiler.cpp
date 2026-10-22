#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/shaders/TextureDetile_spv.h"
#include "prx/libSceAgcDriver/Graphics/shaders/TextureRepack_spv.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

struct Push {
    std::uint32_t srcBase;
    std::uint32_t dstBase;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pitchBytes;
    std::uint32_t blocksPerRow;
    std::uint32_t tail;
    std::uint32_t tailX;
    std::uint32_t tailY;
    std::uint32_t elementBytes;
    std::uint32_t slice;
    std::uint32_t rangeBegin;
    std::uint32_t rangeEnd;
    std::uint32_t tiledBase;
    std::uint32_t linearBase;
    std::uint32_t columnBegin;
    std::uint32_t rowBegin;
};

std::uint32_t BlockBytesFor(TextureTileMode tileMode) {
    switch (tileMode) {
        case TextureTileMode::kLinear: return 0u;
        case TextureTileMode::kStandard256B:
        case TextureTileMode::kD256B: return 256u;
        case TextureTileMode::kStandard4KB:
        case TextureTileMode::kD4KB:
        case TextureTileMode::kS4KBX:
        case TextureTileMode::kD4KBX: return 4096u;
        case TextureTileMode::kStandard64KB:
        case TextureTileMode::kD64KB:
        case TextureTileMode::kS64KBT:
        case TextureTileMode::kD64KBT:
        case TextureTileMode::kZ64KBX:
        case TextureTileMode::kS64KBX:
        case TextureTileMode::kD64KBX:
        case TextureTileMode::kR64KBX: return 65536u;
    }
    throw std::runtime_error("AGC graphics: TextureDetiler encountered an unknown tile mode");
}

std::uint32_t PipelineKey(TextureTileMode tileMode, std::uint32_t elementBytes, bool retile, bool thick, bool imageIo) {
    return (imageIo ? 1u << 18 : 0u) | (thick ? 1u << 17 : 0u) | (retile ? 1u << 16 : 0u) | (static_cast<std::uint32_t>(tileMode) << 8) | elementBytes;
}

}

TextureDetiler::TextureDetiler(const Context& context) : context(context) {
    try {
        // Binding 2, the image of DispatchImage, is used by the image pipelines only.
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = index == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        // With push descriptors a dispatch writes its two buffers into the command buffer instead of
        // allocating and updating a set.
        if (context.pushDescriptors) layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &descriptorLayout), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(Push);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(TEXTURE_DETILE_SPV);
        moduleInfo.pCode = TEXTURE_DETILE_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
    } catch (...) {
        release();
        throw;
    }
}

TextureDetiler::~TextureDetiler() {
    release();
}

void TextureDetiler::release() noexcept {
    for (const auto pool : descriptorPools) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    for (const auto& entry : pipelines) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, entry.second, nullptr);
    if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
    if (repackPipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, repackPipeline, nullptr);
    if (repackModule) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, repackModule, nullptr);
    if (pipelineLayout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, pipelineLayout, nullptr);
    if (descriptorLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, descriptorLayout, nullptr);
}

VkPipeline TextureDetiler::pipeline(TextureTileMode tileMode, std::uint32_t elementBytes, bool retile, bool thick, bool imageIo) {
    Require(std::has_single_bit(elementBytes) && elementBytes <= 16u, "unsupported element size for texture detiling");
    const auto key = PipelineKey(tileMode, elementBytes, retile, thick, imageIo);
    for (const auto& entry : pipelines) {
        if (entry.first == key) return entry.second;
    }
    // Constants 0-3 select element size, block size, addressing family and direction; 4-19 carry the
    // per-bit XOR equation for the equation family (2); 20-21 override the block extent in elements.
    std::array<std::uint32_t, 23> values{elementBytes, BlockBytesFor(tileMode), tileMode == TextureTileMode::kLinear ? 0u : 1u, retile ? 1u : 0u};
    values[22] = imageIo ? 1u : 0u;
    if (thick) {
        const auto thickMode = tileMode == TextureTileMode::kStandard4KB ? 0x105u : tileMode == TextureTileMode::kStandard64KB ? 0x109u : tileMode == TextureTileMode::kS64KBX ? 0x119u : 0u;
        Require(thickMode != 0, "3D textures are only detiled thick from SW_4KB_S, SW_64KB_S or SW_64KB_S_X");
        const auto* equation = FindTextureSwizzleEquation(thickMode, elementBytes);
        Require(equation != nullptr, "no thick swizzle equation for the element size");
        values[2] = 2u;
        std::copy(equation->bits.begin(), equation->bits.end(), values.begin() + 4);
        const auto extent = ThickBlockExtent(tileMode, elementBytes);
        values[20] = extent[0];
        values[21] = extent[1];
    } else if (const auto mode = EquationSwizzleMode(tileMode); mode != 0) {
        const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
        if (equation == nullptr) Require(false, "no swizzle equation for tile mode " + std::to_string(mode) + " at " + std::to_string(elementBytes) + " bytes per element");
        values[2] = 2u;
        std::copy(equation->bits.begin(), equation->bits.end(), values.begin() + 4);
    }
    std::array<VkSpecializationMapEntry, 23> entries{};
    for (std::uint32_t index = 0; index < entries.size(); ++index) entries[index] = {index, index * 4u, 4};
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = static_cast<std::uint32_t>(entries.size());
    specialization.pMapEntries = entries.data();
    specialization.dataSize = sizeof(values);
    specialization.pData = values.data();
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    stage.pSpecializationInfo = &specialization;
    VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    createInfo.stage = stage;
    createInfo.layout = pipelineLayout;
    VkPipeline result = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, context.pipelineCache, 1, &createInfo, nullptr, &result), "vkCreateComputePipelines");
    pipelines.emplace_back(key, result);
    return result;
}

void TextureDetiler::Dispatch(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkBuffer source, std::uint64_t sourceOffset, VkBuffer destination, std::uint64_t destinationOffset, const TileMipLayout& layout, bool retile, std::uint32_t slice, bool thick, const DetileWindow& window) {
    Require(commands != VK_NULL_HANDLE, "texture detiling requires an active command buffer");
    Require(window.rangeBegin < window.rangeEnd && window.tiledBase <= window.rangeBegin, "texture detiling window is empty");
    Require(source != VK_NULL_HANDLE && destination != VK_NULL_HANDLE, "texture detiling requires source and destination buffers");
    Require(layout.width != 0 && layout.height != 0, "texture detiling requires a non-empty mip layout");
    Require(layout.tiledSize != 0 && layout.linearSize != 0, "texture detiling requires a non-empty mip layout");
    const auto columnEnd = window.columnEnd != 0 ? std::min(window.columnEnd, layout.width) : layout.width;
    const auto rowEnd = window.rowEnd != 0 ? std::min(window.rowEnd, layout.height) : layout.height;
    if (!(window.columnBegin < columnEnd && window.rowBegin < rowEnd)) Require(false, "texture detiling window lies outside the mip: columns " + std::to_string(window.columnBegin) + ".." + std::to_string(columnEnd) + ", rows " + std::to_string(window.rowBegin) + ".." + std::to_string(rowEnd) + " of " + std::to_string(layout.width) + "x" + std::to_string(layout.height));
    const auto target = pipeline(tileMode, elementBytes, retile, thick);
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto sourceDescriptorOffset = sourceOffset - sourceOffset % alignment;
    const auto destinationDescriptorOffset = destinationOffset - destinationOffset % alignment;
    const auto sourceBase = sourceOffset - sourceDescriptorOffset;
    const auto destinationBase = destinationOffset - destinationDescriptorOffset;
    Require(sourceBase <= UINT32_MAX && destinationBase <= UINT32_MAX, "texture detiling buffer offset exceeds addressable range");
    // A window covers the mip's bytes from its base to the range's end (tiled) or the last row the
    // range can touch (linear).
    const std::uint64_t tiledBytes = window.rangeEnd == 0xffffffffu ? layout.tiledSize : std::min<std::uint64_t>(layout.tiledSize, window.rangeEnd) - window.tiledBase;
    const std::uint64_t linearBytes = window.linearBytes != 0 ? window.linearBytes : layout.linearSize - std::min<std::uint64_t>(window.linearBase, layout.linearSize);
    const auto sourceRange = (sourceBase + (retile ? linearBytes : tiledBytes) + 3) / 4 * 4;
    const auto destinationRange = (destinationBase + (retile ? tiledBytes : linearBytes) + 3) / 4 * 4;
    Require(sourceRange <= context.limits.maxStorageBufferRange && destinationRange <= context.limits.maxStorageBufferRange, "texture detiling buffer range exceeds device limits");
    const auto set = context.pushDescriptors ? VK_NULL_HANDLE : allocateSet();
    const VkDescriptorBufferInfo sourceInfo{source, sourceDescriptorOffset, sourceRange};
    const VkDescriptorBufferInfo destinationInfo{destination, destinationDescriptorOffset, destinationRange};
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &sourceInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &destinationInfo;
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, target);
    if (context.pushDescriptors) {
        context.Function<PFN_vkCmdPushDescriptorSetKHR>("vkCmdPushDescriptorSetKHR")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    } else {
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
    }
    Push push{};
    push.srcBase = static_cast<std::uint32_t>(sourceBase);
    push.dstBase = static_cast<std::uint32_t>(destinationBase);
    push.width = layout.width;
    push.height = layout.height;
    push.pitchBytes = layout.pitchBytes;
    push.blocksPerRow = layout.blocksPerRow;
    push.tail = layout.tail ? 1u : 0u;
    push.tailX = layout.tailX;
    push.tailY = layout.tailY;
    push.elementBytes = elementBytes;
    push.slice = slice;
    push.rangeBegin = window.rangeBegin;
    push.rangeEnd = window.rangeEnd;
    push.tiledBase = window.tiledBase;
    push.linearBase = window.linearBase;
    push.columnBegin = window.columnBegin;
    push.rowBegin = window.rowBegin;
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
    const auto groupsX = (columnEnd - window.columnBegin + 7u) / 8u;
    const auto groupsY = (rowEnd - window.rowBegin + 7u) / 8u;
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groupsX, groupsY, 1);
}

void TextureDetiler::DispatchRepack(VkCommandBuffer commands, VkBuffer source, std::uint64_t sourceOffset, VkBuffer destination, std::uint64_t destinationOffset, std::uint64_t bytes) {
    Require(commands != VK_NULL_HANDLE && source != VK_NULL_HANDLE && destination != VK_NULL_HANDLE, "a texel repack needs a command buffer and its buffers");
    Require(bytes != 0 && bytes % 4u == 0 && sourceOffset % 4u == 0 && destinationOffset % 4u == 0, "a texel repack needs whole aligned words");
    if (repackPipeline == VK_NULL_HANDLE) {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = sizeof(TEXTURE_REPACK_SPV);
        moduleInfo.pCode = TEXTURE_REPACK_SPV;
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &repackModule), "vkCreateShaderModule repack");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = repackModule;
        stage.pName = "main";
        VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        createInfo.stage = stage;
        createInfo.layout = pipelineLayout;
        Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &repackPipeline), "vkCreateComputePipelines repack");
    }
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto sourceDescriptorOffset = sourceOffset - sourceOffset % alignment;
    const auto destinationDescriptorOffset = destinationOffset - destinationOffset % alignment;
    const auto sourceRange = sourceOffset - sourceDescriptorOffset + bytes;
    const auto destinationRange = destinationOffset - destinationDescriptorOffset + bytes;
    Require(sourceRange <= context.limits.maxStorageBufferRange && destinationRange <= context.limits.maxStorageBufferRange, "a texel repack exceeds the device's storage buffer range");
    const auto words = bytes / 4u;
    constexpr std::uint64_t rowWords = 64u * 1024u;
    const auto rows = (words + rowWords - 1u) / rowWords;
    Require(words <= UINT32_MAX && rows <= context.limits.maxComputeWorkGroupCount[1], "a texel repack exceeds the device's dispatch limits");
    const auto set = context.pushDescriptors ? VK_NULL_HANDLE : allocateSet();
    const VkDescriptorBufferInfo sourceInfo{source, sourceDescriptorOffset, sourceRange};
    const VkDescriptorBufferInfo destinationInfo{destination, destinationDescriptorOffset, destinationRange};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::uint32_t index = 0; index < writes.size(); ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = index == 0 ? &sourceInfo : &destinationInfo;
    }
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, repackPipeline);
    if (context.pushDescriptors) {
        context.Function<PFN_vkCmdPushDescriptorSetKHR>("vkCmdPushDescriptorSetKHR")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    } else {
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
    }
    const std::array<std::uint32_t, 4> push{static_cast<std::uint32_t>(sourceOffset - sourceDescriptorOffset), static_cast<std::uint32_t>(destinationOffset - destinationDescriptorOffset), static_cast<std::uint32_t>(words), static_cast<std::uint32_t>(std::min<std::uint64_t>(words, rowWords))};
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, static_cast<std::uint32_t>((std::min<std::uint64_t>(words, rowWords) + 63u) / 64u), static_cast<std::uint32_t>(rows), 1);
}

void TextureDetiler::DispatchImage(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkImageView view, VkBuffer tiled, std::uint64_t tiledOffset, const TileMipLayout& layout, bool retile, std::uint32_t slice, const DetileWindow& window) {
    Require(context.pushDescriptors, "image detiling needs push descriptors");
    Require(tileMode != TextureTileMode::kLinear, "image detiling of a linear surface");
    Require(commands != VK_NULL_HANDLE && view != VK_NULL_HANDLE && tiled != VK_NULL_HANDLE, "image detiling requires a command buffer, a view and a buffer");
    Require(window.rangeBegin < window.rangeEnd && window.tiledBase <= window.rangeBegin, "image detiling window is empty");
    const auto columnEnd = window.columnEnd != 0 ? std::min(window.columnEnd, layout.width) : layout.width;
    const auto rowEnd = window.rowEnd != 0 ? std::min(window.rowEnd, layout.height) : layout.height;
    Require(window.columnBegin < columnEnd && window.rowBegin < rowEnd, "image detiling window lies outside the mip");
    const auto target = pipeline(tileMode, elementBytes, retile, false, true);
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    const auto descriptorOffset = tiledOffset - tiledOffset % alignment;
    const auto base = tiledOffset - descriptorOffset;
    const std::uint64_t tiledBytes = std::min<std::uint64_t>(layout.tiledSize, window.rangeEnd) - window.tiledBase;
    const auto range = (base + tiledBytes + 3) / 4 * 4;
    Require(base <= UINT32_MAX && range <= context.limits.maxStorageBufferRange, "image detiling buffer range exceeds device limits");
    const VkDescriptorBufferInfo bufferInfo{tiled, descriptorOffset, range};
    const VkDescriptorImageInfo imageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstBinding = retile ? 1 : 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &bufferInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstBinding = 2;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &imageInfo;
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, target);
    context.Function<PFN_vkCmdPushDescriptorSetKHR>("vkCmdPushDescriptorSetKHR")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    Push push{};
    push.srcBase = retile ? 0u : static_cast<std::uint32_t>(base);
    push.dstBase = retile ? static_cast<std::uint32_t>(base) : 0u;
    push.width = layout.width;
    push.height = layout.height;
    push.pitchBytes = layout.pitchBytes;
    push.blocksPerRow = layout.blocksPerRow;
    push.tail = layout.tail ? 1u : 0u;
    push.tailX = layout.tailX;
    push.tailY = layout.tailY;
    push.elementBytes = elementBytes;
    push.slice = slice;
    push.rangeBegin = window.rangeBegin;
    push.rangeEnd = window.rangeEnd;
    push.tiledBase = window.tiledBase;
    push.linearBase = 0;
    push.columnBegin = window.columnBegin;
    push.rowBegin = window.rowBegin;
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, (columnEnd - window.columnBegin + 7u) / 8u, (rowEnd - window.rowBegin + 7u) / 8u, 1);
}

}
