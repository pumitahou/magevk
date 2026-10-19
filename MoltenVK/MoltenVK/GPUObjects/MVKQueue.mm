/*
 * MVKQueue.mm
 *
 * Copyright (c) 2015-2026 The Brenwill Workshop Ltd. (http://www.brenwill.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MVKInstance.h"
#include "MVKQueue.h"
#include "MVKBuffer.h"
#include "MVKImage.h"
#include "MVKDeviceMemory.h"
#include "MVKSurface.h"
#include "MVKSwapchain.h"
#include "MVKSync.h"
#include "MVKFoundation.h"
#include "MVKOSExtensions.h"
#include "MVKGPUCapture.h"
#include <unistd.h>

using namespace std;


#pragma mark -
#pragma mark MVKQueueFamily

// MTLCommandQueues are cached in MVKQueueFamily/MVKPhysicalDevice because they are very
// limited in number. An app that creates multiple VkDevices over time (such as a test suite)
// will soon find 15 second delays when creating subsequent MTLCommandQueues.
id<MTLCommandQueue> MVKQueueFamily::getMTLCommandQueue(uint32_t queueIndex) {
	lock_guard<mutex> lock(_qLock);
	id<MTLCommandQueue> mtlQ = _mtlQueues[queueIndex];
	if ( !mtlQ ) {
		@autoreleasepool {		// Catch any autoreleased objects created during MTLCommandQueue creation
			uint32_t maxCmdBuffs = getMVKConfig().maxActiveMetalCommandBuffersPerQueue;
			mtlQ = [_physicalDevice->getMTLDevice() newCommandQueueWithMaxCommandBufferCount: maxCmdBuffs];		// retained
			_mtlQueues[queueIndex] = mtlQ;
		}
	}
	return mtlQ;
}

MVKQueueFamily::MVKQueueFamily(MVKPhysicalDevice* physicalDevice, uint32_t queueFamilyIndex, const VkQueueFamilyProperties* pProperties) {
	_physicalDevice = physicalDevice;
	_queueFamilyIndex = queueFamilyIndex;
	_properties = *pProperties;
	_mtlQueues.assign(_properties.queueCount, nil);
}

MVKQueueFamily::~MVKQueueFamily() {
	mvkReleaseContainerContents(_mtlQueues);
}


#pragma mark -
#pragma mark MVKQueue

void MVKQueue::propagateDebugName() { setMetalObjectLabel(_mtlQueue, _debugName); }


#pragma mark Queue submissions

// Execute the queue submission under an autoreleasepool to ensure transient Metal objects are autoreleased.
// This is critical for apps that don't use standard OS autoreleasing runloop threading.
static inline VkResult execute(MVKQueueSubmission* qSubmit) { @autoreleasepool { return qSubmit->execute(); } }

// Executes the submmission, either immediately, or by dispatching to an execution queue.
// Submissions to the execution queue are wrapped in a dedicated autoreleasepool.
// Relying on the dispatch queue to find time to drain the autoreleasepool can
// result in significant memory creep under heavy workloads.
VkResult MVKQueue::submit(MVKQueueSubmission* qSubmit) {
	if (_device->getConfigurationResult() != VK_SUCCESS) { return _device->getConfigurationResult(); }

	if ( !qSubmit ) { return VK_SUCCESS; }     // Ignore nils
	if (qSubmit->requiresHostReadback() || qSubmit->requiresEncodingDependencyWait()) { initExecQueue(true); }

	// Extract result before submission to avoid race condition with early destruction
	// Submit regardless of config result, to ensure submission semaphores and fences are signalled.
	// The submissions will ensure a misconfiguration will be safe to execute.
	VkResult rslt = qSubmit->getConfigurationResult();
	if (_execQueue) {
		std::unique_lock lock(_execQueueMutex);
		_execQueueJobCount++;

		dispatch_async(_execQueue, ^{
			execute(qSubmit);

			std::unique_lock execLock(_execQueueMutex);
			if (!--_execQueueJobCount)
				_execQueueConditionVariable.notify_all();
		} );
	} else {
		rslt = execute(qSubmit);
	}
	return rslt;
}

static inline uint32_t getCommandBufferCount(const VkSubmitInfo2* pSubmitInfo) { return pSubmitInfo->commandBufferInfoCount; }
static inline uint32_t getCommandBufferCount(const VkSubmitInfo* pSubmitInfo) { return pSubmitInfo->commandBufferCount; }

template <typename S>
VkResult MVKQueue::submit(uint32_t submitCount, const S* pSubmits, VkFence fence, MVKCommandUse cmdUse) {

    // Fence-only submission
    if (submitCount == 0 && fence) {
        return submit(new MVKQueueCommandBufferSubmission(this, (S*)nullptr, fence, cmdUse));
    }

    VkResult rslt = VK_SUCCESS;
    for (uint32_t sIdx = 0; sIdx < submitCount; sIdx++) {
        VkFence fenceOrNil = (sIdx == (submitCount - 1)) ? fence : VK_NULL_HANDLE; // last one gets the fence

		const S* pVkSub = &pSubmits[sIdx];
		MVKQueueCommandBufferSubmission* mvkSub;
		uint32_t cbCnt = getCommandBufferCount(pVkSub);
		if (cbCnt <= 1) {
			mvkSub = new MVKQueueFullCommandBufferSubmission<1>(this, pVkSub, fenceOrNil, cmdUse);
		} else if (cbCnt <= 16) {
			mvkSub = new MVKQueueFullCommandBufferSubmission<16>(this, pVkSub, fenceOrNil, cmdUse);
		} else if (cbCnt <= 32) {
			mvkSub = new MVKQueueFullCommandBufferSubmission<32>(this, pVkSub, fenceOrNil, cmdUse);
		} else if (cbCnt <= 64) {
			mvkSub = new MVKQueueFullCommandBufferSubmission<64>(this, pVkSub, fenceOrNil, cmdUse);
		} else if (cbCnt <= 128) {
			mvkSub = new MVKQueueFullCommandBufferSubmission<128>(this, pVkSub, fenceOrNil, cmdUse);
		} else if (cbCnt <= 256) {
			mvkSub = new MVKQueueFullCommandBufferSubmission<256>(this, pVkSub, fenceOrNil, cmdUse);
		} else {
			mvkSub = new MVKQueueFullCommandBufferSubmission<512>(this, pVkSub, fenceOrNil, cmdUse);
		}

        VkResult subRslt = submit(mvkSub);
        if (rslt == VK_SUCCESS) { rslt = subRslt; }
    }
    return rslt;
}

// Concrete implementations of templated MVKQueue::submit().
template VkResult MVKQueue::submit(uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence, MVKCommandUse cmdUse);
template VkResult MVKQueue::submit(uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence, MVKCommandUse cmdUse);

// When a frame rate cap is configured via MVK_CONFIG_FRAME_RATE_CAP, enforce a minimum frame
// duration of 1/FPS seconds between presents, by sleeping the remainder on the calling thread.
static void throttleFrameRateCap() {
	static std::atomic<uint64_t> lastPresentTimestamp{0};

	uint32_t frameRateCapFPS = getGlobalMVKConfig().frameRateCapFPS;
	if ( !frameRateCapFPS ) { return; }

	uint64_t minFrameIntervalNS = 1000000000 / frameRateCapFPS;
	uint64_t lastTS = lastPresentTimestamp.load();
	uint64_t elapsedNS = lastTS ? mvkGetElapsedNanoseconds(lastTS) : minFrameIntervalNS;
	if (elapsedNS < minFrameIntervalNS) {
		usleep((useconds_t)(minFrameIntervalNS - elapsedNS) / 1000);
	}
	lastPresentTimestamp.store(mvkGetTimestamp());
}

VkResult MVKQueue::submit(const VkPresentInfoKHR* pPresentInfo) {
	throttleFrameRateCap();
	return submit(new MVKQueuePresentSurfaceSubmission(this, pPresentInfo));
}

VkResult MVKQueue::bindSparse(uint32_t bindInfoCount, const VkBindSparseInfo* pBindInfo, VkFence fence) {
#if MVK_XCODE_26 && MVK_MACOS
	if (!_device->isPrivateSparseBufferEnabled() || !_mtl4SparseQueue || !_mtl4SparseEvent) {
		return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Private sparse buffers are not available.");
	}
	if (!bindInfoCount) {
		return fence ? submit(new MVKQueueCommandBufferSubmission(this, (VkSubmitInfo*)nullptr, fence, kMVKCommandUseQueueSubmit)) : VK_SUCCESS;
	}
	if (!pBindInfo) {
		return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Bind info is missing.");
	}
	if (bindInfoCount > (UINT64_MAX - _mtl4SparseEventValue) / 2) {
		return reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkQueueBindSparse(): Internal event values are exhausted.");
	}
	if (fence && ((MVKFence*)fence)->getDevice() != _device) {
		return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Fence belongs to another device.");
	}

	for (uint32_t infoIdx = 0; infoIdx < bindInfoCount; infoIdx++) {
		const VkBindSparseInfo& bindInfo = pBindInfo[infoIdx];
		if (bindInfo.sType != VK_STRUCTURE_TYPE_BIND_SPARSE_INFO ||
			(bindInfo.waitSemaphoreCount && !bindInfo.pWaitSemaphores) ||
			(bindInfo.bufferBindCount && !bindInfo.pBufferBinds) ||
			(bindInfo.imageOpaqueBindCount && !bindInfo.pImageOpaqueBinds) ||
			(bindInfo.imageBindCount && !bindInfo.pImageBinds) ||
			(bindInfo.signalSemaphoreCount && !bindInfo.pSignalSemaphores)) {
			return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Bind info %u is not supported.", infoIdx);
		}

		const VkTimelineSemaphoreSubmitInfo* timelineInfo = nullptr;
		for (const auto* next = (const VkBaseInStructure*)bindInfo.pNext; next; next = next->pNext) {
			if (next->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO || timelineInfo) {
				return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Bind info %u has an unsupported pNext structure.", infoIdx);
			}
			timelineInfo = (const VkTimelineSemaphoreSubmitInfo*)next;
		}
		if (timelineInfo &&
			(timelineInfo->waitSemaphoreValueCount != bindInfo.waitSemaphoreCount ||
			 timelineInfo->signalSemaphoreValueCount != bindInfo.signalSemaphoreCount ||
			 (timelineInfo->waitSemaphoreValueCount && !timelineInfo->pWaitSemaphoreValues) ||
			 (timelineInfo->signalSemaphoreValueCount && !timelineInfo->pSignalSemaphoreValues))) {
			return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Bind info %u has invalid timeline semaphore values.", infoIdx);
		}

		bool hasTimelineSemaphore = false;
		for (uint32_t semIdx = 0; semIdx < bindInfo.waitSemaphoreCount; semIdx++) {
			auto* semaphore = (MVKSemaphore*)bindInfo.pWaitSemaphores[semIdx];
			if (!semaphore || semaphore->getDevice() != _device) {
				return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Wait semaphore belongs to another device.");
			}
			hasTimelineSemaphore |= semaphore->getSemaphoreType() == VK_SEMAPHORE_TYPE_TIMELINE;
		}
		for (uint32_t semIdx = 0; semIdx < bindInfo.signalSemaphoreCount; semIdx++) {
			auto* semaphore = (MVKSemaphore*)bindInfo.pSignalSemaphores[semIdx];
			if (!semaphore || semaphore->getDevice() != _device) {
				return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Signal semaphore belongs to another device.");
			}
			hasTimelineSemaphore |= semaphore->getSemaphoreType() == VK_SEMAPHORE_TYPE_TIMELINE;
		}
		if (hasTimelineSemaphore && !timelineInfo) {
			return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Timeline semaphore values are missing.");
		}

		for (uint32_t bufferIdx = 0; bufferIdx < bindInfo.bufferBindCount; bufferIdx++) {
			const VkSparseBufferMemoryBindInfo& bufferInfo = bindInfo.pBufferBinds[bufferIdx];
			auto* buffer = (MVKBuffer*)bufferInfo.buffer;
			if (!buffer || buffer->getDevice() != _device || !buffer->isPrivateSparseBuffer() ||
				!buffer->getMTLBuffer() || (bufferInfo.bindCount && !bufferInfo.pBinds)) {
				return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Buffer bind %u in info %u is not supported.", bufferIdx, infoIdx);
			}
			VkDeviceSize bufferSize = buffer->getByteCount();
			if (bufferSize > UINT64_MAX - (kMVKPrivateSparseBufferPageSize - 1)) {
				return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Sparse buffer size overflows.");
			}
			bufferSize = mvkAlignByteCount(bufferSize, kMVKPrivateSparseBufferPageSize);

			for (uint32_t bindIdx = 0; bindIdx < bufferInfo.bindCount; bindIdx++) {
				const VkSparseMemoryBind& bind = bufferInfo.pBinds[bindIdx];
				if (bind.flags ||
					!bind.size ||
					bind.resourceOffset % kMVKPrivateSparseBufferPageSize ||
					bind.size % kMVKPrivateSparseBufferPageSize ||
					bind.resourceOffset > UINT64_MAX - bind.size ||
					bind.resourceOffset + bind.size > bufferSize) {
					return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Buffer bind range is invalid.");
				}
				if (bind.memory) {
					auto* memory = (MVKDeviceMemory*)bind.memory;
					if (memory->getDevice() != _device ||
						!memory->isPlacementSparseCompatible() ||
						!memory->getMTLPlacementSparseHeap() ||
						bind.memoryOffset % kMVKPrivateSparseBufferPageSize ||
						bind.memoryOffset > UINT64_MAX - bind.size ||
						bind.memoryOffset + bind.size > memory->getDeviceMemoryCommitment()) {
						return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Sparse memory bind range is invalid.");
					}
				}
			}
		}

		for (uint32_t opaqueIdx = 0; opaqueIdx < bindInfo.imageOpaqueBindCount; opaqueIdx++) {
			const VkSparseImageOpaqueMemoryBindInfo& opaqueInfo = bindInfo.pImageOpaqueBinds[opaqueIdx];
			auto* image = (MVKImage*)opaqueInfo.image;
			if (!image || image->getDevice() != _device || !image->isPrivateSparseImage() ||
				(opaqueInfo.bindCount && !opaqueInfo.pBinds)) {
				return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Image opaque bind %u in info %u is not supported.", opaqueIdx, infoIdx);
			}
			// Metal maps the mip tail of one texture slice all at once, so an opaque
			// bind must cover the whole mip tail of exactly one array layer.
			VkDeviceSize tailSize = image->getMipTailSize();
			for (uint32_t bindIdx = 0; bindIdx < opaqueInfo.bindCount; bindIdx++) {
				const VkSparseMemoryBind& bind = opaqueInfo.pBinds[bindIdx];
				if (bind.flags ||
					!tailSize ||
					bind.size != tailSize ||
					bind.resourceOffset % tailSize ||
					bind.resourceOffset / tailSize >= image->getLayerCount()) {
					return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Image opaque bind range must cover the full mip tail of one array layer.");
				}
				if (bind.memory) {
					auto* memory = (MVKDeviceMemory*)bind.memory;
					if (memory->getDevice() != _device ||
						!memory->isPlacementSparseCompatible() ||
						!memory->getMTLPlacementSparseHeap() ||
						bind.memoryOffset % image->getSparseTileByteCount() ||
						bind.memoryOffset > UINT64_MAX - bind.size ||
						bind.memoryOffset + bind.size > memory->getDeviceMemoryCommitment()) {
						return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Sparse memory bind range is invalid.");
					}
				}
			}
		}

		for (uint32_t imageIdx = 0; imageIdx < bindInfo.imageBindCount; imageIdx++) {
			const VkSparseImageMemoryBindInfo& imageInfo = bindInfo.pImageBinds[imageIdx];
			auto* image = (MVKImage*)imageInfo.image;
			if (!image || image->getDevice() != _device || !image->isPrivateSparseImage() ||
				(imageInfo.bindCount && !imageInfo.pBinds)) {
				return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Image bind %u in info %u is not supported.", imageIdx, infoIdx);
			}
			// Metal maps whole sparse tiles, so bind offsets must align to the sparse
			// block granularity, and extents must be granularity-aligned except when
			// they reach the edge of the mip level (partial edge blocks).
			VkExtent3D granularity = image->getSparseTileExtent();
			for (uint32_t bindIdx = 0; bindIdx < imageInfo.bindCount; bindIdx++) {
				const VkSparseImageMemoryBind& bind = imageInfo.pBinds[bindIdx];
				if (bind.flags ||
					bind.subresource.aspectMask != image->getSparseAspectMask() ||
					bind.subresource.mipLevel >= image->getMipTailFirstLod() ||
					bind.subresource.arrayLayer >= image->getLayerCount()) {
					return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Image bind subresource is invalid. Mip tails must be bound with opaque binds.");
				}
				VkExtent3D mipExtent = image->getExtent3D(0, bind.subresource.mipLevel);
				if (!bind.extent.width || !bind.extent.height || !bind.extent.depth ||
					bind.extent.width > mipExtent.width ||
					bind.extent.height > mipExtent.height ||
					bind.extent.depth > mipExtent.depth ||
					bind.offset.x % granularity.width ||
					bind.offset.y % granularity.height ||
					bind.offset.z % granularity.depth ||
					bind.offset.x > mipExtent.width - bind.extent.width ||
					bind.offset.y > mipExtent.height - bind.extent.height ||
					bind.offset.z > mipExtent.depth - bind.extent.depth ||
					((bind.extent.width % granularity.width) && bind.offset.x + bind.extent.width != mipExtent.width) ||
					((bind.extent.height % granularity.height) && bind.offset.y + bind.extent.height != mipExtent.height) ||
					((bind.extent.depth % granularity.depth) && bind.offset.z + bind.extent.depth != mipExtent.depth)) {
					return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Image bind region is not aligned to the sparse block granularity.");
				}
				// Partial edge blocks still occupy a full sparse block in memory.
				VkDeviceSize tileByteCount = image->getSparseTileByteCount();
				uint64_t tilesX = (bind.offset.x + bind.extent.width + granularity.width - 1) / granularity.width - bind.offset.x / granularity.width;
				uint64_t tilesY = (bind.offset.y + bind.extent.height + granularity.height - 1) / granularity.height - bind.offset.y / granularity.height;
				uint64_t tilesZ = (bind.offset.z + bind.extent.depth + granularity.depth - 1) / granularity.depth - bind.offset.z / granularity.depth;
				uint64_t regionByteCount = tilesX * tilesY * tilesZ * tileByteCount;
				if (bind.memory) {
					auto* memory = (MVKDeviceMemory*)bind.memory;
					if (memory->getDevice() != _device ||
						!memory->isPlacementSparseCompatible() ||
						!memory->getMTLPlacementSparseHeap() ||
						bind.memoryOffset % tileByteCount ||
						bind.memoryOffset > UINT64_MAX - regionByteCount ||
						bind.memoryOffset + regionByteCount > memory->getDeviceMemoryCommitment()) {
						return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Sparse memory bind range is invalid.");
					}
				}
			}
		}
	}

	MVKSmallVector<MVKQueueSparseSubmission*, 4> submissions;
	submissions.reserve(bindInfoCount);
	uint64_t eventValue = _mtl4SparseEventValue;
	for (uint32_t infoIdx = 0; infoIdx < bindInfoCount; infoIdx++) {
		VkFence submissionFence = infoIdx + 1 == bindInfoCount ? fence : VK_NULL_HANDLE;
		auto* submission = new MVKQueueSparseSubmission(this,
														&pBindInfo[infoIdx],
														submissionFence,
														eventValue + 1,
														eventValue + 2);
		eventValue += 2;
		submissions.push_back(submission);
		if (submission->getConfigurationResult() != VK_SUCCESS) {
			VkResult result = submission->getConfigurationResult();
			for (auto* ownedSubmission : submissions) { ownedSubmission->destroy(); }
			return result;
		}
	}
	_mtl4SparseEventValue = eventValue;

	VkResult result = VK_SUCCESS;
	for (auto* submission : submissions) {
		VkResult submissionResult = submit(submission);
		if (result == VK_SUCCESS) { result = submissionResult; }
	}
	return result;
#else
	return reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkQueueBindSparse(): Private sparse buffers are not available.");
#endif
}

VkResult MVKQueue::waitIdle(MVKCommandUse cmdUse) {
	if (_execQueue) {
		std::unique_lock lock(_execQueueMutex);
		while (_execQueueJobCount)
			_execQueueConditionVariable.wait(lock);
	}
	@autoreleasepool {
		auto* mtlCmdBuff = getMTLCommandBuffer(cmdUse);
		[mtlCmdBuff commit];
		[mtlCmdBuff waitUntilCompleted];
	}
	return _device->getConfigurationResult();
}

id<MTLCommandBuffer> MVKQueue::getMTLCommandBuffer(MVKCommandUse cmdUse, bool retainRefs) {
	id<MTLCommandBuffer> mtlCmdBuff = nil;
	uint64_t startTime = getPerformanceTimestamp();

	MTLCommandBufferDescriptor* mtlCmdBuffDesc = [MTLCommandBufferDescriptor new];	// temp retain
	static const bool retainForRadiance = [] {
		const char* value = getenv("RADIANCE_MAGEVK_RETAIN_RESOURCES");
		return value && strcmp(value, "1") == 0;
	}();
	// Keep resources that Metal binds, including temporary RT tables and
	// buffers, alive until this command buffer completes. This is opt-in
	// while investigating the Invalid Resource error observed in Minecraft.
	mtlCmdBuffDesc.retainedReferences = retainRefs || retainForRadiance;
	if (getMVKConfig().debugMode || retainForRadiance) {
		mtlCmdBuffDesc.errorOptions |= MTLCommandBufferErrorOptionEncoderExecutionStatus;
	}
	if (retainForRadiance) {
		static std::once_flag reportOnce;
		std::call_once(reportOnce, [&] {
			fprintf(stderr, "[RADIANCE-E13-MAGEVK] retainedReferences=1 encoderErrorStatus=1 residencySet=%u\n",
				getDevice()->hasResidencySet() ? 1u : 0u);
		});
	}
	mtlCmdBuff = [_mtlQueue commandBufferWithDescriptor: mtlCmdBuffDesc];
	[mtlCmdBuffDesc release];														// temp release

	addPerformanceInterval(getPerformanceStats().queue.retrieveMTLCommandBuffer, startTime);
	NSString* mtlCmdBuffLabel = getMTLCommandBufferLabel(cmdUse);
	setMetalObjectLabel(mtlCmdBuff, mtlCmdBuffLabel);
	[mtlCmdBuff addCompletedHandler: ^(id<MTLCommandBuffer> mtlCB) { handleMTLCommandBufferError(mtlCB); }];

	if ( !mtlCmdBuff ) { reportError(VK_ERROR_OUT_OF_POOL_MEMORY, "%s could not be acquired.", mtlCmdBuffLabel.UTF8String); }
	return mtlCmdBuff;
}

NSString* MVKQueue::getMTLCommandBufferLabel(MVKCommandUse cmdUse) {
#define CASE_GET_LABEL(cu)  \
	case kMVKCommandUse ##cu:  \
		if ( !_mtlCmdBuffLabel ##cu ) { _mtlCmdBuffLabel ##cu = [[NSString stringWithFormat: @"%s MTLCommandBuffer on Queue %d-%d", mvkVkCommandName(kMVKCommandUse ##cu), _queueFamily->getIndex(), _index] retain]; }  \
		return _mtlCmdBuffLabel ##cu

	switch (cmdUse) {
		CASE_GET_LABEL(BeginCommandBuffer);
		CASE_GET_LABEL(QueueSubmit);
		CASE_GET_LABEL(QueuePresent);
		CASE_GET_LABEL(QueueWaitIdle);
		CASE_GET_LABEL(DeviceWaitIdle);
		CASE_GET_LABEL(AcquireNextImage);
		CASE_GET_LABEL(InvalidateMappedMemoryRanges);
		CASE_GET_LABEL(CopyImageToMemory);
		default:
			MVKAssert(false, "Uncached MTLCommandBuffer label for command use %s.", mvkVkCommandName(cmdUse));
			return [NSString stringWithFormat: @"%s MTLCommandBuffer on Queue %d-%d", mvkVkCommandName(cmdUse), _queueFamily->getIndex(), _index];
	}
#undef CASE_GET_LABEL
}

static const char* mvkStringFromMTLCommandEncoderErrorState(MTLCommandEncoderErrorState errState) {
	switch (errState) {
		case MTLCommandEncoderErrorStateUnknown:   return "unknown";
		case MTLCommandEncoderErrorStateAffected:  return "affected";
		case MTLCommandEncoderErrorStateCompleted: return "completed";
		case MTLCommandEncoderErrorStateFaulted:   return "faulted";
		case MTLCommandEncoderErrorStatePending:   return "pending";
	}
	return "unknown";
}

void MVKQueue::handleMTLCommandBufferError(id<MTLCommandBuffer> mtlCmdBuff) {
	if (mtlCmdBuff.status != MTLCommandBufferStatusError) { return; }

	// If a command buffer error has occurred, report the error. If the error affects
	// the physical device, always mark both the device and physical device as lost.
	// If the error is local to this command buffer, optionally mark the device (but not the
	// physical device) as lost, depending on the value of MVKConfiguration::resumeLostDevice.
	VkResult vkErr = VK_ERROR_UNKNOWN;
	bool markDeviceLoss = !getMVKConfig().resumeLostDevice;
	bool markPhysicalDeviceLoss = false;
	switch (mtlCmdBuff.error.code) {
		case MTLCommandBufferErrorBlacklisted:
		case MTLCommandBufferErrorNotPermitted:	// May also be used for command buffers executed in the background without the right entitlement.
#if MVK_MACOS && !MVK_MACCAT
		case MTLCommandBufferErrorDeviceRemoved:
#endif
			vkErr = VK_ERROR_DEVICE_LOST;
			markDeviceLoss = true;
			markPhysicalDeviceLoss = true;
			break;
		case MTLCommandBufferErrorTimeout:
			vkErr = VK_TIMEOUT;
			break;
		case MTLCommandBufferErrorInvalidResource:
			vkErr = VK_ERROR_DEVICE_LOST;
			break;
		case MTLCommandBufferErrorStackOverflow:
		case MTLCommandBufferErrorPageFault:
		case MTLCommandBufferErrorOutOfMemory:
		default:
			vkErr = VK_ERROR_OUT_OF_DEVICE_MEMORY;
			break;
	}
	if (markDeviceLoss) {
		getDevice()->stopAutoGPUCapture(MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE_DEVICE);
		getDevice()->markLost(markPhysicalDeviceLoss);
	}
	reportResult(vkErr, (markDeviceLoss ? MVK_CONFIG_LOG_LEVEL_ERROR : MVK_CONFIG_LOG_LEVEL_WARNING),
				 "%s VkDevice after MTLCommandBuffer \"%s\" execution failed (code %li): %s",
				 (markDeviceLoss ? "Lost" : "Resumed"),
				 (mtlCmdBuff.label ? mtlCmdBuff.label.UTF8String : ""),
				 mtlCmdBuff.error.code, mtlCmdBuff.error.localizedDescription.UTF8String);

	if (NSArray<id<MTLCommandBufferEncoderInfo>>* mtlEncInfo = mtlCmdBuff.error.userInfo[MTLCommandBufferEncoderInfoErrorKey]) {
		MVKLogInfo("Encoders for %p \"%s\":", mtlCmdBuff, mtlCmdBuff.label ? mtlCmdBuff.label.UTF8String : "");
		for (id<MTLCommandBufferEncoderInfo> enc in mtlEncInfo) {
			MVKLogInfo(" - %s: %s", enc.label.UTF8String, mvkStringFromMTLCommandEncoderErrorState(enc.errorState));
			if (enc.debugSignposts.count > 0) {
				MVKLogInfo("   Debug signposts:");
				for (NSString* signpost in enc.debugSignposts) {
					MVKLogInfo("    - %s", signpost.UTF8String);
				}
			}
		}
	}

	bool isFirstMsg = true;
	for (id<MTLFunctionLog> log in mtlCmdBuff.logs) {
		if (isFirstMsg) {
			MVKLogInfo("Shader log messages:");
			isFirstMsg = false;
		}
		MVKLogInfo("%s", log.description.UTF8String);
	}
}

#pragma mark Construction

MVKQueue::MVKQueue(MVKDevice* device, MVKQueueFamily* queueFamily, uint32_t index, float priority, VkQueueGlobalPriority globalPriority) : MVKDeviceTrackingMixin(device) {
	_queueFamily = queueFamily;
	_index = index;
	_priority = priority;
	_globalPriority = globalPriority;

	initName();
	initExecQueue();
	initMTLCommandQueue();
}

void MVKQueue::initName() {
	const char* fmt = "MoltenVKQueue-%d-%d-%.1f";
	char name[256];
	snprintf(name, sizeof(name)/sizeof(char), fmt, _queueFamily->getIndex(), _index, _priority);
	_name = name;
}

void MVKQueue::initExecQueue(bool force) {
	if ( !_execQueue && (!getMVKConfig().synchronousQueueSubmits || force) ) {
		// Determine the dispatch queue priority
		dispatch_qos_class_t dqQOS;
		switch (_globalPriority) {
			case VK_QUEUE_GLOBAL_PRIORITY_LOW:
				dqQOS = QOS_CLASS_UTILITY;
				break;
			case VK_QUEUE_GLOBAL_PRIORITY_HIGH:
				dqQOS = QOS_CLASS_USER_INTERACTIVE;
				break;
			case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM:
			default: // Fall back to default (medium)
				dqQOS = QOS_CLASS_USER_INITIATED;
				break;
		}
		int dqPriority = (1.0 - _priority) * QOS_MIN_RELATIVE_PRIORITY;
		dispatch_queue_attr_t dqAttr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, dqQOS, dqPriority);

		// Create the dispatch queue
		_execQueue = dispatch_queue_create((getName() + "-Dispatch").c_str(), dqAttr);		// retained
	}
}

// Retrieves and initializes the Metal command queue and Xcode GPU capture scopes
void MVKQueue::initMTLCommandQueue() {
	_mtlQueue = _queueFamily->getMTLCommandQueue(_index);	// not retained (cached in queue family)
	_device->addResidencySet(_mtlQueue);

#if MVK_XCODE_26 && MVK_MACOS
	if (_device->isPrivateSparseBufferEnabled()) {
		if (@available(macOS 26.4, *)) {
			MTL4CommandQueueDescriptor* descriptor = [MTL4CommandQueueDescriptor new];
			_mtl4SparseQueue = [getMTLDevice() newMTL4CommandQueueWithDescriptor:descriptor error:nil];
			[descriptor release];
			_mtl4SparseEvent = [getMTLDevice() newSharedEvent];
			if (_mtl4SparseQueue && _mtl4SparseEvent) {
				_device->addResidencySet((id<MTLCommandQueue>)_mtl4SparseQueue);
			} else {
				[_mtl4SparseQueue release];
				[_mtl4SparseEvent release];
				_mtl4SparseQueue = nil;
				_mtl4SparseEvent = nil;
			}
		}
	}
#endif

	_submissionCaptureScope = new MVKGPUCaptureScope(this);
	if (_queueFamily->getIndex() == getMVKConfig().defaultGPUCaptureScopeQueueFamilyIndex &&
		_index == getMVKConfig().defaultGPUCaptureScopeQueueIndex) {
		getDevice()->startAutoGPUCapture(MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE_FRAME, _mtlQueue);
		_submissionCaptureScope->makeDefault();
	}
	_submissionCaptureScope->beginScope();	// Allow Xcode to capture the first frame if desired.
}

MVKQueue::~MVKQueue() {
	destroyExecQueue();
	_submissionCaptureScope->destroy();
	_device->removeResidencySet(_mtlQueue);

#if MVK_XCODE_26 && MVK_MACOS
	if (_mtl4SparseQueue) {
		_device->removeResidencySet((id<MTLCommandQueue>)_mtl4SparseQueue);
	}
	[_mtl4SparseQueue release];
	[_mtl4SparseEvent release];
#endif

	[_mtlCmdBuffLabelBeginCommandBuffer release];
	[_mtlCmdBuffLabelQueueSubmit release];
	[_mtlCmdBuffLabelQueuePresent release];
	[_mtlCmdBuffLabelDeviceWaitIdle release];
	[_mtlCmdBuffLabelQueueWaitIdle release];
	[_mtlCmdBuffLabelAcquireNextImage release];
	[_mtlCmdBuffLabelInvalidateMappedMemoryRanges release];
}

// Destroys the execution dispatch queue.
void MVKQueue::destroyExecQueue() {
	if (_execQueue) {
		dispatch_release(_execQueue);
		_execQueue = nullptr;
	}
}


#pragma mark -
#pragma mark MVKQueueSubmission

void MVKSemaphoreSubmitInfo::encodeWait(id<MTLCommandBuffer> mtlCmdBuff) {
	if (_semaphore) { _semaphore->encodeWait(mtlCmdBuff, value); }
}

void MVKSemaphoreSubmitInfo::encodeSignal(id<MTLCommandBuffer> mtlCmdBuff) {
	if (_semaphore) { _semaphore->encodeSignal(mtlCmdBuff, value); }
}

MVKSemaphoreSubmitInfo::MVKSemaphoreSubmitInfo(const VkSemaphoreSubmitInfo& semaphoreSubmitInfo) :
	_semaphore((MVKSemaphore*)semaphoreSubmitInfo.semaphore),
	value(semaphoreSubmitInfo.value),
	stageMask(semaphoreSubmitInfo.stageMask),
	deviceIndex(semaphoreSubmitInfo.deviceIndex) {
		if (_semaphore) { _semaphore->retain(); }
}

MVKSemaphoreSubmitInfo::MVKSemaphoreSubmitInfo(const VkSemaphore semaphore,
											   VkPipelineStageFlags stageMask) :
	_semaphore((MVKSemaphore*)semaphore),
	value(0),
	stageMask(stageMask),
	deviceIndex(0) {
		if (_semaphore) { _semaphore->retain(); }
}

MVKSemaphoreSubmitInfo::MVKSemaphoreSubmitInfo(const MVKSemaphoreSubmitInfo& other) :
	_semaphore(other._semaphore),
	value(other.value),
	stageMask(other.stageMask),
	deviceIndex(other.deviceIndex) {
		if (_semaphore) { _semaphore->retain(); }
}

MVKSemaphoreSubmitInfo& MVKSemaphoreSubmitInfo::operator=(const MVKSemaphoreSubmitInfo& other) {
	// Retain new object first in case it's the same object
	if (other._semaphore) {other._semaphore->retain(); }
	if (_semaphore) { _semaphore->release(); }
	_semaphore = other._semaphore;

	value = other.value;
	stageMask = other.stageMask;
	deviceIndex = other.deviceIndex;
	return *this;
}

MVKSemaphoreSubmitInfo::~MVKSemaphoreSubmitInfo() {
	if (_semaphore) { _semaphore->release(); }
}

MVKCommandBufferSubmitInfo::MVKCommandBufferSubmitInfo(const VkCommandBufferSubmitInfo& commandBufferInfo) :
	commandBuffer(MVKCommandBuffer::getMVKCommandBuffer(commandBufferInfo.commandBuffer)),
	deviceMask(commandBufferInfo.deviceMask) {}

MVKCommandBufferSubmitInfo::MVKCommandBufferSubmitInfo(VkCommandBuffer commandBuffer) :
	commandBuffer(MVKCommandBuffer::getMVKCommandBuffer(commandBuffer)),
	deviceMask(0) {}

MVKQueueSubmission::MVKQueueSubmission(MVKQueue* queue,
									   uint32_t waitSemaphoreInfoCount,
									   const VkSemaphoreSubmitInfo* pWaitSemaphoreSubmitInfos) : 
	MVKBaseDeviceObject(queue->getDevice()),
	_queue(queue) {

	_queue->retain();	// Retain here and release in destructor. See note for MVKQueueCommandBufferSubmission::finish().
	_creationTime = getPerformanceTimestamp();

	_waitSemaphores.reserve(waitSemaphoreInfoCount);
	for (uint32_t i = 0; i < waitSemaphoreInfoCount; i++) {
		_waitSemaphores.emplace_back(pWaitSemaphoreSubmitInfos[i]);
	}
}

MVKQueueSubmission::MVKQueueSubmission(MVKQueue* queue,
									   uint32_t waitSemaphoreCount,
									   const VkSemaphore* pWaitSemaphores,
									   const VkPipelineStageFlags* pWaitDstStageMask) :
	MVKBaseDeviceObject(queue->getDevice()),
	_queue(queue) {

	_queue->retain();	// Retain here and release in destructor. See note for MVKQueueCommandBufferSubmission::finish().
	_creationTime = getPerformanceTimestamp();

	_waitSemaphores.reserve(waitSemaphoreCount);
	for (uint32_t i = 0; i < waitSemaphoreCount; i++) {
		_waitSemaphores.emplace_back(pWaitSemaphores[i], pWaitDstStageMask ? pWaitDstStageMask[i] : 0);
	}
}

MVKQueueSubmission::~MVKQueueSubmission() {
	_queue->release();
}


#pragma mark -
#pragma mark MVKQueueCommandBufferSubmission

VkResult MVKQueueCommandBufferSubmission::execute() {

	_queue->_submissionCaptureScope->beginScope();

	// If using encoded semaphore waiting, do so now.
	for (auto& ws : _waitSemaphores) { ws.encodeWait(getActiveMTLCommandBuffer()); }
	if (requiresEncodingDependencyWait()) {
		setConfigurationResult(commitActiveMTLCommandBufferAndWait());
	}

	// Wait time from an async vkQueueSubmit() call to starting submit and encoding of the command buffers
	addPerformanceInterval(_queue->getPerformanceStats().queue.waitSubmitCommandBuffers, _creationTime);

	// Submit each command buffer.
	submitCommandBuffers();

	// If using encoded semaphore signaling, do so now.
	for (auto& ss : _signalSemaphores) { ss.encodeSignal(getActiveMTLCommandBuffer()); }

	// Commit the last MTLCommandBuffer.
	// Nothing after this because callback might destroy this instance before this function ends.
	return commitActiveMTLCommandBuffer(true);
}

// Returns the active MTLCommandBuffer, lazily retrieving it from the queue if needed.
id<MTLCommandBuffer> MVKQueueCommandBufferSubmission::getActiveMTLCommandBuffer() {
	if ( !_activeMTLCommandBuffer ) {
		bool needsRetain = false;
		if (!_device->hasResidencySet() &&
			(getEnabledDescriptorIndexingFeatures().descriptorBindingPartiallyBound ||
			 getMVKConfig().liveCheckAllResources)) {
			// Partially bound descriptors will get bound by us even if they're not used at runtime by the shader.
			// The application is free to destroy them even if they're not used at runtime even if we bound them.
			// Metal will be very unhappy if we destroy something we bound, even if it isn't used at runtime.
			needsRetain = true;
		}
		setActiveMTLCommandBuffer(_queue->getMTLCommandBuffer(_commandUse, needsRetain));
	}
	return _activeMTLCommandBuffer;
}

// Commits the current active MTLCommandBuffer, if it exists, and sets a new active MTLCommandBuffer.
void MVKQueueCommandBufferSubmission::setActiveMTLCommandBuffer(id<MTLCommandBuffer> mtlCmdBuff) {

	if (_activeMTLCommandBuffer) { commitActiveMTLCommandBuffer(); }

	_activeMTLCommandBuffer = [mtlCmdBuff retain];		// retained to handle prefilled
	[_activeMTLCommandBuffer enqueue];
}

// Commits and releases the currently active MTLCommandBuffer, optionally signalling
// when the MTLCommandBuffer is done. The first time this is called, it will wait on
// any semaphores. We have delayed signalling the semaphores as long as possible to
// allow as much filling of the MTLCommandBuffer as possible before forcing a wait.
VkResult MVKQueueCommandBufferSubmission::commitActiveMTLCommandBuffer(bool signalCompletion) {

	// If using inline semaphore waiting, do so now.
	// When prefilled command buffers are used, multiple commits will happen because native semaphore
	// waits need to be committed before the prefilled command buffer is committed. Since semaphores
	// will reset their internal signal flag on wait, we need to make sure that we only wait once, otherwise we will freeze.
	// Another option to wait on emulated semaphores once is to do it in the execute function, but doing it here
	// should be more performant when prefilled command buffers aren't used, because we spend time encoding commands
	// first, thus giving the command buffer signalling these semaphores more time to complete.
	if ( !_emulatedWaitDone ) {
		for (auto& ws : _waitSemaphores) { ws.encodeWait(nil); }
		_emulatedWaitDone = true;
	}

	// The visibility result buffer will be returned to its pool when the active MTLCommandBuffer
	// finishes executing, and therefore cannot be used beyond the active MTLCommandBuffer.
	// By now, it's been submitted to the MTLCommandBuffer, so remove it from the encoding context,
	// to ensure a fresh one will be used by commands executing on any subsequent MTLCommandBuffers.
	if (_encodingContext.visibilityResultBuffer.buffer())
		_device->returnVisibilityBuffer(std::move(_encodingContext.visibilityResultBuffer));

	// If this is the last command buffer in the submission, we're losing the context and need synchronize
	// current barrier fences to the ones at index 0, which will be what the next submision starts with.
	if (isUsingMetalArgumentBuffers() && signalCompletion) {
		_encodingContext.syncFences(getDevice(), _activeMTLCommandBuffer);
	}

	// If we need to signal completion, use getActiveMTLCommandBuffer() to ensure at least
	// one MTLCommandBuffer is used, otherwise if this instance has no content, it will not
	// finish(), signal the fence and semaphores, and be destroyed.
	// Use temp var for MTLCommandBuffer commit and release because completion callback
	// may destroy this instance before this function ends.
	id<MTLCommandBuffer> mtlCmdBuff = signalCompletion ? getActiveMTLCommandBuffer() : _activeMTLCommandBuffer;
	_activeMTLCommandBuffer = nil;

	uint64_t startTime = getPerformanceTimestamp();
	[mtlCmdBuff addCompletedHandler: ^(id<MTLCommandBuffer> mtlCB) {
		addPerformanceInterval(getPerformanceStats().queue.mtlCommandBufferExecution, startTime);
		if (signalCompletion) { this->finish(); }	// Must be the last thing the completetion callback does.
	}];

	// Retrieve the result before committing MTLCommandBuffer, because finish() will destroy this instance.
	VkResult rslt = mtlCmdBuff ? getConfigurationResult() : VK_ERROR_OUT_OF_POOL_MEMORY;
	[mtlCmdBuff commit];
	[mtlCmdBuff release];		// retained

	// If we need to signal completion, but an error occurred and the MTLCommandBuffer
	// was not created, call the finish() function directly.
	if (signalCompletion && !mtlCmdBuff) { finish(); }

	return rslt;
}

VkResult MVKQueueCommandBufferSubmission::commitActiveMTLCommandBufferAndWait() {
	id<MTLCommandBuffer> mtlCmdBuff = [_activeMTLCommandBuffer retain];
	VkResult rslt = commitActiveMTLCommandBuffer();
	[mtlCmdBuff waitUntilCompleted];
	if (rslt == VK_SUCCESS && mtlCmdBuff.status != MTLCommandBufferStatusCompleted) {
		rslt = VK_ERROR_DEVICE_LOST;
	}
	[mtlCmdBuff release];
	_encodingContext.fenceSlots = {};
	return rslt;
}

// Be sure to retain() any API objects referenced in this function, and release() them in the
// destructor (or superclass destructor). It is possible for rare race conditions to result
// in the app destroying API objects before this function completes execution. For example,
// this may occur if a GPU semaphore here triggers another submission that triggers a fence,
// and the app immediately destroys objects. Rare, but it has been encountered.
void MVKQueueCommandBufferSubmission::finish() {

	// Performed here instead of as part of execute() for rare case where app destroys queue
	// immediately after a waitIdle() is cleared by fence below, taking the capture scope with it.
	_queue->_submissionCaptureScope->endScope();

	// If using inline semaphore signaling, do so now.
	for (auto& ss : _signalSemaphores) { ss.encodeSignal(nil); }

	// If a fence exists, signal it.
	if (_fence) { _fence->signal(); }

	this->destroy();
}

// On device loss, the fence and signal semaphores may be signalled early, and they might then
// be destroyed on the waiting thread before this submission is done with them. We therefore
// retain() each here to ensure they live long enough for this submission to finish using them.
MVKQueueCommandBufferSubmission::MVKQueueCommandBufferSubmission(MVKQueue* queue,
																 const VkSubmitInfo2* pSubmit,
																 VkFence fence,
																 MVKCommandUse cmdUse) :
	MVKQueueSubmission(queue,
					   pSubmit ? pSubmit->waitSemaphoreInfoCount : 0,
					   pSubmit ? pSubmit->pWaitSemaphoreInfos : nullptr),
	_fence((MVKFence*)fence),
	_commandUse(cmdUse) {
	
	if (_fence) { _fence->retain(); }

	// pSubmit can be null if just tracking the fence alone
	if (pSubmit) {
		uint32_t ssCnt = pSubmit->signalSemaphoreInfoCount;
		_signalSemaphores.reserve(ssCnt);
		for (uint32_t i = 0; i < ssCnt; i++) {
			_signalSemaphores.emplace_back(pSubmit->pSignalSemaphoreInfos[i]);
		}
	}
}

// On device loss, the fence and signal semaphores may be signalled early, and they might then
// be destroyed on the waiting thread before this submission is done with them. We therefore
// retain() each here to ensure they live long enough for this submission to finish using them.
MVKQueueCommandBufferSubmission::MVKQueueCommandBufferSubmission(MVKQueue* queue,
																 const VkSubmitInfo* pSubmit,
																 VkFence fence,
																 MVKCommandUse cmdUse)
	: MVKQueueSubmission(queue,
						 pSubmit ? pSubmit->waitSemaphoreCount : 0,
						 pSubmit ? pSubmit->pWaitSemaphores : nullptr,
						 pSubmit ? pSubmit->pWaitDstStageMask : nullptr),

	_fence((MVKFence*)fence),
	_commandUse(cmdUse) {
	
	if (_fence) { _fence->retain(); }

    // pSubmit can be null if just tracking the fence alone
    if (pSubmit) {
		uint32_t ssCnt = pSubmit->signalSemaphoreCount;
		_signalSemaphores.reserve(ssCnt);
		for (uint32_t i = 0; i < ssCnt; i++) {
			_signalSemaphores.emplace_back(pSubmit->pSignalSemaphores[i], 0);
		}

		VkTimelineSemaphoreSubmitInfo* pTimelineSubmit = nullptr;
        for (const auto* next = (const VkBaseInStructure*)pSubmit->pNext; next; next = next->pNext) {
            switch (next->sType) {
                case VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO:
                    pTimelineSubmit = (VkTimelineSemaphoreSubmitInfo*)next;
                    break;
                default:
                    break;
            }
        }
        if (pTimelineSubmit) {
            uint32_t wsvCnt = pTimelineSubmit->waitSemaphoreValueCount;
            for (uint32_t i = 0; i < wsvCnt; i++) {
                _waitSemaphores[i].value = pTimelineSubmit->pWaitSemaphoreValues[i];
            }

			uint32_t ssvCnt = pTimelineSubmit->signalSemaphoreValueCount;
			for (uint32_t i = 0; i < ssvCnt; i++) {
				_signalSemaphores[i].value = pTimelineSubmit->pSignalSemaphoreValues[i];
			}
        }
    }
}

MVKQueueCommandBufferSubmission::MVKQueueCommandBufferSubmission(MVKQueue* queue,
																 const VkBindSparseInfo* pBindInfo,
																 VkFence fence,
																 MVKCommandUse cmdUse)
	: MVKQueueSubmission(queue,
						 pBindInfo ? pBindInfo->waitSemaphoreCount : 0,
						 pBindInfo ? pBindInfo->pWaitSemaphores : nullptr,
						 nullptr),
	  _fence((MVKFence*)fence),
	  _commandUse(cmdUse) {
	if (_fence) { _fence->retain(); }

	if (pBindInfo) {
		_signalSemaphores.reserve(pBindInfo->signalSemaphoreCount);
		for (uint32_t i = 0; i < pBindInfo->signalSemaphoreCount; i++) {
			_signalSemaphores.emplace_back(pBindInfo->pSignalSemaphores[i], 0);
		}

		const VkTimelineSemaphoreSubmitInfo* timelineInfo = nullptr;
		for (const auto* next = (const VkBaseInStructure*)pBindInfo->pNext; next; next = next->pNext) {
			if (next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
				timelineInfo = (const VkTimelineSemaphoreSubmitInfo*)next;
			}
		}
		if (timelineInfo) {
			for (uint32_t i = 0; i < pBindInfo->waitSemaphoreCount; i++) {
				_waitSemaphores[i].value = timelineInfo->pWaitSemaphoreValues[i];
			}
			for (uint32_t i = 0; i < pBindInfo->signalSemaphoreCount; i++) {
				_signalSemaphores[i].value = timelineInfo->pSignalSemaphoreValues[i];
			}
		}
	}

}

MVKQueueCommandBufferSubmission::~MVKQueueCommandBufferSubmission() {
	if (_fence) { _fence->release(); }
}

#if MVK_XCODE_26 && MVK_MACOS
VkResult MVKQueueSparseSubmission::execute() {
	if (@available(macOS 26.4, *)) {
		_queue->_submissionCaptureScope->beginScope();

		id<MTLCommandBuffer> tail = _queue->getMTLCommandBuffer(_commandUse, true);
		id<MTLCommandBuffer> head = tail ? getActiveMTLCommandBuffer() : nil;
		if (!head || !tail) {
			setConfigurationResult(VK_ERROR_OUT_OF_POOL_MEMORY);
			VkResult result = getConfigurationResult();
			finish();
			return result;
		}

		for (auto& wait : _waitSemaphores) { wait.encodeWait(head); }
		addPerformanceInterval(_queue->getPerformanceStats().queue.waitSubmitCommandBuffers, _creationTime);
		[head encodeSignalEvent:_queue->_mtl4SparseEvent value:_startEventValue];
		VkResult result = commitActiveMTLCommandBuffer();
		if (result != VK_SUCCESS) {
			finish();
			return result;
		}

		[_queue->_mtl4SparseQueue waitForEvent:_queue->_mtl4SparseEvent value:_startEventValue];
		for (const auto& bind : _bufferBinds) {
			MTL4UpdateSparseBufferMappingOperation operation = {
				.mode = bind.memory ? MTLSparseTextureMappingModeMap : MTLSparseTextureMappingModeUnmap,
				.bufferRange = NSMakeRange((NSUInteger)(bind.resourceOffset / kMVKPrivateSparseBufferPageSize),
										  (NSUInteger)(bind.size / kMVKPrivateSparseBufferPageSize)),
				.heapOffset = (NSUInteger)(bind.memoryOffset / kMVKPrivateSparseBufferPageSize),
			};
			[_queue->_mtl4SparseQueue updateBufferMappings:bind.mtlBuffer
													 heap:bind.mtlHeap
											   operations:&operation
													count:1];
		}
		for (const auto& bind : _imageBinds) {
			MTL4UpdateSparseTextureMappingOperation operation = {
				.mode = bind.isMap ? MTLSparseTextureMappingModeMap : MTLSparseTextureMappingModeUnmap,
				.textureRegion = bind.tileRegion,
				.textureLevel = bind.textureLevel,
				.textureSlice = bind.textureSlice,
				.heapOffset = bind.heapOffset,
			};
			[_queue->_mtl4SparseQueue updateTextureMappings:bind.mtlTexture
													  heap:bind.mtlHeap
												operations:&operation
													 count:1];
		}

		[_mtl4BarrierBuffer beginCommandBufferWithAllocator:_mtl4Allocator];
		id<MTL4ComputeCommandEncoder> barrier = [_mtl4BarrierBuffer computeCommandEncoder];
		if (!barrier) {
			setConfigurationResult(VK_ERROR_OUT_OF_DEVICE_MEMORY);
			_queue->_mtl4SparseEvent.signaledValue = _doneEventValue;
			_queue->getDevice()->markLost(false);
		} else {
			[barrier barrierAfterQueueStages:MTLStageResourceState
							   beforeStages:MTLStageAll
						  visibilityOptions:MTL4VisibilityOptionResourceAlias];
			[barrier endEncoding];
			[_mtl4BarrierBuffer endCommandBuffer];
		}

		MVKQueue* retainedQueue = _queue;
		if (barrier) {
			retainedQueue->retain();
			id<MTLSharedEvent> retainedEvent = [_queue->_mtl4SparseEvent retain];
			uint64_t doneEventValue = _doneEventValue;
			MTL4CommitOptions* options = [MTL4CommitOptions new];
			[options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
				if (feedback.error) {
					if (retainedEvent.signaledValue < doneEventValue) {
						retainedEvent.signaledValue = doneEventValue;
					}
					retainedQueue->getDevice()->markLost(false);
					retainedQueue->reportError(VK_ERROR_DEVICE_LOST,
						"vkQueueBindSparse(): Metal sparse mapping failed: %s",
						feedback.error.localizedDescription.UTF8String);
				}
				[retainedEvent release];
				retainedQueue->release();
			}];
			id<MTL4CommandBuffer> barrierBuffers[] = { _mtl4BarrierBuffer };
			[_queue->_mtl4SparseQueue commit:barrierBuffers count:1 options:options];
			[options release];
			[_queue->_mtl4SparseQueue signalEvent:_queue->_mtl4SparseEvent value:_doneEventValue];
		}

		setActiveMTLCommandBuffer(tail);
		[tail encodeWaitForEvent:_queue->_mtl4SparseEvent value:_doneEventValue];
		for (auto& signal : _signalSemaphores) { signal.encodeSignal(tail); }
		return commitActiveMTLCommandBuffer(true);
	}

	setConfigurationResult(VK_ERROR_FEATURE_NOT_PRESENT);
	VkResult result = getConfigurationResult();
	finish();
	return result;
}

MVKQueueSparseSubmission::MVKQueueSparseSubmission(MVKQueue* queue,
												   const VkBindSparseInfo* pBindInfo,
												   VkFence fence,
												   uint64_t startEventValue,
												   uint64_t doneEventValue)
	: MVKQueueCommandBufferSubmission(queue, pBindInfo, fence, kMVKCommandUseQueueSubmit),
	  _startEventValue(startEventValue),
	  _doneEventValue(doneEventValue) {
	if (@available(macOS 26.4, *)) {
		for (uint32_t bufferIdx = 0; bufferIdx < pBindInfo->bufferBindCount; bufferIdx++) {
			const VkSparseBufferMemoryBindInfo& bufferInfo = pBindInfo->pBufferBinds[bufferIdx];
			auto* buffer = (MVKBuffer*)bufferInfo.buffer;
			for (uint32_t bindIdx = 0; bindIdx < bufferInfo.bindCount; bindIdx++) {
				const VkSparseMemoryBind& bind = bufferInfo.pBinds[bindIdx];
				auto* memory = (MVKDeviceMemory*)bind.memory;
				bool coalesce = !_bufferBinds.empty() &&
					_bufferBinds.back().buffer == buffer &&
					_bufferBinds.back().memory == memory &&
					_bufferBinds.back().resourceOffset + _bufferBinds.back().size == bind.resourceOffset &&
					(!memory || _bufferBinds.back().memoryOffset + _bufferBinds.back().size == bind.memoryOffset);
				if (coalesce) {
					_bufferBinds.back().size += bind.size;
					continue;
				}

				MVKSparseBufferBind ownedBind = {
					.buffer = buffer,
					.memory = memory,
					.mtlBuffer = [buffer->getMTLBuffer() retain],
					.mtlHeap = memory ? [memory->getMTLPlacementSparseHeap() retain] : nil,
					.resourceOffset = bind.resourceOffset,
					.memoryOffset = memory ? bind.memoryOffset : 0,
					.size = bind.size,
				};
				buffer->retain();
				if (memory) { memory->retain(); }
				_bufferBinds.push_back(ownedBind);
			}
		}

		for (uint32_t opaqueIdx = 0; opaqueIdx < pBindInfo->imageOpaqueBindCount; opaqueIdx++) {
			const VkSparseImageOpaqueMemoryBindInfo& opaqueInfo = pBindInfo->pImageOpaqueBinds[opaqueIdx];
			auto* image = (MVKImage*)opaqueInfo.image;
			VkDeviceSize tileByteCount = image->getSparseTileByteCount();
			for (uint32_t bindIdx = 0; bindIdx < opaqueInfo.bindCount; bindIdx++) {
				const VkSparseMemoryBind& bind = opaqueInfo.pBinds[bindIdx];
				auto* memory = (MVKDeviceMemory*)bind.memory;
				// Validation guaranteed the bind covers the whole mip tail of one layer.
				// Metal maps a slice tail as a linear run of tiles at the tail mip level.
				MVKSparseImageBind ownedBind = {
					.image = image,
					.memory = memory,
					.mtlTexture = [image->getMTLTexture(0) retain],
					.mtlHeap = memory ? [memory->getMTLPlacementSparseHeap() retain] : nil,
					.tileRegion = MTLRegionMake3D(0, 0, 0, (NSUInteger)(bind.size / tileByteCount), 1, 1),
					.textureLevel = (NSUInteger)image->getMipTailFirstLod(),
					.textureSlice = (NSUInteger)(bind.resourceOffset / bind.size),
					.heapOffset = (NSUInteger)(memory ? bind.memoryOffset / tileByteCount : 0),
					.isMap = memory != nullptr,
				};
				image->retain();
				if (memory) { memory->retain(); }
				_imageBinds.push_back(ownedBind);
			}
		}

		for (uint32_t imageIdx = 0; imageIdx < pBindInfo->imageBindCount; imageIdx++) {
			const VkSparseImageMemoryBindInfo& imageInfo = pBindInfo->pImageBinds[imageIdx];
			auto* image = (MVKImage*)imageInfo.image;
			VkExtent3D granularity = image->getSparseTileExtent();
			VkDeviceSize tileByteCount = image->getSparseTileByteCount();
			for (uint32_t bindIdx = 0; bindIdx < imageInfo.bindCount; bindIdx++) {
				const VkSparseImageMemoryBind& bind = imageInfo.pBinds[bindIdx];
				auto* memory = (MVKDeviceMemory*)bind.memory;
				NSUInteger tileX = bind.offset.x / granularity.width;
				NSUInteger tileY = bind.offset.y / granularity.height;
				NSUInteger tileZ = bind.offset.z / granularity.depth;
				// Validation guaranteed aligned offsets and mip-edge-clamped extents.
				MVKSparseImageBind ownedBind = {
					.image = image,
					.memory = memory,
					.mtlTexture = [image->getMTLTexture(0) retain],
					.mtlHeap = memory ? [memory->getMTLPlacementSparseHeap() retain] : nil,
					.tileRegion = MTLRegionMake3D(tileX, tileY, tileZ,
												 (bind.offset.x + bind.extent.width + granularity.width - 1) / granularity.width - tileX,
												 (bind.offset.y + bind.extent.height + granularity.height - 1) / granularity.height - tileY,
												 (bind.offset.z + bind.extent.depth + granularity.depth - 1) / granularity.depth - tileZ),
					.textureLevel = (NSUInteger)bind.subresource.mipLevel,
					.textureSlice = (NSUInteger)bind.subresource.arrayLayer,
					.heapOffset = (NSUInteger)(memory ? bind.memoryOffset / tileByteCount : 0),
					.isMap = memory != nullptr,
				};
				image->retain();
				if (memory) { memory->retain(); }
				_imageBinds.push_back(ownedBind);
			}
		}

		_mtl4Allocator = [getMTLDevice() newCommandAllocator];
		_mtl4BarrierBuffer = [getMTLDevice() newCommandBuffer];
		if (!_mtl4Allocator || !_mtl4BarrierBuffer) {
			setConfigurationResult(reportError(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkQueueBindSparse(): Could not create the Metal sparse barrier."));
			return;
		}
	}
}

MVKQueueSparseSubmission::~MVKQueueSparseSubmission() {
	for (auto& bind : _bufferBinds) {
		[bind.mtlBuffer release];
		[bind.mtlHeap release];
		if (bind.memory) { bind.memory->release(); }
		bind.buffer->release();
	}
	for (auto& bind : _imageBinds) {
		[bind.mtlTexture release];
		[bind.mtlHeap release];
		if (bind.memory) { bind.memory->release(); }
		bind.image->release();
	}
	[_mtl4BarrierBuffer release];
	[_mtl4Allocator release];
}
#endif

template <size_t N>
void MVKQueueFullCommandBufferSubmission<N>::submitCommandBuffers() {
	uint64_t startTime = getPerformanceTimestamp();

	for (auto& cbInfo : _cmdBuffers) { cbInfo.commandBuffer->submit(this, &_encodingContext); }

	addPerformanceInterval(getPerformanceStats().queue.submitCommandBuffers, startTime);
}

template <size_t N>
bool MVKQueueFullCommandBufferSubmission<N>::requiresHostReadback() const {
	for (auto& cbInfo : _cmdBuffers) {
		if (cbInfo.commandBuffer->requiresHostReadback()) { return true; }
	}
	return false;
}

template <size_t N>
bool MVKQueueFullCommandBufferSubmission<N>::requiresEncodingDependencyWait() {
	bool hasDependentCommandBuffers = false;
	for (auto& cbInfo : _cmdBuffers) {
		hasDependentCommandBuffers |= cbInfo.commandBuffer->requiresEncodingDependencyWait();
	}
	// MageVK: MAGE_MVK_RT_DEP_WAIT_NARROW=1 also drops the sparse-buffer clause,
	// which otherwise forces the CPU-blocking GPU drain on every submit when
	// private sparse buffers are enabled. Experimental; A/B only.
	static int rtDepWaitNarrow = -1;
	if (rtDepWaitNarrow < 0) {
		const char* rdwEnv = getenv("MAGE_MVK_RT_DEP_WAIT_NARROW");
		rtDepWaitNarrow = (rdwEnv && !strcmp(rdwEnv, "1")) ? 1 : 0;
	}
	return !_waitSemaphores.empty() &&
		   ((getEnabledAccelerationStructureFeatures().accelerationStructure && hasDependentCommandBuffers) ||
			(!rtDepWaitNarrow && _device->isPrivateSparseBufferEnabled()));
}

template <size_t N>
MVKQueueFullCommandBufferSubmission<N>::MVKQueueFullCommandBufferSubmission(MVKQueue* queue,
																			const VkSubmitInfo2* pSubmit,
																			VkFence fence,
																			MVKCommandUse cmdUse)
	: MVKQueueCommandBufferSubmission(queue, pSubmit, fence, cmdUse) {

	if (pSubmit) {
		uint32_t cbCnt = pSubmit->commandBufferInfoCount;
		_cmdBuffers.reserve(cbCnt);
		for (uint32_t i = 0; i < cbCnt; i++) {
			_cmdBuffers.emplace_back(pSubmit->pCommandBufferInfos[i]);
			setConfigurationResult(_cmdBuffers.back().commandBuffer->getConfigurationResult());
		}
	}
}

template <size_t N>
MVKQueueFullCommandBufferSubmission<N>::MVKQueueFullCommandBufferSubmission(MVKQueue* queue,
																			const VkSubmitInfo* pSubmit,
																			VkFence fence,
																			MVKCommandUse cmdUse)
	: MVKQueueCommandBufferSubmission(queue, pSubmit, fence, cmdUse) {

	if (pSubmit) {
		uint32_t cbCnt = pSubmit->commandBufferCount;
		_cmdBuffers.reserve(cbCnt);
		for (uint32_t i = 0; i < cbCnt; i++) {
			_cmdBuffers.emplace_back(pSubmit->pCommandBuffers[i]);
			setConfigurationResult(_cmdBuffers.back().commandBuffer->getConfigurationResult());
		}
	}
}


#pragma mark -
#pragma mark MVKQueuePresentSurfaceSubmission

// If the semaphores are encodable, wait on them by encoding them on the MTLCommandBuffer before presenting.
// If the semaphores are not encodable, wait on them inline after presenting.
// The semaphores know what to do.
VkResult MVKQueuePresentSurfaceSubmission::execute() {
	// MTLCommandBuffer retain references to avoid rare case where objects are destroyed too early.
	// Although testing could not determine which objects were being lost, queue present MTLCommandBuffers
	// are used only once per frame, and retain so few objects, that blanket retention is still performant.
	id<MTLCommandBuffer> mtlCmdBuff = _queue->getMTLCommandBuffer(kMVKCommandUseQueuePresent, true);

	for (auto& ws : _waitSemaphores) {
		ws.encodeWait(mtlCmdBuff);	// Encoded semaphore waits
		ws.encodeWait(nil);			// Inline semaphore waits
	}

	// Wait time from an async vkQueuePresentKHR() call to starting presentation of the swapchains
	addPerformanceInterval(getPerformanceStats().queue.waitPresentSwapchains, _creationTime);

	for (int i = 0; i < _presentInfo.size(); i++ ) {
		setConfigurationResult(_presentInfo[i].presentableImage->presentCAMetalDrawable(mtlCmdBuff, _presentInfo[i]));
	}

	if (_queue->_queueFamily->getIndex() == getMVKConfig().defaultGPUCaptureScopeQueueFamilyIndex &&
		_queue->_index == getMVKConfig().defaultGPUCaptureScopeQueueIndex) {
		getDevice()->stopAutoGPUCapture(MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE_ON_DEMAND);
		getDevice()->startAutoGPUCapture(MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE_ON_DEMAND, _queue->getMTLCommandQueue());
	}

	if ( !mtlCmdBuff ) { setConfigurationResult(VK_ERROR_OUT_OF_POOL_MEMORY); }	// Check after images may set error.

	// Add completion callback to the MTLCommandBuffer to call finish(), 
	// or if the MTLCommandBuffer could not be created, call finish() directly.
	// Retrieve the result first, because finish() will destroy this instance.
	VkResult rslt = getConfigurationResult();
	if (mtlCmdBuff) {
		[mtlCmdBuff addCompletedHandler: ^(id<MTLCommandBuffer> mtlCB) { this->finish(); }];
		[mtlCmdBuff commit];
	} else {
		finish();
	}
	return rslt;
}

void MVKQueuePresentSurfaceSubmission::finish() {

	// Let Xcode know the current frame is done, then start a new frame,
	// and if auto GPU capture is active, and it's time to stop it, do so.
	auto cs = _queue->_submissionCaptureScope;
	cs->endScope();
	cs->beginScope();
	if (_queue->_queueFamily->getIndex() == getMVKConfig().defaultGPUCaptureScopeQueueFamilyIndex &&
		_queue->_index == getMVKConfig().defaultGPUCaptureScopeQueueIndex) {
		getDevice()->stopAutoGPUCapture(MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE_FRAME);
	}

	this->destroy();
}

MVKQueuePresentSurfaceSubmission::MVKQueuePresentSurfaceSubmission(MVKQueue* queue,
																   const VkPresentInfoKHR* pPresentInfo)
	: MVKQueueSubmission(queue, pPresentInfo->waitSemaphoreCount, pPresentInfo->pWaitSemaphores, nullptr) {

	const VkPresentTimesInfoGOOGLE* pPresentTimesInfo = nullptr;
	const VkSwapchainPresentFenceInfoKHR* pPresentFenceInfo = nullptr;
	const VkSwapchainPresentModeInfoKHR* pPresentModeInfo = nullptr;
	const VkPresentRegionsKHR* pPresentRegions = nullptr;
	const VkPresentIdKHR* pPresentId = nullptr;
	const VkPresentId2KHR* pPresentId2 = nullptr;
	for (auto* next = (const VkBaseInStructure*)pPresentInfo->pNext; next; next = next->pNext) {
		switch (next->sType) {
			case VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR:
				pPresentRegions = (const VkPresentRegionsKHR*) next;
				break;
			case VK_STRUCTURE_TYPE_PRESENT_ID_KHR:
				pPresentId = (const VkPresentIdKHR*) next;
				break;
			case VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR:
				pPresentId2 = (const VkPresentId2KHR*) next;
				break;
			case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR:
				pPresentFenceInfo = (const VkSwapchainPresentFenceInfoKHR*) next;
				break;
			case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR:
				pPresentModeInfo = (const VkSwapchainPresentModeInfoKHR*) next;
				break;
			case VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE:
				pPresentTimesInfo = (const VkPresentTimesInfoGOOGLE*) next;
				break;
			default:
				break;
		}
	}

	// Populate the array of swapchain images, testing each one for status
	uint32_t scCnt = pPresentInfo->swapchainCount;
	const VkPresentTimeGOOGLE* pPresentTimes = nullptr;
	if (pPresentTimesInfo) {
		pPresentTimes = pPresentTimesInfo->pTimes;
		MVKAssert(pPresentTimesInfo->swapchainCount == scCnt, "VkPresentTimesInfoGOOGLE swapchainCount must match VkPresentInfo swapchainCount.");
	}
	const VkPresentModeKHR* pPresentModes = nullptr;
	if (pPresentModeInfo) {
		pPresentModes = pPresentModeInfo->pPresentModes;
		MVKAssert(pPresentModeInfo->swapchainCount == scCnt, "VkSwapchainPresentModeInfoKHR swapchainCount must match VkPresentInfo swapchainCount.");
	}
	const VkFence* pFences = nullptr;
	if (pPresentFenceInfo) {
		pFences = pPresentFenceInfo->pFences;
		MVKAssert(pPresentFenceInfo->swapchainCount == scCnt, "VkSwapchainPresentFenceInfoKHR swapchainCount must match VkPresentInfo swapchainCount.");
	}
	const VkPresentRegionKHR* pRegions = nullptr;
	if (pPresentRegions) {
		pRegions = pPresentRegions->pRegions;
	}
	const uint64_t* pPresentIds = nullptr;
	if (pPresentId2) {
		pPresentIds = pPresentId2->pPresentIds;
	} else if (pPresentId) {
		pPresentIds = pPresentId->pPresentIds;
	}

	VkResult* pSCRslts = pPresentInfo->pResults;
	_presentInfo.reserve(scCnt);
	for (uint32_t scIdx = 0; scIdx < scCnt; scIdx++) {
		MVKSwapchain* mvkSC = (MVKSwapchain*)pPresentInfo->pSwapchains[scIdx];
		MVKImagePresentInfo presentInfo = {};	// Start with everything zeroed
		presentInfo.queue = _queue;
		presentInfo.presentableImage = mvkSC->getPresentableImage(pPresentInfo->pImageIndices[scIdx]);
		presentInfo.presentMode = pPresentModes ? pPresentModes[scIdx] : VK_PRESENT_MODE_MAX_ENUM_KHR;
		presentInfo.fence = pFences ? (MVKFence*)pFences[scIdx] : nullptr;
		presentInfo.presentId = pPresentIds ? pPresentIds[scIdx] : 0;
		if (pPresentTimes) {
			presentInfo.presentIDGoogle = pPresentTimes[scIdx].presentID;
			presentInfo.desiredPresentTime = pPresentTimes[scIdx].desiredPresentTime;
		}
		mvkSC->setLayerNeedsDisplay(pRegions ? &pRegions[scIdx] : nullptr);
		_presentInfo.push_back(presentInfo);
		VkResult scRslt = mvkSC->getSurfaceStatus();
		if (pSCRslts) { pSCRslts[scIdx] = scRslt; }
		setConfigurationResult(scRslt);
	}
}
