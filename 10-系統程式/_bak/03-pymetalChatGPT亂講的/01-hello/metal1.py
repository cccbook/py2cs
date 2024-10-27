import metal as mtl
import numpy as np

# 初始化 Metal 物件
device = mtl.MTLCreateSystemDefaultDevice()
command_queue = device.newCommandQueue()

# Metal shader 程式碼，用於向量加法
kernel_code = """
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(const device float* inA [[ buffer(0) ]],
                       const device float* inB [[ buffer(1) ]],
                       device float* result [[ buffer(2) ]],
                       uint id [[ thread_position_in_grid ]]) {
    result[id] = inA[id] + inB[id];
}
"""

# 建立 Metal 資源
library = device.newLibraryWithSource_options_error_(kernel_code, None, None)
kernel_function = library.newFunctionWithName_("add_arrays")
pipeline_state = device.newComputePipelineStateWithFunction_error_(kernel_function, None)

# 向量資料
n = 1024
inA = np.random.rand(n).astype(np.float32)
inB = np.random.rand(n).astype(np.float32)
result = np.zeros(n, dtype=np.float32)

# 建立 Metal buffer
buffer_inA = device.newBufferWithBytes_length_options_(inA.tobytes(), inA.nbytes, mtl.MTLResourceStorageModeShared)
buffer_inB = device.newBufferWithBytes_length_options_(inB.tobytes(), inB.nbytes, mtl.MTLResourceStorageModeShared)
buffer_result = device.newBufferWithBytes_length_options_(result.tobytes(), result.nbytes, mtl.MTLResourceStorageModeShared)

# 設定計算命令
command_buffer = command_queue.commandBuffer()
command_encoder = command_buffer.computeCommandEncoder()
command_encoder.setComputePipelineState_(pipeline_state)
command_encoder.setBuffer_offset_atIndex_(buffer_inA, 0, 0)
command_encoder.setBuffer_offset_atIndex_(buffer_inB, 0, 1)
command_encoder.setBuffer_offset_atIndex_(buffer_result, 0, 2)

# 設定執行執行緒數量
thread_group_size = 32
grid_size = (n + thread_group_size - 1) // thread_group_size
command_encoder.dispatchThreadgroups_threadsPerThreadgroup_((grid_size, 1, 1), (thread_group_size, 1, 1))
command_encoder.endEncoding()

# 執行命令
command_buffer.commit()
command_buffer.waitUntilCompleted()

# 取得結果
result = np.frombuffer(buffer_result.contents(), dtype=np.float32)

# 驗證結果
print("計算成功:", np.allclose(result, inA + inB))
