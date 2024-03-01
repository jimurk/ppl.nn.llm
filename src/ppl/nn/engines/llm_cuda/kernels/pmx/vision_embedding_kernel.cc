// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "vision_embedding_kernel.h"

#include "ppl/common/cuda/nccl_utils.h"
#include "ppl/common/destructor.h"

// #include "ppl/kernel/llm/cuda/pmx/vision_embedding.h"//TODO:

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {


ppl::common::RetCode VisionEmbeddingKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(pixel_values, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(cls_emb_weight, 1);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(patch_emb_weight, 2);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(pos_emb_weight, 3);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(vision_embeddings, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [pixel_values]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(pixel_values);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [cls_emb_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(cls_emb_weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [patch_emb_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(patch_emb_weight);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [pos_emb_weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(pos_emb_weight);

    PPLNN_LLM_CUDA_DEBUG_TRACE("hidden_dim: %d\n", param_->hidden_dim);
    PPLNN_LLM_CUDA_DEBUG_TRACE("image_size: %d\n", param_->image_size);
    PPLNN_LLM_CUDA_DEBUG_TRACE("patch_size: %d\n", param_->patch_size);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(vision_embeddings);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [vision_embeddings]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(vision_embeddings);

    //TODO:
    // return ppl::kernel::llm::cuda::pmx::column_parallel_linear(
    //     GetStream(),
    //     cublas_handle,
    //     nullptr,
    //     input_shape,
    //     input->GetBufferPtr(),
    //     weight_shape,
    //     weight->GetBufferPtr(),
    //     bias_shape,
    //     bias_data,
    //     param_->in_features,
    //     param_->out_features,
    //     nccl_param,
    //     param_->gather_output,
    //     gather_buffer,
    //     use_workspace ? GetCudaDevice()->GetCublasWorkspaceSize() : 0,
    //     use_workspace ? GetCudaDevice()->GetCublasWorkspace() : nullptr,
    //     output_shape,
    //     output->GetBufferPtr()
    // );
    return ppl::common::RC_UNSUPPORTED;
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
