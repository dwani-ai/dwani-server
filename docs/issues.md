2025-03-17 22:33:24,340 - parler_tts.modeling_parler_tts - WARNING - `prompt_attention_mask` is specified but `attention_mask` is not. A full `attention_mask` will be created. Make sure this is the intended behaviour.
W0317 22:33:36.322000 1 torch/_inductor/utils.py:1137] [0/0] Not enough SMs to use max_autotune_gemm mode
CUDAGraph supports dynamic shapes by recording a new graph for each distinct input size. Recording too many CUDAGraphs may lead to extra overhead. We have observed 51 distinct sizes. Please consider the following options for better performance: a) padding inputs to a few fixed number of shapes; or b) set torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None to silence this warning.
 

 