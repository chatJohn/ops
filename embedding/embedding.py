import time
from functools import partial
from typing import Optional
import torch
from torch.nn.functional import embedding
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load CUDA kernel as python module
lib = load(
    name = 'embedding', # TORCH_EXTENSION_NAME binding
    sources = '../embedding.cu',
    extra_cuda_cflags = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags = ['-std=c++17'],
)

def run_benchmark(
        perf_func: callable,
        a: torch.Tensor,
        b: torch.Tensor,
        tag: str,
        out: Optional[torch.Tensor] = None, # the type of `out` is `torch.Tensr` or `None`
        warmup: int = 2,
        iters: int = 20,
        show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>23}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time

