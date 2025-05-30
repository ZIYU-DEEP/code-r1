# Code-R1: Reproducing R1 for Code with Reliable Rewards

This repository includes implementations to reproduce the R1 pipeline for code generation:

* **Result:** 2K Code-R1 samples + `Qwen2.5-7B-Instruct-1M` beats `Qwen2.5-Coder-7B-Instruct` (even better w/ 12K samples).
* **Finding:** Quality of rewards matters. False positives in datasets and execution confuse the model.
* **Implementation:** A reliable, scalable, and sandboxed pipeline to minimize reward false positives in datasets and execution.

More results and findings to come...

## Setup

### Environment

```bash
# For training
git submodule update --init --recursive
pip install -e .
pip install vllm==0.7.3
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib gpustat # utility
```

Update:
```bash
pip install vllm==0.8.2
pip install tensordict==0.6.2
```

### Sandboxing

I tried multiple ways for sandboxing including calling code execution servers, running dockerized Python, calling paid services, etc.
`firejail` is the approach I found to meet all the three:

1. Reliability -- False positive comes when "autograders" have concurrency issue (timeouts), violating OS limits (`OSError`), etc.
2. Scalability -- e.g., dockerized Python run is generally reliable but too slow (e.g., 20 samples/s on 192 cores).
3. Security -- ... otherwise the school IT will email you and stop your server...

```bash
sudo add-apt-repository ppa:deki/firejail
sudo apt-get update
sudo apt-get install firejail firejail-profiles
```

Update: The above can lead to errors. Let's build from the source instead:
```bash
git clone https://github.com/netblue30/firejail
cd firejail
sudo apt-get install gawk -y
chmod +x ./configure
chmod +x src/man/mkman.sh
./configure && make && sudo make install-strip
```

Test the functionality of the sandbox:
```bash
python scripts/stress_exec.py
python scripts/test_sandbox.py
```

### Datasets

The current version has 12K RL samples (prompt + tests) at [🤗 ganler/code-r1-12k](https://huggingface.co/datasets/ganler/code-r1-12k):

* [2K LeetCode data](https://github.com/newfacade/LeetCodeDataset) where the tests are generally reliable
* 10K verified data filtered from 26K [TACO](https://huggingface.co/datasets/BAAI/TACO) data.

In general, it's suggesgted to test data & sandbox on every dataset & environment before training code RL.
Directly using noisy data and mismatched envornments can lead to reward false positives, confusing the model.
These noise could come from (i) wrong tests, (ii) unsolvable prompts (e.g., images tags), and (iii) execution environment mismatch.

To produce locally validated RL data:

```bash
python examples/data_preprocess/coder1.py
```

One may directly download from huggingface:
```python
import os
from datasets import load_dataset

target_folder = "data/code-r1-12k"
os.makedirs(target_folder, exist_ok=True)

dataset = load_dataset("ganler/code-r1-12k")
dataset["train"].to_parquet(f"{target_folder}/train.parquet")
dataset["test"].to_parquet(f"{target_folder}/test.parquet")
```

Download huggingface models:
```bash
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-7B-Instruct-1M')"
```

### Run!

```bash
bash main_grpo.sh
```

> [!NOTE]
>
> The script was optimized for single-node 8x H200 setup. You might need to customize the settings for your own workstation.

## Code-R1 Zero based on 7B models

We trained two models based on Qwen2.5-7B-Instruct-1M by pure R1 Zero:
* [🤗 CodeR1-Zero-Qwen2.5-7B-12k-832](https://huggingface.co/ganler/CodeR1-Zero-Qwen2.5-7B-12k-832): using 12K RL samples trained in 832 steps ([training logs](https://api.wandb.ai/links/llm4code/y13vs8d9)).
* [🤗 CodeR1-Zero-Qwen2.5-7B-LC2k-1088](https://huggingface.co/ganler/CodeR1-Zero-Qwen2.5-7B-LC2k-1088): using 2K RL samples from LeetCode,  trained in 1088 steps ([training logs](https://api.wandb.ai/links/llm4code/k8q6zu51)).

|                    Model                       |     LCB (v5)  |   HumanEval+   |    MBPP+    | **Average** |
|------------------------------------------------|---------------|----------------|-------------|------------:|
| Qwen2.5-7B-Instruct-1M                         |     24.0      |     80.5       |    66.7     |   57.1      |
| + Code-R1-Zero (2k  - 1088s GRPO)              |     28.6      |     84.8       |    70.1     |   61.2      |
| + Code-R1-Zero (12k -  832s GRPO)              |     29.7      |     83.5       |    74.3     | 🌟**62.5**  |

* 2K leetcode training samples can already show promising results without any additional SFT or distillation.
* Adding it to 12K data (10K more verified data from TACO) can further improve the performance.

Some existing models:

|                    Model                       |     LCB (v5)  |   HumanEval+   |    MBPP+    | **Average** |
|------------------------------------------------|---------------|----------------|-------------|------------:|
| Qwen2.5-Coder-7B-Instruct                      |     31.1      |     82.3       |    69.6     |  61.0       |
| Eurus-2-7B-PRIME                               |     23.8      |     65.9       |    29.9     |  39.9       |
| Sky-T1-7B                                      |     21.3      |     54.3       |    50.8     |  42.1       |

* Qwen2.5-Coder-7B-Instruct, despite released months back, is still very performant as the best baseline, but we don't know where the improvement comes from.
* Eurus-2-7B-PRIME starts from Qwen2.5-Math-7B-Instruct and is RL only. Its training data includes (unfiltered) extensive coding datasets, including APPS, CodeContests, TACO, and Codeforces. Code-R1-Zero outperforms it significantly despite using fewer data, likely because we use validated datasets and sandboxes.
* Sky-T1-7B uses a combination of RL and SFT/distillation steps. Its RL partially uses PRIME but its training data does not (seem to) include coding datasets.

## Citation

If you find this work helpful...

```bibtex
@article{code-r1,
  title={Code-R1: Reproducing R1 for Code with Reliable Rewards},
  author={Liu, Jiawei and Zhang, Lingming},
  howpublished={\url{https://github.com/ganler/code-r1}},
  year={2025}
}
```

## Acknowledgements

* [Verl](https://github.com/volcengine/verl)
* [Logic-RL](https://github.com/Unakar/Logic-RL)

## License

Apache-2.0. See [LICENSE.code-r1](LICENSE.code-r1) for more details.

## Troubleshooting

### Issues on `firejail`
Building from the source:
```bash
git clone https://github.com/netblue30/firejail
cd firejail
sudo apt-get install gawk -y
chmod +x ./configure
chmod +x src/man/mkman.sh
./configure && make && sudo make install-strip
```
Then test with:
```bash
python scripts/stress_exec.py
python scripts/test_sandbox.py
```

### Issues on `torch` and `vllm`
Download torch on [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).
For now, the `torch=2.4.0` and `vllm=0.6.3` works.
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Memory Issues During Training
```
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=data/code-r1-2k-leetcode2k-taco/train.parquet', 'data.val_files=data/code-r1-2k-leetcode2k-taco/test.parquet', 'data.train_batch_size=16', 'data.max_prompt_length=2048', 'data.max_response_length=4096', 'actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct-1M', 'actor_rollout_ref.actor.optim.lr=5e-7', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=256', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=False', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.rollout.log_prob_micro_batch_size=256', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.5', 'actor_rollout_ref.rollout.n=16', 'actor_rollout_ref.ref.log_prob_micro_batch_size=256', 'actor_rollout_ref.ref.fsdp_config.param_offload=False', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[wandb]', 'trainer.project_name=code-r1', 'trainer.experiment_name=code-r1-2k-leetcode2k-taco-grpo', 'trainer.nnodes=1', 'trainer.default_local_dir=./models/code-r1-2k-leetcode2k-taco-grpo', 'trainer.n_gpus_per_node=8', 'trainer.save_freq=64', 'trainer.test_freq=16', 'trainer.total_epochs=8', 'reward_model.reward_manager=prime']
Traceback (most recent call last):
  File "/home/***/github/code-r1/verl/trainer/main_ppo.py", line 25, in main
    run_ppo(config)
  File "/home/***/github/code-r1/verl/trainer/main_ppo.py", line 33, in run_ppo
    ray.get(main_task.remote(config, compute_score))
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/worker.py", line 2755, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/worker.py", line 906, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(OutOfMemoryError): ray::main_task() (pid=399789, ip=10.128.0.9)
  File "/home/***/github/code-r1/verl/trainer/main_ppo.py", line 128, in main_task
    trainer.fit()
  File "/home/***/github/code-r1/verl/trainer/ppo/ray_trainer.py", line 1004, in fit
    actor_output = self.actor_rollout_wg.update_actor(batch)
  File "/home/***/github/code-r1/verl/single_controller/ray/base.py", line 42, in func
    output = ray.get(output)
ray.exceptions.RayTaskError(OutOfMemoryError): ray::WorkerDict.actor_rollout_update_actor() (pid=400862, ip=10.128.0.9, actor_id=87523c02ec964b128fbc710001000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7edde69d7850>)
  File "/home/***/github/code-r1/verl/single_controller/ray/base.py", line 399, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/home/***/github/code-r1/verl/single_controller/base/decorator.py", line 404, in inner
    return func(*args, **kwargs)
  File "/home/***/github/code-r1/verl/workers/fsdp_workers.py", line 435, in update_actor
    metrics = self.actor.update_policy(data=data)
  File "/home/***/github/code-r1/verl/workers/actor/dp_actor.py", line 313, in update_policy
    loss.backward()
  File "/opt/conda/envs/code/lib/python3.10/site-packages/torch/_tensor.py", line 521, in backward
    torch.autograd.backward(
  File "/opt/conda/envs/code/lib/python3.10/site-packages/torch/autograd/__init__.py", line 289, in backward
    _engine_run_backward(
  File "/opt/conda/envs/code/lib/python3.10/site-packages/torch/autograd/graph.py", line 768, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.16 GiB. GPU 0 has a total capacity of 79.10 GiB of which 1.63 GiB is free. Including non-PyTorch memory, this process has 77.42 GiB memory in use. Of the allocated memory 68.16 GiB is allocated by PyTorch, and 3.49 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

Potential fix:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

However, this fix is temporary, and can only helps to train for more than 100 steps, then there is OOM again.

Additionally, when running with `vllm==0.8.2`, `verl==0.2.0.dev0`, `torch==2.6.0`, the expandable segments do not work with the memory pool mechanism of vllm, which gives the following error:
```bash
Traceback (most recent call last):
  File "/home/hazelye_google_com/github/code-r1/verl/trainer/main_ppo.py", line 25, in main
    run_ppo(config)
  File "/home/hazelye_google_com/github/code-r1/verl/trainer/main_ppo.py", line 33, in run_ppo
    ray.get(main_task.remote(config, compute_score))
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/worker.py", line 2771, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/ray/_private/worker.py", line 919, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(AssertionError): ray::main_task() (pid=2537612, ip=10.138.0.2)
  File "/home/hazelye_google_com/github/code-r1/verl/trainer/main_ppo.py", line 127, in main_task
    trainer.init_workers()
  File "/home/hazelye_google_com/github/code-r1/verl/trainer/ppo/ray_trainer.py", line 757, in init_workers
    self.actor_rollout_wg.init_model()
  File "/home/hazelye_google_com/github/code-r1/verl/single_controller/ray/base.py", line 42, in func
    output = ray.get(output)
ray.exceptions.RayTaskError(AssertionError): ray::WorkerDict.actor_rollout_init_model() (pid=2538987, ip=10.138.0.2, actor_id=6025581de22f7b458fe8e20001000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7f6aed92a440>)
  File "/home/hazelye_google_com/github/code-r1/verl/single_controller/ray/base.py", line 399, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/home/hazelye_google_com/github/code-r1/verl/single_controller/base/decorator.py", line 404, in inner
    return func(*args, **kwargs)
  File "/home/hazelye_google_com/github/code-r1/verl/workers/fsdp_workers.py", line 390, in init_model
    self.rollout, self.rollout_sharding_manager = self._build_rollout()
  File "/home/hazelye_google_com/github/code-r1/verl/workers/fsdp_workers.py", line 325, in _build_rollout
    rollout = vLLMRollout(model_path=local_path,
  File "/home/hazelye_google_com/github/code-r1/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py", line 100, in __init__
    self.inference_engine = LLM(
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/utils.py", line 1037, in inner
    return fn(*args, **kwargs)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/entrypoints/llm.py", line 243, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 520, in from_engine_args
    return engine_cls.from_vllm_config(
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 496, in from_vllm_config
    return cls(
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 280, in __init__
    self.model_executor = executor_class(vllm_config=vllm_config, )
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/executor/executor_base.py", line 52, in __init__
    self._init_executor()
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/executor/uniproc_executor.py", line 122, in _init_executor
    self.collective_rpc("load_model")
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/executor/uniproc_executor.py", line 56, in collective_rpc
    answer = run_method(self.driver_worker, method, args, kwargs)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/utils.py", line 2255, in run_method
    return func(*args, **kwargs)
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/worker/worker.py", line 174, in load_model
    allocator = CuMemAllocator.get_instance()
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/device_allocator/cumem.py", line 140, in get_instance
    CuMemAllocator.instance = CuMemAllocator()
  File "/opt/conda/envs/code/lib/python3.10/site-packages/vllm/device_allocator/cumem.py", line 145, in __init__
    assert "expandable_segments:True" not in conf, \
AssertionError: Expandable segments are not compatible with memory pool. Please track https://github.com/pytorch/pytorch/issues/147851 for the latest updates.
```

At this time, one can set the expandable segments to be False, and decrease the batch size. This will be good for about 140 epochs, then OOM again!

To solve this, we consider to reduce the memory by (i) **gradient checkpointing**: using GPU computation to save GPU memory and (ii) **param/optim offload**: using CPU memory to save GPU memory. Both will slow the training.
```bash
MODEL_ENABLE_GRADIENT_CHECKPOINTING=True  # TRUE: reduce memory by activation, increase computation time
ACTOR_PARAM_OFFLOAD=FALSE  # TRUE: reduce memory, good for nvlink
ACTOR_OPTIMIZER_OFFLOAD=True  # TRUE: reduce memory, good for nvlink
REF_PARAM_OFFLOAD=FALSE  # TRUE: reduce memory, but not much
VLLM_GPU_MEMORY_UTILIZATION=0.45
```

Upload models:
```bash
(a1) karl:~/github/code-r1/models/1-epoch-off-checkpoint/global_step_64/actor$ ls
extra_state_world_size_8_rank_0.pt  extra_state_world_size_8_rank_5.pt  model_world_size_8_rank_1.pt  model_world_size_8_rank_6.pt  optim_world_size_8_rank_3.pt
extra_state_world_size_8_rank_1.pt  extra_state_world_size_8_rank_6.pt  model_world_size_8_rank_2.pt  model_world_size_8_rank_7.pt  optim_world_size_8_rank_4.pt
extra_state_world_size_8_rank_2.pt  extra_state_world_size_8_rank_7.pt  model_world_size_8_rank_3.pt  optim_world_size_8_rank_0.pt  optim_world_size_8_rank_5.pt
extra_state_world_size_8_rank_3.pt  huggingface                         model_world_size_8_rank_4.pt  optim_world_size_8_rank_1.pt  optim_world_size_8_rank_6.pt
extra_state_world_size_8_rank_4.pt  model_world_size_8_rank_0.pt        model_world_size_8_rank_5.pt  optim_world_size_8_rank_2.pt  optim_world_size_8_rank_7.pt
(a1) karl:~/github/code-r1/models/1-epoch-off-checkpoint/global_step_64/actor$ cd huggingface/
(a1) karl:~/github/code-r1/models/1-epoch-off-checkpoint/global_step_64/actor/huggingface$ ls
added_tokens.json  config.json  merges.txt  special_tokens_map.json  tokenizer.json  tokenizer_config.json  vocab.json
(a1) karl:~/github/code-r1/models/1-epoch-off-checkpoint/global_step_64/actor/huggingface$ 
```