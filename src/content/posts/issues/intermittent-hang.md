---
title: How I Debugged an Intermittent Hang
pubDatetime: 2026-11-08T00:00:00Z
description: Narrowing down a flaky vLLM CI timeout from a hanging test to a GPU queue that never finishes a memset.
tags:
  - vllm
  - debugging
  - xpu
---

A unit test in our CI for vLLM started timing out now and then. It never failed on my local machine, and on CI it only failed once in a while. This post walks through how I narrowed it down, layer by layer, from "a flaky test" to "a GPU queue that never finishes a memset", and what I would do differently next time.

**TL;DR**

- The hang happens while two vLLM workers (tensor parallel = 2) set up their oneCCL communicator.
- One rank waits for its peer's data; the peer is stuck waiting for a simple `memset` on its own GPU that never completes.
- The kernel log shows the GPU queue was really stuck: `Schedule disable failed to respond`, followed by a device coredump. The same signature showed up on a second machine, at exactly the time the CI test timed out.
- The most useful tools were, in order: call stacks (py-spy, gdb), an offset comparison inside the native frame, and the kernel log (`journalctl -k`).

## 1. Reproduce the hang and find out where it happens

First you need to know *where* the hang happens; that decides which layer to look at. Reproduce the
issue and read the log carefully. The evidence is usually already there.

In my case, the CI log of the timed-out test ended like this:

```text
../vllm/entrypoints/llm.py:348: in __init__
    self.llm_engine = LLMEngine.from_engine_args(
...
../vllm/v1/engine/utils.py:1393: in launch_core_engines
    wait_for_engine_startup(
../vllm/v1/engine/utils.py:1440: in wait_for_engine_startup
    events = poller.poll(STARTUP_POLL_PERIOD_MS)
```

The main process was waiting for the EngineCore process to report "ready", and that never happened.
During startup, EngineCore creates the workers and initializes the devices, then the KV cache and the
scheduler. With two workers, the most likely place to hang is the communication between them, for
example the oneCCL warm-up `all_reduce` in `XPUWorker.init_device`.

> **Tip:** If the hang only happens occasionally, run the same command in a loop until it hangs:
>
> ```bash
> for i in $(seq 1 30); do
>   echo "=== iter $i ==="
>   <your command> -p no:timeout 2>&1 | tee /tmp/iter_$i.log || break
> done
> ```
>
> `-p no:timeout` disables pytest-timeout, so a hung test stays alive for you to inspect instead of
> being killed after 120 s.

> **Note:** If the hang happens inside a container, start your container with the *same* image and
> the *same* `docker run` arguments as CI. Small differences matter. In my case, a container without
> `--privileged` made oneCCL silently fall back from the `pidfd` IPC exchange mode to `drmfd`, which
> changes the code path under test.

## 2. Turn on debug logging

Most libraries have environment variables for debug output, for example `VLLM_LOGGING_LEVEL=DEBUG` for
vLLM. Since I suspected oneCCL, I also set `CCL_LOG_LEVEL=info`:

```bash
grep -n "CCL" /tmp/iter_2.log | tail -40
```

The most useful lines were:

```text
|CCL_INFO| CCL_ZE_IPC_EXCHANGE: pidfd
|CCL_INFO| level zero driver version: 17013219
|CCL_WARN| topology recognition shows PCIe connection between devices. ...
|CCL_INFO| all vertices connected: is_matrix_connected: 0
|CCL_INFO| host: 0, plane 0 contains ranks: 0
|CCL_INFO| host: 0, plane 1 contains ranks: 1
|CCL_INFO| PCI device [0:139:0:0] is on NUMA node 1 (CPU intersection: 96)
|CCL_INFO| PCI device [0:148:0:0] is on NUMA node 1 (CPU intersection: 96)
```

The library log is a good *fingerprint of the environment*: which mode is used, which devices, which
topology. That was very useful later when comparing machines. But it only tells you roughly where the
hang is (between which two log lines). To see what each process is actually doing, you need step 3.

## 3. Dump the call stacks of every process

Hangs usually happen in the lower layers of the stack, such as the communication library or the
driver, not in the application itself. So the next question is: what is each process doing right now?

I used `py-spy` for Python plus native frames, and `gdb` for all threads:

```bash
for p in $(pgrep -f "^VLLM::Worker") $(pgrep -f "^VLLM::EngineCore"); do
  echo "== $p =="; py-spy dump --native -p $p
done > /tmp/stacks.txt 2>&1
```

Both workers were stuck in the same Python line, the warm-up `all_reduce`, but at different places
underneath (trimmed):

```text
== 8087 (rank 0) ==
    ofi_dynpoll_wait_epoll (libtcp-fi.so)
    all_reduce (torch/distributed/distributed_c10d.py:4174)
    XPUWorker.init_device (vllm/v1/worker/xpu_worker.py:195)

== 8088 (rank 1) ==
    0x75610678208e (libze_intel_gpu.so.1.17.39395)
    ur::level_zero::urEventWait (libur_adapter_level_zero_v2.so.0)
    all_reduce (torch/distributed/distributed_c10d.py:4174)
    XPUWorker.init_device (vllm/v1/worker/xpu_worker.py:195)
```

`gdb` shows the full native picture, including background threads that py-spy cannot see:

```bash
for p in 8087 8088; do gdb -p $p -batch -ex "thread apply all bt" > /tmp/gdb_$p.txt 2>&1; done
```

Main thread of each rank (trimmed):

```text
rank 1:
#2  ur::level_zero::urEventWait(...)          libur_adapter_level_zero_v2.so.0
#5  sycl::_V1::detail::event_impl::waitInternal(bool*)
#7  coll_init(ccl_comm*, ccl_stream*)         libccl.so.1
#8  ccl_comm::init(...)
#17 c10d::ProcessGroupXCCL::initXCCLComm(...)

rank 0:
#1  ofi_dynpoll_wait_epoll ()                 libtcp-fi.so
#6  atl_ofi_comm::allgatherv(...)
#9  ze_handle_exchange_entry::common_fd_mode_exchange()
#11 do_ipc_exchange(ccl_comm*, ccl_stream*, ...)
#12 coll_init(ccl_comm*, ccl_stream*)         libccl.so.1
```

So both ranks were inside `coll_init`, which oneCCL runs while it creates the communicator. Rank 0 was 
waiting in an allgather for rank 1's IPC handles. Rank 1 was waiting for some GPU event. At this point
I still didn't know which one was the cause and which one was the victim.

## 4. Find the exact position inside the native function

The stack shows function names but no line numbers, but you can still tell *where* inside a function 
each process is, by asking gdb for the offset of the return address. Do it in a single gdb session, 
because a spinning process changes its top frames between attaches:

```bash
for p in <rank0_pid> <rank1_pid>; do
  gdb -p $p -batch -ex "thread 1" -ex "frame function coll_init" \
      -ex 'info symbol $pc' -ex 'x/i $pc-5'
done
```

```text
rank 0: coll_init(ccl_comm*, ccl_stream*) + 24905
        <+24900>: call do_ipc_exchange(...)
rank 1: coll_init(ccl_comm*, ccl_stream*) + 11078
        <+11073>: call sycl::_V1::event::wait()
```

Both processes load the same `libccl.so.1`, so the offsets are directly comparable: rank 1 (11073) had
not even reached the IPC exchange (24900) yet. A static disassembly (no running process needed) shows
what it was waiting for:

```bash
gdb -batch -ex "disassemble 'coll_init(ccl_comm*, ccl_stream*)'" /opt/venv/lib/libccl.so.1 | grep call
```

```text
<+11060>: call  sycl::_V1::queue::memset(...)
<+11073>: call  sycl::_V1::event::wait()      <- rank 1 is here
```

Now the story is clear: rank 1 submitted a `memset` on its own GPU and waits for it forever. Because
it never reaches the exchange, it never sends its IPC handles, so rank 0 waits forever in the
allgather. Rank 0 is the victim, and the real question is why a simple `memset` never completes.

## 5. Check the kernel log

When a process is stuck waiting for the GPU, check the GPU driver's kernel log. I should have done
this much earlier: it takes seconds.

```bash
sudo journalctl -k --since "7 days ago" | grep -iE "xe .*(engine reset|schedule disable|coredump)"
```

On the CI machine, when I killed the hung processes, the GPU used by rank 1 logged:

```text
xe 0000:94:00.0: [drm] GT0: Schedule disable failed to respond, guc_id=22
xe 0000:94:00.0: [drm] Xe device coredump has been created
xe 0000:94:00.0: [drm] GT0: trying reset from xe_guc_exec_queue_lr_cleanup.cold [xe]
xe 0000:94:00.0: [drm] GT0: reset done
```

When the process was killed, the kernel tried to tear down its GPU queue, but the GuC (the GPU's
scheduler firmware) could not even stop it. That means the queue was really stuck on the GPU side.
Nothing was logged *during* the hang itself, probably because these long-running ("lr") queues have 
no timeout.

Then I checked a second CI machine that had hit the same timeout, and the timestamps lined up
exactly:

| Time (UTC) | Event |
|---|---|
| 07:30:52 | pytest-timeout fires in the CI job |
| 07:30:57 | vLLM force-kills the two workers |
| 07:31:02 | kernel: `Schedule disable failed to respond` + device coredump |

The same signature on a second machine ruled out a single badly configured host. At this point the
evidence (the stacks, the offset comparison and the device coredump) pointed below the user-space
libraries, which is where this write-up stops.

> **Prefer `journalctl -k` over `dmesg`:** on a busy Kubernetes node, AppArmor audit messages from
> container probes (one every 10 s) quickly push older driver messages out of the `dmesg` ring buffer.
> `journalctl -k` keeps them, and its timestamps are also more reliable than `dmesg -T`.

## 6. Lessons learned

**Check the GPU early.** As soon as one process is waiting on a GPU event (`urEventWait`,
`zeEventHostSynchronize`, ...), look at the kernel log. I spent many CI runs trying to reproduce the
hang before doing this.

**A reproducer must reproduce what happened *before* the hang, not only where it hangs.** I wrote a
minimal script that does exactly what the worker does: create a two-rank `xccl` process group and run
one `all_reduce`. It hung a few times at first, but later ran 500+ iterations without a single hang,
while the real unit test reproduced it again quickly. The difference: in my script the processes exit
cleanly, while the shutdown tests *force-kill* workers that still have GPU work in flight and then
immediately start new ones on the same GPUs.

**Clean up after every hang.** When you stop a hung test with Ctrl+C, check for leftover
`VLLM::Worker` / `VLLM::EngineCore` processes and kill them. Otherwise they keep using the GPU and
silently change the conditions of your next experiment.

**Don't stop at the first workaround.** Switching oneCCL to `CCL_ZE_IPC_EXCHANGE=sockets` *seemed* to
fix it. But once the offsets showed the hang happens *before* the IPC exchange, that "fix" was most
likely luck. Small sample sizes are misleading for intermittent bugs.

## Appendix: a watchdog that dumps stacks automatically

If you don't know when the hang will happen (for example in CI), run a background watchdog that dumps
any worker that lives too long.

```bash
(while sleep 15; do
   for p in $(pgrep -f "VLLM::Worker"); do
     [ "$(ps -o etimes= -p $p)" -gt 90 ] && { echo "=== py-spy $p ==="; py-spy dump --native -p $p; }
   done
 done)
```

