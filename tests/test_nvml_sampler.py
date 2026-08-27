from streamguard.metrics.nvml_sampler import NVMLSampler


def test_start_stop_without_pynvml_produces_no_samples_and_sets_reason():
    sampler = NVMLSampler(interval_s=0.05)
    sampler.start()
    sampler.stop()
    # No NVIDIA driver / pynvml on this dev machine -- must degrade
    # gracefully, not raise, and must say why rather than silently
    # returning an empty list that looks like "GPU was just idle."
    assert sampler.samples == []
    assert sampler.unavailable_reason is not None


def test_stop_without_start_does_not_raise():
    sampler = NVMLSampler(interval_s=0.05)
    sampler.stop()  # must be safe even if start() was never called


def test_start_resets_samples_between_runs():
    sampler = NVMLSampler(interval_s=0.05)
    sampler.start()
    sampler.stop()
    sampler.samples.append("sentinel")  # simulate leftover state from a prior run
    sampler.start()
    assert sampler.samples == []
    sampler.stop()
