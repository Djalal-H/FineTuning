import dataclasses
import sys
from pathlib import Path

import pytest
import yaml

from mathscribe import ablations
from mathscribe.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = yaml.safe_load((ROOT / "configs" / "ablations.yaml").read_text())
BASE = ROOT / ABLATIONS["base_config"]
RUNS = ABLATIONS["runs"]


def _flat(prefix: str, obj) -> dict:
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            out.update(_flat(f"{prefix}.{key}" if prefix else key, value))
        return out
    return {prefix: obj}


def _differences(a, b) -> set[str]:
    flat_a, flat_b = _flat("", a), _flat("", b)
    return {key for key in flat_a if flat_a[key] != flat_b.get(key)}


def test_run_names_are_unique():
    names = [run["name"] for run in RUNS]
    assert len(names) == len(set(names))
    assert names[0] == "base"


@pytest.mark.parametrize("run", RUNS, ids=lambda r: r["name"])
def test_run_loads_and_changes_only_its_factor(run):
    base = load_config(BASE)
    cfg = load_config(BASE, run["overrides"])
    changed = _differences(cfg, base)
    stated = {o.partition("=")[0] for o in run["overrides"]}
    # An override that sets the base value again changes nothing, so "changed" can be smaller.
    assert changed <= stated
    assert len(stated) == len(run["overrides"])  # no key twice
    if run["name"] != "base":
        assert changed  # the run must differ from the base


def test_base_run_has_no_overrides():
    assert RUNS[0]["overrides"] == []


def test_ablation_base_is_the_fixed_subset():
    cfg = load_config(BASE)
    assert cfg.data.max_train_samples == 4000
    assert cfg.validation.manifest in cfg.data.exclude
    assert cfg.validation.early_stopping_patience is None  # every run trains all steps
    assert cfg.train["load_best_model_at_end"] is True


def test_dora_is_a_config_key():
    assert load_config(BASE, ["lora.use_dora=true"]).lora.use_dora is True
    assert load_config(BASE).lora.use_dora is False


# --------------------------------------------------------------------------- #
# The runner (mathscribe.ablations)
# --------------------------------------------------------------------------- #
SPEC = ablations.load_spec(ROOT / "configs" / "ablations.yaml")


def _jobs(tmp_path, names=None, **kwargs):
    return ablations.build_jobs(
        SPEC,
        names,
        python="python",
        output_root=tmp_path / "out",
        results_dir=tmp_path / "res",
        **kwargs,
    )


def test_build_jobs_covers_every_run_and_loads_every_config(tmp_path):
    jobs = _jobs(tmp_path)
    assert [j.name for j in jobs] == [f"abl_{r['name']}" for r in SPEC["runs"]]
    for job in jobs:
        assert [s.label for s in job.steps] == ["train", "eval"]


def test_train_command_resumes_pushes_and_sets_its_factor(tmp_path):
    job = _jobs(tmp_path, ["rslora"])[0]
    cmd = job.steps[0].command()
    assert "--resume-if-available" in cmd
    assert "lora.use_rslora=true" in cmd
    assert "hub.repo_id=djaloul/mathscribe-abl-rslora" in cmd
    assert f"train.output_dir={tmp_path / 'out' / 'abl_rslora'}" in cmd
    assert job.steps[0].done_file == tmp_path / "out" / "abl_rslora" / "run_info.json"


def test_eval_uses_the_local_adapter_if_it_exists_else_the_hub(tmp_path):
    job = _jobs(tmp_path, ["r8"])[0]
    eval_step = job.steps[1]
    assert "model.name=djaloul/mathscribe-abl-r8" in eval_step.command()
    adapter = tmp_path / "out" / "abl_r8"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    assert f"model.name={adapter}" in eval_step.command()
    assert "testset.manifest=data/testsets/latex_ocr_val.json" in eval_step.command()
    assert "output.baseline_run=abl_base" in eval_step.command()


def test_qlora_run_is_evaluated_in_4bit(tmp_path):
    cmd = _jobs(tmp_path, ["qlora4"])[0].steps[1].command()
    assert "model.load_in_4bit=true" in cmd
    assert "model.load_in_4bit=false" in _jobs(tmp_path, ["base"])[0].steps[1].command()


def test_tag_extra_overrides_and_options(tmp_path):
    job = _jobs(
        tmp_path,
        ["base"],
        tag="smoke",
        extra_overrides=["train.max_steps=10"],
        hub=False,
        evaluate=False,
    )[0]
    cmd = job.steps[0].command()
    assert job.name == "abl_base_smoke"
    assert "train.max_steps=10" in cmd
    assert not any(c.startswith("hub.") for c in cmd)
    assert len(job.steps) == 1


def test_unknown_run_and_bad_override_are_errors(tmp_path):
    with pytest.raises(ConfigError, match="Unknown ablation runs"):
        _jobs(tmp_path, ["nope"])
    with pytest.raises(ConfigError):
        _jobs(tmp_path, ["base"], extra_overrides=["lora.not_a_key=1"])


# A fake step: print the GPU id, then write the done file (unless the name says "fail").
FAKE = (
    "import os, sys; print('gpu', os.environ['CUDA_VISIBLE_DEVICES'], "
    "os.environ['UNSLOTH_COMPILE_LOCATION']); "
    "sys.exit(1) if sys.argv[2] == 'fail' else open(sys.argv[1], 'w').write('x')"
)


def _fake_job(tmp_path, name, mode="ok", steps=2):
    out = []
    for i in range(steps):
        done = tmp_path / f"{name}_{i}.done"
        cmd = [sys.executable, "-c", FAKE, str(done), mode if i == 0 else "ok"]
        out.append(ablations.Step(f"step{i}", lambda c=cmd: c, done))
    return ablations.Job(name, out)


def test_run_jobs_uses_each_gpu_and_gives_each_its_own_cache(tmp_path, capsys):
    jobs = [_fake_job(tmp_path, f"j{i}") for i in range(4)]
    results = ablations.run_jobs(jobs, [0, 1], log_dir=tmp_path / "logs", cache_root=tmp_path / "c")
    assert [r.status for r in results] == ["done"] * 4
    assert {r.gpu for r in results} == {0, 1}
    out = capsys.readouterr().out
    assert f"gpu 0 {tmp_path / 'c' / 'gpu0'}" in out
    assert f"gpu 1 {tmp_path / 'c' / 'gpu1'}" in out
    assert all(j.steps[-1].done_file.exists() for j in jobs)
    assert (tmp_path / "logs" / "j0.log").read_text().startswith("$ ")


def test_run_jobs_skips_finished_steps_and_keeps_going_after_a_failure(tmp_path):
    finished = _fake_job(tmp_path, "finished")
    for step in finished.steps:
        step.done_file.write_text("x")
    broken = _fake_job(tmp_path, "broken", mode="fail")
    fine = _fake_job(tmp_path, "fine")
    results = ablations.run_jobs(
        [finished, broken, fine], [0], log_dir=tmp_path / "logs", cache_root=tmp_path / "c"
    )
    by_name = {r.name: r for r in results}
    assert by_name["finished"].status == "skipped"
    assert by_name["broken"].status == "FAILED"
    assert by_name["broken"].failed_step == "step0"
    assert not broken.steps[1].done_file.exists()  # the next step of a failed job does not run
    assert by_name["fine"].status == "done"


@pytest.mark.parametrize("run", SPEC.get("skipped", []), ids=lambda r: r["name"])
def test_skipped_runs_have_a_reason_and_are_not_scheduled(run):
    assert run["reason"].strip()
    assert run["name"] not in {r["name"] for r in SPEC["runs"]}
    load_config(BASE, run["overrides"])  # the override is still a valid config
