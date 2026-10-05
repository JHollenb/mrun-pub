"""Metadata planning must work in a service container without torch."""
import builtins
import importlib
from mrun.policy import HostCaps,plan_run

def test_light_client_uses_canonical_policy_when_selector_requires_torch(monkeypatch):
    monkeypatch.setenv("GATHER_MAX_BATCH","1")
    submit=importlib.import_module("mrun.client.submit")
    native_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name=="selector" and kwargs.get("level",args[3] if len(args)>3 else 0):
            raise ModuleNotFoundError("No module named 'torch'",name="torch")
        return native_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded_import)
    class FakeApi:
        def json(self,*args):
            return [{"name":"beast","ram_total_mb":64000,"vram_total_mb":16000,"cpu_threads":32,"caps":{"cuda":True},"models":[]}]
    sink={}
    plans=submit._plans_for_hosts("qwen3.5-2b","forward",FakeApi(),seq_lens=[3072],backend="hf",dtype="bfloat16",device="cuda",error_sink=sink)
    expected=plan_run("qwen3.5-2b","forward",host=HostCaps(name="beast",ram_mb=64000,vram_mb=16000,has_cuda=True,cpus=32),seq_lens=[3072],backend="hf",dtype="bfloat16",device="cuda").as_dict()
    assert not sink
    assert plans["beast"]["ram_limit_mb"]==expected["ram_limit_mb"]
    assert plans["beast"]["est_vram_mb"]==expected["est_vram_mb"]
    assert plans["beast"]["device"]=="cuda"
