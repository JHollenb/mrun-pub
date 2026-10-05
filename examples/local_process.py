"""Run a contained offline process without a scheduler or model frameworks."""
import sys
from mrun import run

result = run([sys.executable, "-c", "print('mrun local execution works')"], ram_limit_mb=256)
print(result)
