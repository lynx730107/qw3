"""Exercise the existing llama-server web-tool fixture for one GGUF."""
import json
import runpy
import sys
from pathlib import Path

fixture = runpy.run_path("tests/test_agent_llama_reference.py")
url = fixture["REFERENCE_URL"]
expected = fixture["request_json"](url, timeout=60)
target = {k: expected[k] for k in ("full_name", "default_branch", "archived")}
target["license"] = expected["license"]["spdx_id"]
prompt = (f"Use visit_url exactly once on {url}. Read the returned GitHub repository JSON. "
          "Return only one JSON object with exactly these keys: full_name, default_branch, license, "
          "archived. license must be license.spdx_id and archived must be a JSON boolean.")
directory = Path(sys.argv[3])
directory.mkdir(exist_ok=True)
result = fixture["run_llama_reference"](sys.argv[1], Path(sys.argv[2]), url, prompt, directory, 180)
ok, error = fixture["validate_output"]("llama-server", result["output"], target)
print(json.dumps(dict(result=result, expected=target, valid=ok, error=error), indent=2))
sys.exit(0 if ok and result["tool_calls"] == 1 else 1)
