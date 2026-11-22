"""Small, constrained C/Python/Git tool-use checks on a local llama-server."""
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

fixture = runpy.run_path("tests/test_agent_llama_reference.py")
port = fixture["free_port"]()
url = f"http://127.0.0.1:{port}"
write_tool = {"name": "write", "description": "Write the given file contents.",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                             "required": ["path", "content"]}}
run_tool = {"name": "bash", "description": "Run the requested shell command in the working directory.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
cases = [
    ("C", "agent.c", '#include <stdio.h>\nint main(void) { puts("CODING_SMOKE_OK"); return 0; }',
     "cc agent.c -o agent && ./agent", "CODING_SMOKE_OK"),
    ("Python", "demo.py", 'print("vado, a, casa")', "python3 demo.py", "vado, a, casa"),
    ("Git", "", "", "git log -3 --pretty=%s", "THIRD\nSECOND\nFIRST"),
]

def check_case(case, cwd):
    kind, name, content, command, expected = case
    if name:
        prompt = (f"Use write to create {name} with exactly the following contents:\n{content}\n"
                  f"Then use bash to run exactly: {command}\nReport only the printed output.")
    else:
        subprocess.run(["git", "init", "-q"], cwd=cwd, check=True)
        for msg in ("FIRST", "SECOND", "THIRD"):
            subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@invalid",
                            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                            "commit", "-q", "--allow-empty", "-m", msg], cwd=cwd, check=True)
        prompt = f"Use bash to run exactly: {command}\nReturn only its three output lines in order."
    messages = [{"role": "user", "content": prompt}]
    events = []
    final = ""
    for _ in range(4):
        response = fixture["request_json"](url + "/v1/chat/completions", {
            "messages": messages, "model": "local", "temperature": 0, "seed": 1, "max_tokens": 192,
            "tools": [{"type": "function", "function": t} for t in (write_tool, run_tool)]})
        msg = response["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            final = msg.get("content") or ""
            break
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        for call in calls:
            tool = call["function"]["name"]
            args = fixture["parse_tool_arguments"](call)
            result = "Unsupported arguments"
            if name and tool == "write" and args.get("path") == name and args.get("content","").rstrip("\n") == content:
                (cwd / name).write_text(content + "\n")
                result = "OK"
            elif tool == "bash" and args.get("command") == command and (not name or (cwd / name).is_file()):
                result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
            events.append([tool,result])
            messages.append({"role":"tool","tool_call_id":call["id"],"content":result})
    wanted = ([["write", "OK"]] if name else []) + [["bash", expected]]
    return {"case":kind, "calls":events, "response":final, "pass":events == wanted and final.strip() == expected}

with tempfile.TemporaryDirectory() as temp:
    cwd = Path(temp)
    with open(sys.argv[3],"w") as log:
        process = subprocess.Popen([sys.argv[1], "-m", sys.argv[2], "-ngl", "99", "-c", "4096",
                                    "--parallel", "1", "--host", "127.0.0.1", "--port", str(port),
                                    "--reasoning", "off", "--no-webui"], stdout=log, stderr=log)
        try:
            ready, error = fixture["wait_for_server"](url, process, 120)
            if not ready:
                raise RuntimeError(error)
            for case in cases:
                workspace = cwd / case[0]
                workspace.mkdir()
                print(json.dumps(check_case(case, workspace)), flush=True)
        finally:
            process.terminate()
            process.wait(timeout=15)
