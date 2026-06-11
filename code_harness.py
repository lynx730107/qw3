import json
import subprocess
import os
import hashlib
import re
import shlex
import shutil
from datetime import datetime
import py_compile
from openai import OpenAI
import skeleton_parser
import argparse

# --- CONFIGURATION ---
DEFAULT_MODEL = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" 
DEFAULT_URL = "http://localhost"
DEFAULT_PORT = 8080
MAX_HISTORY = 50
ANCHOR_DELIMITER = "§"
USE_ANCHORS = True
HIDE_MODEL_THOUGHTS = False
WORKSPACE_ROOT = os.path.abspath(os.getcwd())
BACKUP_DIR = os.path.join(WORKSPACE_ROOT, ".harness_backups")
COMMAND_TIMEOUT = 60
SEMANTIC_SEARCH_TIMEOUT = 120
TOOL_OUTPUT_PRUNE_LIMIT = 6000
TOOL_OUTPUT_KEEP_HEAD = 2400
TOOL_OUTPUT_KEEP_TAIL = 1800
TOOL_OUTPUT_PRUNE_MARKER = "[Output tool compattato]"
HISTORY_KEEP_RECENT = 14
HISTORY_SUMMARY_MAX_TOKENS = 900
HISTORY_SUMMARY_INPUT_LIMIT = 5000
MAX_CONTEXT_TOKENS = 120000
AUTO_COMPACT_THRESHOLD = 0.8  # Soglia di compattazione automatica (80% di MAX_CONTEXT_TOKENS)
SAFE_COMMANDS = {
    "cat", "clang", "colgrep", "find", "git", "grep", "head", "ls", "make", "mypy",
    "python", "python3", "pytest", "pwd", "sed", "tail", "wc", "cc", "gcc"
}
BLOCKED_COMMAND_PATTERNS = [
    r"\brm\b", r"\bsudo\b", r"\bdd\b", r"\bmkfs\b", r"\bchmod\b", r"\bchown\b",
    r"\bshutdown\b", r"\breboot\b", r"\bkill\b", r"\bkillall\b", r"\bmv\b",
    r"/etc(/|$)", r"/System(/|$)", r"/Library(/|$)",
    r":\s*\(\s*\)\s*\{"
]

ANCHOR_WORDS = [
    "Apple", "Brave", "Cider", "Delta", "Eagle", "Fox", "Grape", "Hazel",
    "Index", "Joker", "Karma", "Lemon", "Mango", "Nacho", "Ocean", "Piano",
    "Quail", "River", "Snake", "Tiger", "Umbra", "Viper", "Water", "Xenon",
    "Yacht", "Zebra", "Aero", "Bison", "Camel", "Dart", "Echo", "Flare",
    "Ghost", "Halo", "Igloo", "Jazz", "Kite", "Lunar", "Magic", "Nexus",
    "Orbit", "Pulse", "Quest", "Radar", "Spark", "Tango", "Ultra", "Venus",
    "Wagon", "Xray", "Yield", "Zion", "Alpha", "Beta", "Gamma", "Sigma",
    "Omega", "Pluto", "Mars", "Saturn", "Jupiter", "Neptune", "Uranus", "Venus"
]

# --- CACHE FOR CONTEXT EFFICIENCY ---
seen_file_hashes = {}
file_anchor_map = {}
file_content_hash_map = {}

# --- UTILS ---
def get_content_hash(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_line_anchor(line):
    """Generates a stable word-based hash for a line's content, sensitive to indentation."""
    # Use rstrip to ignore trailing whitespace but keep leading indentation
    clean = line.rstrip()
    if not clean.strip(): return "BLANK"
    h = int(hashlib.md5(clean.encode('utf-8')).hexdigest(), 16)
    return ANCHOR_WORDS[h % len(ANCHOR_WORDS)]

def generate_anchors_for_lines(lines, existing_anchors=None):
    if existing_anchors is None:
        existing_anchors = set()
    else:
        existing_anchors = set(existing_anchors)
    
    new_anchors = []
    for line in lines:
        base_h = get_line_anchor(line)
        h = base_h
        counter = 2
        while h in existing_anchors:
            h = f"{base_h}#{counter}"
            counter += 1
        existing_anchors.add(h)
        new_anchors.append(h)
    return new_anchors

def sync_file_anchors(path, lines):
    content_hash = get_content_hash("".join(lines))
    if file_content_hash_map.get(path) != content_hash or path not in file_anchor_map:
        # Try to detect if the file already has anchors
        detected_anchors = []
        all_match = True
        for line in lines:
            # Match pattern: WORD[#N]§ or BLANK[#N]§
            m = re.match(r'^(([a-zA-Z]{4,10}|BLANK)(#[0-9]+)?)§', line)
            if m:
                detected_anchors.append(m.group(1))
            else:
                all_match = False
                break
        
        if all_match and len(detected_anchors) == len(lines) and len(lines) > 0:
            file_anchor_map[path] = detected_anchors
        else:
            file_anchor_map[path] = generate_anchors_for_lines(lines)
        
        file_content_hash_map[path] = content_hash
    return file_anchor_map[path]

def anchor_content(path, content, start_line=1, end_line=None):
    """Prefixes each line with a UNIQUE anchor: ANCHOR§CONTENT (or line numbers if USE_ANCHORS=False)"""
    lines = content.splitlines(keepends=True)
    if end_line is None: end_line = len(lines)
    
    anchored_lines = []
    
    if not USE_ANCHORS:
        for i in range(max(0, start_line - 1), min(end_line, len(lines))):
            anchored_lines.append(f"L{i+1}: {lines[i].rstrip()}")
        return "\n".join(anchored_lines)
        
    all_anchors = sync_file_anchors(path, lines)
    
    for i in range(max(0, start_line - 1), min(end_line, len(lines))):
        # If the line already had an anchor, we don't want to double it
        line_content = lines[i]
        if line_content.startswith(f"{all_anchors[i]}{ANCHOR_DELIMITER}"):
            anchored_lines.append(line_content.rstrip())
        else:
            anchored_lines.append(f"{all_anchors[i]}{ANCHOR_DELIMITER}{line_content.rstrip()}")
    return "\n".join(anchored_lines)

def strip_anchors(content):
    """Removes anchors from content (both with § and standalone names)."""
    # Strip pattern WORD§ or BLANK§
    content = re.sub(r'^([a-zA-Z]{4,10}|BLANK)(#[0-9]+)?§', '', content, flags=re.MULTILINE)
    # Strip standalone anchor names that AI sometimes hallucinates into the content
    # We especially target those with #N suffixes which are safe to strip
    content = re.sub(r'^([a-zA-Z]{4,10}|BLANK)#[0-9]+$', '', content, flags=re.MULTILINE)
    # Also strip BLANK if it's the only thing on the line
    content = re.sub(r'^BLANK$', '', content, flags=re.MULTILINE)
    return content

def strip_thought_tags(content):
    """Removes model reasoning blocks wrapped in <think>...</think>."""
    if not content:
        return content
    return re.sub(r"\s*<think>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE)

def check_syntax(path):
    """Performs syntax check for Python and C/C++ files."""
    if path.endswith('.py'):
        try:
            py_compile.compile(path, doraise=True)
            return ""
        except py_compile.PyCompileError as e:
            err_msg = str(e).split('\n')[-2] if '\n' in str(e) else str(e)
            return f"\n\n⚠️ LINTER (Python): Errore di sintassi:\n{err_msg}"
    elif path.endswith(('.c', '.cpp', '.h', '.hpp', '.cc')):
        try:
            # Use clang for syntax check on C/C++
            res = subprocess.run(f"clang -fsyntax-only \"{path}\"", shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                err = "\n".join(res.stderr.splitlines()[:5])
                return f"\n\n⚠️ LINTER (C/C++): Errore rilevato:\n{err}"
            return ""
        except:
            return ""
    return ""

def _resolve_path(path):
    """Resolve a relative path against the current working directory (_current_cwd)."""
    if os.path.isabs(path) or path.startswith(".."):
        return os.path.normpath(os.path.abspath(path))
    return os.path.normpath(os.path.join(_current_cwd, path))

def _is_inside_workspace(path):
    """Returns True when path is inside the current workspace."""
    try:
        abs_path = os.path.abspath(path)
        return os.path.commonpath([WORKSPACE_ROOT, abs_path]) == WORKSPACE_ROOT
    except ValueError:
        return False

def create_backup(path):
    """Creates a timestamped backup before destructive file writes."""
    if not os.path.exists(path):
        return ""
    if not _is_inside_workspace(path):
        raise ValueError(f"Path fuori workspace non consentito: {path}")

    rel_path = os.path.relpath(os.path.abspath(path), WORKSPACE_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = os.path.join(BACKUP_DIR, rel_path + f".{timestamp}.bak")
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(path, backup_path)
    return backup_path

def get_file_skeleton(path):
    """Returns semantic skeleton using tree-sitter (primary) or colgrep (fallback)."""
    try:
        resolved = _resolve_path(path)
        if not os.path.exists(resolved): return f"Errore: File {resolved} non trovato."
        
        with open(resolved, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        all_anchors = sync_file_anchors(resolved, lines)
        
        # --- PRIMARY: tree-sitter (deterministic, fast, hierarchical) ---
        if skeleton_parser.is_supported(resolved):
            defs = skeleton_parser.extract_skeleton(resolved)
            if defs:
                skeleton = []
                for d in defs:
                    line_num = d['line']
                    if 1 <= line_num <= len(lines):
                        if USE_ANCHORS:
                            prefix = f"{all_anchors[line_num - 1]}{ANCHOR_DELIMITER}"
                        else:
                            prefix = f"L{line_num}: "
                        indent = "  " * d['depth']
                        skeleton.append((
                            line_num,
                            f"{prefix}{indent}[{d['kind']}] {d['signature']}  (L{d['line']}-{d['end_line']})"
                        ))
                skeleton.sort(key=lambda x: x[0])
                formatted_skeleton = [item[1] for item in skeleton]
                return f"--- Skeleton di {resolved} ---\n" + "\n".join(formatted_skeleton)
        
        # --- FALLBACK: colgrep (for unsupported languages) ---
        return _get_file_skeleton_colgrep(resolved, lines, all_anchors)
    except Exception as e:
        return f"Errore skeleton: {e}"


def read_files(paths, start_line=1, end_line=None):
    """Reads multiple files with unique anchors."""
    if isinstance(paths, str): paths = [paths]
    results = []
    for path in paths:
        try:
            resolved = _resolve_path(path)
            if not os.path.exists(resolved):
                results.append(f"--- {path} ---\nErrore: File non trovato.")
                continue
            with open(resolved, 'r', encoding='utf-8') as f:
                content = f.read()
            c_hash = get_content_hash(content)
            header = f"--- {resolved} [Hash: {c_hash[:8]}] ---"
            if seen_file_hashes.get(resolved) == c_hash and start_line == 1 and end_line is None:
                results.append(f"{header}\n(Nessuna modifica dall'ultima lettura)")
            else:
                if start_line == 1 and end_line is None:
                    seen_file_hashes[resolved] = c_hash
                results.append(f"{header}\n{anchor_content(resolved, content, start_line, end_line)}")
        except Exception as e:
            results.append(f"--- {path} ---\nErrore: {e}")
    return "\n\n".join(results)

def edit_file(path, edits):
    """Applies surgical edits using unique anchors."""
    try:
        resolved = _resolve_path(path)
        if not os.path.exists(resolved): return f"Errore: File {resolved} non trovato."
        
        pre_errors = check_syntax(resolved)
        
        with open(resolved, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_anchors = sync_file_anchors(resolved, lines)
        
        resolved_edits = []
        for edit in edits:
            # Clean anchor name from any trailing delimiter
            start_a = edit.get('start_anchor', '').split(ANCHOR_DELIMITER)[0].strip()
            end_a = edit.get('end_anchor', start_a).split(ANCHOR_DELIMITER)[0].strip()
            if not end_a: end_a = start_a
            new_text = strip_anchors(edit.get('content', ''))
            edit_type = edit.get('edit_type', 'replace')
            
            try:
                start_idx = current_anchors.index(start_a)
                end_idx = current_anchors.index(end_a)
            except ValueError:
                return f"Errore: Anchor '{start_a}' o '{end_a}' non trovati. Verifica di aver usato l'anchor ESATTO (es. 'Apple#2')."
                
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            resolved_edits.append((start_idx, end_idx, new_text, edit_type))
            
        resolved_edits.sort(key=lambda x: x[0], reverse=True)
        for start_idx, end_idx, new_text, edit_type in resolved_edits:
            new_lines = new_text.splitlines(keepends=True)
            
            # --- AUTO-INDENT RECOVERY PER MODELLI PICCOLI ---
            # I modelli < 8B spesso dimenticano gli spazi iniziali nei JSON.
            # Se la riga originale aveva un'indentazione e le nuove righe partono da 0, la ripristiniamo.
            if new_lines:
                target_line = lines[start_idx]
                original_indent = len(target_line) - len(target_line.lstrip('\t '))
                
                if original_indent > 0:
                    non_empty_lines = [l for l in new_lines if l.strip()]
                    if non_empty_lines:
                        min_new_indent = min(len(l) - len(l.lstrip('\t ')) for l in non_empty_lines)
                        if min_new_indent < original_indent:
                            indent_str = " " * (original_indent - min_new_indent)
                            for i in range(len(new_lines)):
                                if new_lines[i].strip():
                                    new_lines[i] = indent_str + new_lines[i]

            if new_lines and not new_lines[-1].endswith('\n'):
                new_lines[-1] += '\n'
            
            if edit_type == 'insert_before':
                lines[start_idx:start_idx] = new_lines
                new_anchs = generate_anchors_for_lines(new_lines, existing_anchors=current_anchors)
                current_anchors[start_idx:start_idx] = new_anchs
            elif edit_type == 'insert_after':
                lines[start_idx+1:start_idx+1] = new_lines
                new_anchs = generate_anchors_for_lines(new_lines, existing_anchors=current_anchors)
                current_anchors[start_idx+1:start_idx+1] = new_anchs
            else: # replace
                lines[start_idx:end_idx+1] = new_lines
                del current_anchors[start_idx:end_idx+1]
                new_anchs = generate_anchors_for_lines(new_lines, existing_anchors=current_anchors)
                current_anchors[start_idx:start_idx] = new_anchs
            
        final_content = "".join(lines)
        backup_path = create_backup(resolved)
        with open(resolved, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        file_content_hash_map[resolved] = get_content_hash(final_content)
        
        post_errors = check_syntax(resolved)
        
        result = f"File '{resolved}' aggiornato."
        if backup_path:
            result += f"\nBackup creato: {backup_path}"
        if pre_errors and not post_errors:
            result += "\n✅ Errori di sintassi RISOLTI!"
        elif not pre_errors and post_errors:
            result += f"\n⚠️ NUOVI errori introdotti:{post_errors}"
        elif post_errors:
            result += f"\n⚠️ Errori di sintassi persistono:{post_errors}"
            
        return result
    except Exception as e:
        return f"Errore: {e}"
def replace_lines(path, start_line, end_line, new_code):
    """Sostituisce righe specifiche in base al numero di riga."""
    try:
        resolved = _resolve_path(path)
        if not os.path.exists(resolved): return f"Errore: File {resolved} non trovato."
        pre_errors = check_syntax(resolved)
        with open(resolved, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines) - 1, end_line - 1)
        
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
            
        new_text = strip_anchors(new_code) # just in case
        new_lines = new_text.splitlines(keepends=True)
        
        # Auto-Indent Recovery
        if new_lines and start_idx < len(lines):
            target_line = lines[start_idx]
            original_indent = len(target_line) - len(target_line.lstrip('\t '))
            if original_indent > 0:
                non_empty_lines = [l for l in new_lines if l.strip()]
                if non_empty_lines:
                    min_new_indent = min(len(l) - len(l.lstrip('\t ')) for l in non_empty_lines)
                    if min_new_indent < original_indent:
                        indent_str = " " * (original_indent - min_new_indent)
                        for i in range(len(new_lines)):
                            if new_lines[i].strip():
                                new_lines[i] = indent_str + new_lines[i]

        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
            
        lines[start_idx:end_idx+1] = new_lines
        
        final_content = "".join(lines)
        backup_path = create_backup(resolved)
        with open(resolved, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        file_content_hash_map[resolved] = get_content_hash(final_content)
        if resolved in file_anchor_map:
            del file_anchor_map[resolved]
            
        post_errors = check_syntax(resolved)
        result = f"Righe {start_line}-{end_line} sostituite con successo."
        if backup_path:
            result += f"\nBackup creato: {backup_path}"
        if pre_errors and not post_errors: result += "\n✅ Errori di sintassi RISOLTI!"
        elif not pre_errors and post_errors: result += f"\n⚠️ NUOVI errori introdotti:{post_errors}"
        elif post_errors: result += f"\n⚠️ Errori di sintassi persistono:{post_errors}"
        return result
    except Exception as e:
        return f"Errore: {e}"


def write_file(path, content):
    try:
        resolved = _resolve_path(path)
        if not _is_inside_workspace(resolved):
            return f"Errore: Path fuori workspace non consentito: {resolved}"
        backup_path = create_backup(resolved)
        with open(resolved, 'w', encoding='utf-8') as f:
            f.write(content)
        if resolved in file_anchor_map:
            del file_anchor_map[resolved]
        file_content_hash_map[resolved] = get_content_hash(content)
        linter_result = check_syntax(resolved)
        result = f"File '{resolved}' scritto con successo."
        if backup_path:
            result += f"\nBackup creato: {backup_path}"
        return result + linter_result
    except Exception as e:
        return f"Errore: {e}"

def get_function(path, function_name):
    """Estrae il codice completo di una funzione o metodo per nome usando AST."""
    try:
        resolved = _resolve_path(path)
        if not os.path.exists(resolved): return f"Errore: File {resolved} non trovato."
        if not skeleton_parser.is_supported(resolved):
            return f"Errore: estrazione AST non supportata per questo file."
        
        defs = skeleton_parser.extract_skeleton(resolved)
        if not defs:
            return f"Errore: nessuna funzione trovata nel file {resolved}."
            
        matches = [d for d in defs if d['name'] == function_name]
        if not matches:
            return f"Funzione '{function_name}' non trovata in {resolved}."
            
        # Per C/C++, ignoriamo i prototipi se esiste l'implementazione
        match = next((d for d in matches if d['kind'] != 'prototype'), matches[0])
            
        return read_files([resolved], start_line=match['line'], end_line=match['end_line'])
    except Exception as e:
        return f"Errore: {e}"

def replace_symbol(path, symbol_name, new_code):
    """Sostituisce un'intera funzione/classe per nome usando tree-sitter."""
    try:
        resolved = _resolve_path(path)
        if not os.path.exists(resolved): return f"Errore: File {resolved} non trovato."
        if not skeleton_parser.is_supported(resolved):
            return f"Errore: edit AST non supportato per questo file."
        
        pre_errors = check_syntax(resolved)
        
        defs = skeleton_parser.extract_skeleton(resolved)
        if not defs:
            return f"Errore: nessuno skeleton estraibile per {resolved}."
            
        matches = [d for d in defs if d['name'] == symbol_name]
        if not matches:
            return f"Simbolo '{symbol_name}' non trovato in {resolved}."
            
        # Per C/C++, ignoriamo i prototipi se esiste l'implementazione
        match = next((d for d in matches if d['kind'] != 'prototype'), matches[0])
        
        with open(resolved, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        start_idx = match['line'] - 1
        end_idx = match['end_line'] - 1
        
        new_text = strip_anchors(new_code)
        new_lines = new_text.splitlines(keepends=True)
        if new_text and not new_text.endswith('\n'):
            new_lines[-1] += '\n'
            
        lines[start_idx:end_idx+1] = new_lines
        
        final_content = "".join(lines)
        backup_path = create_backup(resolved)
        with open(resolved, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        file_content_hash_map[resolved] = get_content_hash(final_content)
        if resolved in file_anchor_map:
            del file_anchor_map[resolved]
            
        post_errors = check_syntax(resolved)
        
        result = f"Simbolo '{symbol_name}' sostituito con successo (L{match['line']}-L{match['end_line']})."
        if backup_path:
            result += f"\nBackup creato: {backup_path}"
        if pre_errors and not post_errors:
            result += "\nâœ… Errori di sintassi RISOLTI!"
        elif not pre_errors and post_errors:
            result += f"\n⚠️ NUOVI errori introdotti:{post_errors}"
        elif post_errors:
            result += f"\n⚠️ Errori di sintassi persistono:{post_errors}"
            
        return result
    except Exception as e:
        return f"Errore: {e}"

def run_shell_command(command):
    try:
        if not command or not command.strip():
            return "Errore: comando vuoto."

        for pattern in BLOCKED_COMMAND_PATTERNS:
            if re.search(pattern, command):
                return f"Errore: comando bloccato per sicurezza (pattern: {pattern}): {command}"

        try:
            args = shlex.split(command)
        except ValueError as e:
            return f"Errore: comando non parsabile: {e}"

        if not args:
            return "Errore: comando vuoto."

        executable = os.path.basename(args[0])
        #if executable not in SAFE_COMMANDS:
        #    return f"Errore: comando non consentito: {executable}"

        result = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            cwd=_current_cwd,
        )
        return f"Return code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Errore: Timeout ({COMMAND_TIMEOUT}s)."
    except Exception as e:
        return f"Errore: {e}"

def semantic_search(query, paths=None, results=10, include=None, exclude=None, code_only=True, semantic_only=False, content=False):
    """Runs a real ColGREP semantic/hybrid search and returns compact, script-friendly results."""
    try:
        if not query or not query.strip():
            return "Errore: query semantica vuota."

        if shutil.which("colgrep") is None:
            return "Errore: colgrep non trovato nel PATH. Installa con: brew install lightonai/tap/colgrep"

        if paths is None:
            paths = [_current_cwd]
        elif isinstance(paths, str):
            paths = [paths]
        elif not paths:
            paths = [_current_cwd]

        safe_paths = []
        for path in paths:
            abs_path = os.path.abspath(path)
            if not _is_inside_workspace(abs_path):
                return f"Errore: path fuori workspace non consentito: {path}"
            if not os.path.exists(abs_path):
                return f"Errore: path non trovato: {path}"
            safe_paths.append(abs_path)

        try:
            k = max(1, min(int(results), 50))
        except (TypeError, ValueError):
            k = 10

        args = ["colgrep", "--json", "--results", str(k)]
        if code_only:
            args.append("--code-only")
        if semantic_only:
            args.append("--semantic-only")
        if content:
            args.append("--content")
        if include:
            args.append(f"--include={include}")
        if exclude:
            args.append(f"--exclude={exclude}")
        args.append(query)
        args.extend(safe_paths)

        result = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=SEMANTIC_SEARCH_TIMEOUT,
            cwd=_current_cwd,
        )

        if result.returncode != 0:
            return f"Errore colgrep (code {result.returncode}):\n{result.stderr.strip() or result.stdout.strip()}"

        stdout = result.stdout.strip()
        if not stdout:
            return "Nessun risultato semantico trovato."

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout[:6000]

        if not isinstance(data, list):
            return json.dumps(data, indent=2, ensure_ascii=False)[:6000]

        lines = [f"--- ColGREP semantic_search: {query!r} ({len(data)} risultati) ---"]
        for idx, entry in enumerate(data[:k], 1):
            unit = entry.get("unit", entry) if isinstance(entry, dict) else {}
            score = entry.get("score") or entry.get("rerank_score") or entry.get("similarity") if isinstance(entry, dict) else None
            file_path = unit.get("file") or unit.get("path") or entry.get("file", "?")
            line = unit.get("line") or entry.get("line", "?")
            end_line = unit.get("end_line") or entry.get("end_line", line)
            unit_type = unit.get("unit_type") or unit.get("kind") or entry.get("unit_type", "unit")
            signature = unit.get("signature") or unit.get("name") or entry.get("signature", "")
            rel_file = os.path.relpath(file_path, _current_cwd) if file_path != "?" else file_path
            score_text = f" score={score}" if score is not None else ""
            lines.append(f"{idx}. {rel_file}:L{line}-L{end_line} [{unit_type}]{score_text}")
            if signature:
                lines.append(f"   {signature}")
            snippet = unit.get("content") or unit.get("source") or entry.get("content") or entry.get("snippet")
            if snippet:
                compact = "\n".join(str(snippet).splitlines()[:12])
                lines.append("   ---")
                lines.extend(f"   {s}" for s in compact.splitlines())

        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return f"Errore: Timeout ricerca semantica ({SEMANTIC_SEARCH_TIMEOUT}s)."
    except Exception as e:
        return f"Errore semantic_search: {e}"

def compact_tool_output(content, tool_name="tool"):
    """Compacts old tool outputs while preserving useful head/tail context and metadata."""
    content = str(content)
    if len(content) <= TOOL_OUTPUT_PRUNE_LIMIT or TOOL_OUTPUT_PRUNE_MARKER in content:
        return content

    head = content[:TOOL_OUTPUT_KEEP_HEAD].rstrip()
    tail = content[-TOOL_OUTPUT_KEEP_TAIL:].lstrip()
    omitted = content[TOOL_OUTPUT_KEEP_HEAD:-TOOL_OUTPUT_KEEP_TAIL]
    omitted_lines = omitted.count("\n")
    omitted_chars = len(omitted)

    header = (
        f"{TOOL_OUTPUT_PRUNE_MARKER}\n"
        f"Tool: {tool_name}\n"
        f"Originale: {len(content)} caratteri. Rimossi: {omitted_chars} caratteri, circa {omitted_lines} righe.\n"
        "Nota: il modello ha già visto l'output completo nel turno precedente. "
        "Se servono dettagli mancanti, richiedi una lettura mirata con read_files/get_function/semantic_search.\n"
    )
    separator = "\n\n... [parte centrale rimossa per ridurre prefill/token] ...\n\n"
    return header + head + separator + tail

def _message_content_for_summary(msg):
    """Converts any chat message, including tool calls, into summary-safe plain text."""
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""

    if role == "tool":
        name = msg.get("name", "tool")
        return "user", f"[OUTPUT TOOL: {name}]\n{compact_tool_output(content, name)}"

    if role == "assistant" and msg.get("tool_calls"):
        tool_names = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            tool_names.append(fn.get("name", "unknown"))
        suffix = f"\n[TOOL CALLS RICHIESTE: {', '.join(tool_names)}]"
        content = f"{content}{suffix}" if content else suffix.strip()

    if len(str(content)) > HISTORY_SUMMARY_INPUT_LIMIT:
        content = compact_tool_output(content, f"history_{role}")

    if role not in ("user", "assistant"):
        role = "user"
    return role, str(content)

def build_deterministic_history_digest(old_messages):
    """Local fallback summary when the model-based condensation fails."""
    lines = ["[Fallback locale: riepilogo deterministico della cronologia rimossa]"]
    for i, msg in enumerate(old_messages, 1):
        role, content = _message_content_for_summary(msg)
        first_line = content.replace("\n", " ")[:500]
        lines.append(f"{i}. {role}: {first_line}")
    return "\n".join(lines)

def condense_history(client, model_name, messages):
    """Condenses old chat history without sending raw tool-role messages to the API."""
    if len(messages) <= MAX_HISTORY + 1:
        return messages, None

    recent_count = min(HISTORY_KEEP_RECENT, max(4, MAX_HISTORY // 2))
    old_messages = messages[1:-recent_count]
    recent_messages = messages[-recent_count:]

    summary_messages = []
    for msg in old_messages:
        role, content = _message_content_for_summary(msg)
        if content.strip():
            summary_messages.append({"role": role, "content": content})

    summary_prompt = (
        "Condensa la cronologia tecnica per continuare un task di sviluppo software.\n"
        "Mantieni obbligatoriamente:\n"
        "- obiettivo corrente e richieste utente non ancora completate;\n"
        "- file modificati e funzioni toccate;\n"
        "- decisioni tecniche e motivazioni;\n"
        "- comandi/test eseguiti e relativi esiti;\n"
        "- errori incontrati e fix applicati;\n"
        "- stato corrente e prossimi passi concreti.\n"
        "Non includere conversazione sociale irrilevante. Sii compatto ma specifico."
    )

    try:
        summary_resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": summary_prompt}] + summary_messages,
            max_tokens=HISTORY_SUMMARY_MAX_TOKENS
        )
        summary = summary_resp.choices[0].message.content or "(Riassunto vuoto)"
        status = "Fatto"
    except Exception as e:
        summary = build_deterministic_history_digest(old_messages)
        status = f"Fallback locale: {e}"

    condensed = [
        messages[0],
        {"role": "assistant", "content": f"[Contesto precedente condensato]\n{summary}"}
    ] + recent_messages
    return condensed, status

# --- CONTEXTO TOKEN ESTIMATION ---

def estimate_context_tokens(messages):
    """Stima il numero di token del contesto attuale."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            # OpenAI format con content come lista
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total_chars += len(block["text"])
                elif isinstance(block, str):
                    total_chars += len(block)
        else:
            total_chars += len(str(content))
    
    # Stima approssimativa: ~4 caratteri per token (media per testo inglese/ codice)
    return int(total_chars / 4)


# --- TOOL REGISTRY ---
available_functions = {
    "read_files": read_files,
    "get_file_skeleton": get_file_skeleton,
    "edit_file": edit_file,
    "replace_lines": replace_lines,
    "write_file": write_file,
    "run_shell_command": run_shell_command,
    "get_function": get_function,
    "replace_symbol": replace_symbol,
    "semantic_search": semantic_search,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_file_skeleton",
            "description": "Struttura semantica del file con anchors UNICI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": "Legge file con anchors UNICI (es. 'A1B2#2' per duplicati).",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer"}
                },
                "required": ["paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Modifiche chirurgiche. Usa l'anchor ESATTO (es. 'Apple' o 'Apple#2' se duplicato).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "edit_type": {"type": "string", "enum": ["replace", "insert_before", "insert_after"], "default": "replace"},
                                "start_anchor": {"type": "string"},
                                "end_anchor": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["start_anchor", "content"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Scrive l'intero file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_function",
            "description": "Estrae il codice completo di una funzione o metodo dato il nome (usa AST).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "function_name": {"type": "string", "description": "Nome esatto della funzione o metodo"}
                },
                "required": ["path", "function_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_symbol",
            "description": "Sostituisce un'intera funzione/classe per nome senza usare anchors (usa AST). Ideale per riscrivere funzioni intere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "symbol_name": {"type": "string", "description": "Nome esatto della funzione o classe (es. 'main' o 'MyClass')"},
                    "new_code": {"type": "string", "description": "Il nuovo codice completo della funzione/classe"}
                },
                "required": ["path", "symbol_name", "new_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Cerca codice per significato reale usando ColGREP (semantic/hybrid search). Usa questo prima di grep quando non conosci nomi esatti o vuoi trovare logica per intento.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query naturale, es. 'backup automatico prima di scrivere file'"},
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "File o directory dentro la workspace. Default: workspace corrente."},
                    "results": {"type": "integer", "default": 10, "description": "Numero massimo di risultati, 1-50."},
                    "include": {"type": "string", "description": "Glob opzionale, es. '*.py' o 'src/**/*.ts'."},
                    "exclude": {"type": "string", "description": "Glob opzionale da escludere."},
                    "code_only": {"type": "boolean", "default": True},
                    "semantic_only": {"type": "boolean", "default": False, "description": "Se true disabilita la fusione keyword e usa ricerca semantica pura."},
                    "content": {"type": "boolean", "default": False, "description": "Se true chiede a colgrep contenuto più completo."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Esegue comandi bash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }
]

# --- CHAT LOOP ---

# --- LOCAL COMMANDS STATE ---
_current_cwd = WORKSPACE_ROOT

def _cmd_cd(args_str):
    global _current_cwd
    path = args_str.strip()
    if not path:
        return "Usage: /cd <directory>"
    if path == "~":
        path = os.path.expanduser("~")
    if path.startswith("~"):
        path = os.path.expanduser(path)
    # Permetti solo percorsi relativi o assoluti sicuri
    if path.startswith("/"):
        target = path
    else:
        target = os.path.normpath(os.path.join(_current_cwd, path))
    
    if not os.path.isdir(target):
        return f"Errore: directory non trovata: {target}"
    _current_cwd = target
    return f"Current directory: {_current_cwd}"

def _cmd_ls(args_str):
    path = args_str.strip()
    if not path:
        target = _current_cwd
    else:
        if path.startswith("/"):
            target = path
        else:
            target = os.path.normpath(os.path.join(_current_cwd, path))
        if not os.path.exists(target):
            return f"Errore: percorso non trovato: {target}"
    
    try:
        entries = sorted(os.listdir(target))
        lines = []
        for name in entries:
            full = os.path.join(target, name)
            if os.path.isdir(full):
                lines.append(f"[DIR]  {name}/")
            else:
                lines.append(f"[FILE] {name}")
        return f"Directory contents of {target}:\n" + "\n".join(lines)
    except PermissionError:
        return f"Errore: permessi negati per {target}"

def chat(model_name, url, port):
    global tools, _current_cwd, HIDE_MODEL_THOUGHTS
    _current_cwd = WORKSPACE_ROOT
    base_url = f"{url}:{port}/v1"
    client = OpenAI(base_url=base_url, api_key="sk-locale")

    active_tools = list(tools)

    if not USE_ANCHORS:
        active_tools = [t for t in active_tools if t["function"]["name"] != "edit_file"]
        active_tools.append({
            "type": "function",
            "function": {
                "name": "replace_lines",
                "description": "Sostituisce le righe specificate usando i numeri di riga (1-indexed).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "new_code": {"type": "string", "description": "Nuovo codice puro. NON includere i prefissi L: e mantieni l'indentazione corretta."}
                    },
                    "required": ["path", "start_line", "end_line", "new_code"]
                }
            }
        })
        system_prompt = (
            "Sei un assistente AI sviluppatore esperto. Hai a disposizione i seguenti strumenti:\n"
            "- get_file_skeleton: restituisce la struttura semantica con numeri di riga.\n"
            "- get_function: estrae il codice completo di una funzione tramite il suo nome.\n"
            "- semantic_search: cerca codice per significato reale con ColGREP; usalo quando non conosci nomi esatti o vuoi trovare logica per intento.\n"
            "- read_files: legge il contenuto parziale o totale con numeri di riga inclusi (L<num>:).\n"
            "- replace_symbol: sostituisce un'intera funzione/classe per nome (eccellente per riscrivere blocchi interi).\n"
            "- replace_lines: sostituisce righe esatte usando start_line e end_line.\n"
            "- write_file: scrive o sovrascrive un intero file.\n"
            "- run_shell_command: esegue comandi nel terminale.\n"
            "- /cd: cambia la directory corrente. Esempio: /cd src\n"
            "- /ls: elenca i file nella directory corrente. Esempio: /ls\n"
            "- /help: mostra i comandi locali disponibili.\n"
            "- /reset: resetta il contesto storico (pulizia memoria). Esempio: /reset\n"
            "- /exit: esce dal harness.\n\n"
            "REGOLE PER GLI EDIT:\n"
            "1. INDENTAZIONE: L'indentazione in 'new_code' deve essere ESATTAMENTE quella finale.\n"
            "2. NON copiare il prefisso 'L1: ' dentro 'new_code'. 'new_code' deve essere puro sorgente.\n"
            "3. LINTER: Se ricevi errori di sintassi dopo l'edit, correggi subito!\n"
        )
    else:
        system_prompt = (
            "Sei un assistente AI sviluppatore esperto. Hai a disposizione i seguenti strumenti:\n"
            "- get_file_skeleton: restituisce la struttura semantica (classi/funzioni) con anchors unici.\n"
            "- get_function: estrae il codice completo di una funzione tramite il suo nome.\n"
            "- semantic_search: cerca codice per significato reale con ColGREP; usalo quando non conosci nomi esatti o vuoi trovare logica per intento.\n"
            "- read_files: legge il contenuto completo o parziale dei file con anchors unici.\n"
            "- replace_symbol: sostituisce un'intera funzione/classe per nome (migliore per riscrivere funzioni intere).\n"
            "- edit_file: applica modifiche chirurgiche multiple usando gli anchors.\n"
            "- write_file: scrive o sovrascrive un intero file.\n"
            "- run_shell_command: esegue comandi nel terminale.\n"
            "- /cd: cambia la directory di lavoro locale. Esempio: /cd src\n"
            "- /ls: elenca i file e le cartelle nella directory corrente. Esempio: /ls\n"
            "- /help: mostra i comandi locali disponibili.\n"
            "- /reset: resetta il contesto storico (pulizia memoria). Esempio: /reset\n"
            "- /exit: esce dal harness.\n\n"
            "REGOLE PER GLI EDIT (CRITICHE):\n"
            "L'edit_file supporta 'replace', 'insert_before', 'insert_after'.\n"
            "1. NO ANCHOR NEL CONTENT: Non inserire MAI parole di anchor come 'Apple§' nel 'content'. Il 'content' deve essere codice puro.\n"
            "2. INDENTAZIONE: L'indentazione in 'content' deve essere ESATTAMENTE quella finale.\n"
            "3. VERIFICA LINTER: Se ricevi errori dal LINTER, rimedia subito!\n"
        )

    messages = [{"role": "system", "content": system_prompt}]
    print("\033[95m=== Qwen Harness (Stable Anchoring Mode v5) ===\033[0m")
    while True:
        # Compattazione automatica del contesto quando si supera la soglia di token
        if len(messages) > 1:
            estimated_tokens = estimate_context_tokens(messages)
            if estimated_tokens > MAX_CONTEXT_TOKENS * AUTO_COMPACT_THRESHOLD:
                print(f"\n\033[90m[Compattazione automatica del contesto... ({estimated_tokens} token stimati)]\033[0m", end="", flush=True)
                messages, condense_status = condense_history(client, model_name, messages)
                print(f" {condense_status}.")
        elif len(messages) > MAX_HISTORY + 1:
            print("\n\033[90m[Condensazione del contesto...]\033[0m", end="", flush=True)
            messages, condense_status = condense_history(client, model_name, messages)
            print(f" {condense_status}.")

        try: user_input = input("\n\033[92m❯ \033[0m")
        except EOFError: break
        if user_input.lower() in ['exit', 'quit', 'esci']: break

        # GESTIONE COMANDI LOCALI (/cd, /ls, /reset, /help)
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "/cd":
                result = _cmd_cd(args)
                print(f"\033[94m[Local] {result}\033[0m")
                continue
            elif cmd == "/ls":
                result = _cmd_ls(args)
                print(f"\033[94m[Local] {result}\033[0m")
                continue
            elif cmd == "/reset":
                messages.clear()
                messages.append({"role": "system", "content": system_prompt})
                print("\033[93m[Local] Contesto storico resettato.\033[0m")
                continue
            elif cmd == "/help":
                help_text = (
                    "Comandi locali disponibili:\n"
                    "  /cd [dir]        - Cambia la directory corrente.\n"
                    "  /ls              - Elenco file/cartelle nella directory corrente.\n"
                    "  /hide_thoughts   - Nasconde i blocchi <think>...</think> nell'output del modello.\n"
                    "  /show_thoughts   - Mostra nuovamente i blocchi <think>...</think> se presenti.\n"
                    "  /help            - Mostra questa lista di aiuto.\n"
                    "  /reset           - Resetta il contesto storico (pulizia memoria).\n"
                    "  /exit            - Esce dal harness."
                )
                print(f"\033[94m[Local]\n{help_text}\033[0m")
                continue
            elif cmd == "/exit":
                print("[93m[Local] Uscita dal harness.[0m")
                break
            elif cmd == "/hide_thoughts":
                HIDE_MODEL_THOUGHTS = True
                print("[Local] Pensieri del modello nascosti.")
                continue
            elif cmd == "/show_thoughts":
                HIDE_MODEL_THOUGHTS = False
                print("[Local] Pensieri del modello mostrati.")
                continue
            else:
                print(f"[91m[Local] Comando sconosciuto: {cmd}[0m")
                continue

        # PRUNING: compattiamo gli output dei tool passati per risparmiare token con i LLM locali.
        # Preserviamo testa+coda e metadati, evitando il vecchio taglio secco a 800 caratteri.
        # Se serve un dettaglio rimosso, il modello deve richiedere una lettura mirata.
        for msg in messages:
            if msg.get("role") == "tool":
                msg["content"] = compact_tool_output(msg.get("content", ""), msg.get("name", "tool"))

        messages.append({"role": "user", "content": user_input})
        while True:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages, tools=active_tools, tool_choice="auto", stream=True
                )
                print("\n\033[96mQwen:\033[0m ", end="", flush=True)
                full_content = ""
                displayed_content = ""
                tool_calls_dict = {}
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_content += delta.content
                        if HIDE_MODEL_THOUGHTS:
                            processed = strip_thought_tags(full_content)
                        else:
                            processed = full_content
                        new_segment = processed[len(displayed_content):]
                        if new_segment:
                            print(new_segment, end="", flush=True)
                            displayed_content = processed
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {"id": tc.id or "", "type": "function", "function": {"name": tc.function.name or "", "arguments": ""}}
                            if tc.function.arguments:
                                tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments
                print("\033[0m")
                if HIDE_MODEL_THOUGHTS:
                    stored_content = strip_thought_tags(full_content)
                else:
                    stored_content = full_content
                resp_msg = {"role": "assistant", "content": stored_content or None}
                if tool_calls_dict: resp_msg["tool_calls"] = list(tool_calls_dict.values())
                messages.append(resp_msg)
                if "tool_calls" in resp_msg:
                    for tc in resp_msg["tool_calls"]:
                        name = tc["function"]["name"]
                        func = available_functions.get(name)
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            print(f"\033[93m[Tool] {name}({args})\033[0m")
                            result = func(**args)
                        except Exception as e: result = f"Errore: {e}"
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": str(result)})
                    continue
                break
            except Exception as e:
                print(f"\n\033[91m[Errore]\033[0m {e}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Harness (Stable Anchoring Mode)")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help="Nome del modello da usare con l'API compatibile OpenAI.")
    parser.add_argument("-u", "--url", type=str, default=DEFAULT_URL, help="URL del server API (es. http://localhost).")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help="Porta del server API.")
    parser.add_argument("--no-anchors", action="store_true", help="Disabilita gli anchor e usa il classico editing basato sui numeri di riga (consigliato per modelli < 8B).")
    parser.add_argument("--hide-thoughts", action="store_true", help="Nasconde i blocchi <think>...</think> nell'output del modello.")
    args = parser.parse_args()
    
    if args.no_anchors:
        USE_ANCHORS = False
    if args.hide_thoughts:
        HIDE_MODEL_THOUGHTS = True
        
    chat(args.model, args.url, args.port)