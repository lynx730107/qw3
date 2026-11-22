#!/usr/bin/env python3
"""Exercise native audio.cpp HTTP, failures and prose filtering without a mic/model.

Set QW3_VOICE_TEST_MODEL to additionally run a real QW3 turn against the mock
TTS server. Only a short silent WAV is played; microphone access is never used.
"""
import email.parser
import email.policy
import io
import json
import math
import os
import pty
from pathlib import Path
import select
import signal
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import wave
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent.parent
VOICE = ROOT / "qw3-voice"
AGENT = ROOT / "qw3-agent-voice"


def silence():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)
    return out.getvalue()


class Server(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = []
    speech_events = []
    lock = threading.Lock()

    def log_message(self, *_):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass  # Expected when cancellation closes a keep-alive connection.

    def reply(self, value, status=200):
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav" if isinstance(value, bytes) else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.endswith("/health"):
            if self.path.startswith("/bad-json/"):
                self.reply(b"not JSON")
            elif self.path.startswith("/unready/"):
                self.reply({"status": "loading"})
            else:
                self.reply({"status": "ok"})
        elif self.path.endswith("/v1/models"):
            ids = ["qwen3-asr"] if self.path.startswith("/missing/") else ["qwen3-asr", "kokoro"]
            self.reply({"data": [{"id": name} for name in ids]})
        else:
            self.reply({"error": "unknown route"}, 404)

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        with self.lock:
            self.requests.append((self.path, self.headers["Content-Type"], data))
        if self.path.startswith("/delay/"):
            time.sleep(2)
        if self.path.endswith("/v1/tasks/unload_models"):
            self.reply({"unloaded": [], "not_found": []})
        elif self.path.startswith("/failure/"):
            self.reply({"error": "fixture failure"}, 503)
        elif self.path.endswith("/v1/audio/transcriptions"):
            text = "" if self.path.startswith("/empty/") else " Ciao, perché non parli? "
            self.reply({"text": text, "timing": {}})
        elif self.path.endswith("/v1/audio/speech"):
            if self.path.startswith("/pipeline"):
                text = json.loads(data)["input"]
                with self.lock:
                    self.speech_events.append((time.clock_gettime(time.CLOCK_MONOTONIC), text))
                    index = len(self.speech_events)
                time.sleep(0.8 if self.path.startswith("/pipeline-delay/") else 0.08)
                if self.path.startswith("/pipeline-fail/") and index == 2:
                    self.reply({"error": "second chunk failed"}, 503)
                else:
                    self.reply(silence() + text.encode())
            else:
                self.reply(b"not a WAV" if self.path.startswith("/bad-wav/") else silence())
        else:
            self.reply({"error": "unknown route"}, 404)


class VoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="qw3-voice-test-")
        self.addCleanup(self.directory.cleanup)
        self.tmp = Path(self.directory.name)
        self.wav = self.tmp / "input.wav"
        self.wav.write_bytes(silence())
        with Server.lock:
            Server.requests.clear()
            Server.speech_events.clear()

    def config(self, mode="", **settings):
        path = self.tmp / "voice.json"
        path.write_text(json.dumps({"server_url": self.base + mode, **settings}))
        return path

    def run_voice(self, *args, ok=True):
        result = subprocess.run([str(VOICE), *map(str, args)], capture_output=True, text=True, timeout=10)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_health_and_agent_check_need_no_llm(self):
        cfg = self.config()
        self.run_voice("--config", cfg, "check")
        result = subprocess.run([str(AGENT), "--voice-config", str(cfg), "--voice-check",
                                 "-m", "/no-model.gguf", "--store-dir", str(self.tmp / "store")],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("inference not checked", result.stderr)
        self.assertNotIn("engine open", result.stderr)

    def test_transcription_upload(self):
        result = self.run_voice("--config", self.config(), "transcribe", self.wav)
        self.assertEqual(result.stdout.strip(), "Ciao, perché non parli?")
        path, content_type, body = Server.requests[-1]
        self.assertEqual(path, "/v1/audio/transcriptions")
        message = email.parser.BytesParser(policy=email.policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
        parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
        self.assertEqual(parts["model"].get_payload(decode=True), b"qwen3-asr")
        self.assertEqual(parts["language"].get_payload(decode=True), b"Italian")
        self.assertEqual(parts["file"].get_payload(decode=True), self.wav.read_bytes())

    def test_synthesis_contract_and_prose(self):
        output = self.tmp / "reply.wav"
        self.run_voice("--config", self.config(), "speak", "Ciao.\n```sh\nrm file\n```\nFine.", output)
        self.assertEqual(output.read_bytes(), silence())
        path, content_type, body = Server.requests[-1]
        payload = json.loads(body)
        self.assertEqual(path, "/v1/audio/speech")
        self.assertEqual(content_type, "application/json")
        self.assertEqual(payload, {"model": "kokoro", "voice": "if_sara", "input": "Ciao. Fine.",
                                   "response_format": "wav", "options": {"language": "it"}})

    def test_thinking_code_and_markdown_are_filtered(self):
        fixtures = [
            (True, "pensiero segreto</think>**Ciao**, [mondo](https://example.com).", "Ciao, mondo."),
            (True, "pensiero incompleto", ""),
            (False, "<think>segreto</think>Ciao.", "Ciao."),
            (False, "Ciao. <think>non terminato", "Ciao."),
            (False, "Ciao.\n~~~python\nprint('segreto')\n~~~\nFine.", "Ciao. Fine."),
            (False, "Ciao.\n````sh\nsecret\n```\nstill secret\n````\nFine.", "Ciao. Fine."),
            (False, "Ciao.\n```sh\nblocco incompleto", "Ciao."),
            (False, "# Titolo\n- Primo\n    codice\nSecondo `inline`.", "Titolo. Primo. Secondo inline."),
            (False, "Ciao. <tool_call>segreto</tool_call>", "Ciao."),
            (False, "Ciao. <｜DSML｜tool_calls>segreto", "Ciao."),
            (False, "Ciao.\x1b[31m Rosso.\x1b[0m", "Ciao. Rosso."),
        ]
        for thinking, raw, expected in fixtures:
            with self.subTest(raw=raw):
                args = ["prepare"] + (["--thinking"] if thinking else []) + [raw]
                self.assertEqual(self.run_voice(*args).stdout.strip(), expected)

    def test_inline_data_keeps_values_units_and_identifiers(self):
        fixtures = [
            ("Prezzo `9,99 €`, variazione `-1,5%`, temperatura `21,5 °C`.",
             "Prezzo 9,99 €, variazione -1,5%, temperatura 21,5 °C."),
            ("Modello `Qwen3.6-35B`, versione `1.2.3`, codice `item_42`.",
             "Modello Qwen3.6-35B, versione 1.2.3, codice item_42."),
            ("Operazione `2*3`; testo ``un ` segno``.", "Operazione 2*3; testo un ` segno."),
            ("Testo `senza chiusura.", "Testo `senza chiusura."),
            ("| Misura | Valore |\n|---|---|\n|Peso|`12,50 kg`|",
             "Misura: Peso. Valore: 12,50 kg."),
        ]
        for raw, expected in fixtures:
            with self.subTest(raw=raw):
                self.assertEqual(self.run_voice("prepare", raw).stdout.strip(), expected)

    def test_spoken_sources_keep_names_without_reciting_url_paths(self):
        fixtures = [
            ("Fonte: [Istituto di statistica](https://example.com/dati_(2026)?track=42).",
             "Fonte: Istituto di statistica."),
            ("Vedi https://example.com/dati/2026?track=42 e <https://other.org/archivio>.",
             "Vedi example.com e other.org."),
            ("[https://example.com/dati?q=42](https://example.com/dati?q=42)", "example.com"),
            ('Secondo [Fonte A][dati] e [Fonte B][].\n\n'
             '[dati]: https://example.com/a "Titolo"\n[Fonte B]: <https://other.org/b>',
             "Secondo Fonte A e Fonte B."),
            ("Unità [kg] e [riferimento non risolto][42].", "Unità [kg] e [riferimento non risolto][42]."),
            ("![Grafico](https://example.com/chart_(1).png)\nRestano 48% e 21,5 °C.",
             "Restano 48% e 21,5 °C."),
            ("```\n[segreto]: https://secret.org\n```\n[Dato][segreto]", "[Dato][segreto]"),
            ("| Fonte | Dato |\n|---|---|\n|[Istituto](https://example.com/a)|`48%`|",
             "Fonte: Istituto. Dato: 48%."),
        ]
        for raw, expected in fixtures:
            with self.subTest(raw=raw):
                self.assertEqual(self.run_voice("prepare", raw).stdout.strip(), expected)
        self.assertFalse(Server.requests, "source formatting must not fetch URLs")

    def test_speech_speed_config_and_runtime_changes(self):
        cfg = self.config(tts_speed=0.85)
        self.run_voice("--config", cfg, "speak", "Prezzo `9,99 €`.", self.tmp / "speed.wav")
        payload = json.loads(Server.requests[-1][2])
        self.assertEqual(payload["options"], {"language": "it", "speed": 0.85})
        self.assertEqual(payload["input"], "Prezzo 9,99 €.")
        source = self.tmp / "speed.c"
        source.write_text(r'''
#include <math.h>
#include "qw3_voice.h"
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    qw3_voice *voice = 0;
    char err[512] = {0};
    if (qw3_voice_open(&voice, argv[1], err, sizeof(err))) return 2;
    if (qw3_voice_get_speed(voice) != 0.85) return 3;
    if (qw3_voice_set_speed(voice, 0.7, err, sizeof(err))) return 4;
    double bad[] = {NAN, INFINITY, -INFINITY, -1, 0.49, 2.01};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        if (!qw3_voice_set_speed(voice, bad[i], err, sizeof(err))) return 5;
        if (qw3_voice_get_speed(voice) != 0.7) return 6;
    }
    if (qw3_voice_synthesize(voice, "Prima.", argv[2], 0, 0, err, sizeof(err))) return 7;
    if (qw3_voice_set_speed(voice, 0, err, sizeof(err))) return 8;
    if (qw3_voice_get_speed(voice) != 0) return 9;
    if (qw3_voice_synthesize(voice, "Seconda.", argv[2], 0, 0, err, sizeof(err))) return 10;
    qw3_voice_close(voice);
    return 0;
}
''')
        executable = self.tmp / "speed"
        subprocess.run(["cc", "-I", str(ROOT), str(source), str(ROOT / "qw3_voice.o"),
                        "-framework", "Foundation", "-framework", "AVFoundation",
                        "-o", str(executable)], capture_output=True, check=True, timeout=30)
        result = subprocess.run([str(executable), str(cfg), str(self.tmp / "runtime.wav")],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        speeches = [json.loads(body) for path, _, body in Server.requests if path.endswith("/speech")]
        self.assertEqual(speeches[-2]["options"], {"language": "it", "speed": 0.7})
        self.assertEqual(speeches[-1]["options"], {"language": "it"})

    def test_speech_plan_preserves_data_and_unicode_without_network(self):
        cfg = self.config(speech_chunk_chars=80)
        text = ("Il dott. Rossi usa la versione `1.2.3` e rileva `21,5 °C`. "
                "Il prezzo è `9,99 €` e la variazione è `-1,5%`. "
                "La seconda misura è di `12,50 kg`; il codice è `item_42`. "
                "Questa frase prosegue molto a lungo " * 3)
        prose = self.run_voice("prepare", text).stdout.strip()
        plan = json.loads(self.run_voice("--config", cfg, "plan", text).stdout)
        self.assertGreater(len(plan), 2)
        self.assertEqual(" ".join(plan), prose)
        self.assertTrue(all(chunk and len(chunk) <= 80 for chunk in plan))
        for value in ["1.2.3", "21,5", "9,99", "-1,5%", "12,50", "item_42"]:
            self.assertTrue(any(value in chunk for chunk in plan))
        token = "à🦊" * 100
        plan = json.loads(self.run_voice("--config", cfg, "plan", token).stdout)
        self.assertEqual(plan, [token], "a long identifier must not be cut mid-word or mid-Unicode")
        self.assertEqual(json.loads(self.run_voice("--config", cfg, "plan", "```\nno\n```").stdout), [])
        self.assertFalse(Server.requests)
        for value in [0, 79, 1001, 80.5, True, "320"]:
            with self.subTest(speech_chunk_chars=value):
                self.run_voice("--config", self.config(speech_chunk_chars=value), "plan", "Ciao.", ok=False)

    def pipeline_fixture(self):
        executable = self.tmp / "pipeline"
        subprocess.run(["cc", "-I", str(ROOT), str(ROOT / "tests/test_voice_pipeline.c"),
                        str(ROOT / "qw3_voice.o"), "-framework", "Foundation",
                        "-framework", "AVFoundation", "-o", str(executable)],
                       capture_output=True, check=True, timeout=30)
        return executable

    @staticmethod
    def pipeline_text():
        return ("Prima misura: 21,5 gradi Celsius e umidità 48 per cento. "
                "Seconda misura: 19 gradi Celsius e umidità 55 per cento. "
                "Terza misura: 12,50 chilogrammi e variazione meno 1,5 per cento. "
                "Quarta misura: costo 9,99 euro e codice item_42.")

    def pipeline_result(self, result):
        stats = next(line[6:] for line in result.stdout.splitlines() if line.startswith("STATS:")).split(",")
        paths = [Path(line[5:]) for line in result.stdout.splitlines() if line.startswith("PATH:")]
        for path in paths:
            self.assertFalse(path.parent.exists(), "the private WAV directory must be removed")
        self.assertLessEqual(len(set(paths)), 2, "keep only current/next WAV slots")
        return [float(value) for value in stats]

    def test_progressive_speech_is_ordered_bounded_and_overlaps_playback(self):
        cfg = self.config("/pipeline", speech_chunk_chars=80)
        expected = json.loads(self.run_voice("--config", cfg, "plan", self.pipeline_text()).stdout)
        result = subprocess.run([str(self.pipeline_fixture()), str(cfg), self.pipeline_text(), "normal"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        begin = [line.split(":", 2) for line in result.stdout.splitlines() if line.startswith("BEGIN:")]
        end = [float(line[4:]) for line in result.stdout.splitlines() if line.startswith("END:")]
        self.assertEqual([parts[2] for parts in begin], expected)
        self.assertEqual([event[1] for event in Server.speech_events], expected)
        self.assertLess(Server.speech_events[1][0], end[0], "synthesize next before playback ends")
        for index in range(2, len(expected)):
            self.assertGreaterEqual(Server.speech_events[index][0], end[index - 2], "prefetch only one ahead")
        total, played, first, synthesis, playback, elapsed = self.pipeline_result(result)
        self.assertEqual(total, len(expected))
        self.assertEqual(played, total)
        self.assertGreater(first, 60)
        self.assertLess(first, elapsed)
        self.assertLess(elapsed + 30, synthesis + playback, "synthesis and playback must overlap")

    def test_progressive_speech_joins_on_failure_and_cancellation(self):
        fixture = self.pipeline_fixture()
        for prefix, mode, error, played in [("/pipeline-fail", "normal", "HTTP 503", 1),
                                             ("/pipeline-delay", "cancel", "interrupted", 0)]:
            with self.subTest(mode=prefix):
                with Server.lock:
                    Server.requests.clear()
                    Server.speech_events.clear()
                cfg = self.config(prefix, speech_chunk_chars=80)
                before = time.monotonic()
                result = subprocess.run([str(fixture), str(cfg), self.pipeline_text(), mode],
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                stats = self.pipeline_result(result)
                self.assertEqual(stats[1], played)
                self.assertEqual(len(Server.speech_events), 2, "never synthesize later chunks after error/cancel")
                self.assertLess(time.monotonic() - before, 2)

    def test_progressive_speech_snapshots_dictionary_for_whole_reply(self):
        dictionary = self.dictionary({"misura": "misùra"})
        cfg = self.config("/pipeline", speech_chunk_chars=80, pronunciation_dictionary="pronunciation.it.json")
        expected = json.loads(self.run_voice("--config", cfg, "plan", self.pipeline_text()).stdout)
        result = subprocess.run([str(self.pipeline_fixture()), str(cfg), self.pipeline_text(), "dictionary"],
                                capture_output=True, text=True, timeout=10,
                                env={**os.environ, "QW3_PIPELINE_DICTIONARY": str(dictionary)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.pipeline_result(result)
        self.assertEqual([event[1] for event in Server.speech_events], expected)
        self.assertIn("MODIFICATO", dictionary.read_text())
        self.assertTrue(all("misùra" in event[1] for event in Server.speech_events))

    def test_visual_data_fallback_preserves_values_across_domains(self):
        fixtures = [
            ("## 📺 La7\n| Ora | Programma | Genere | Durata |\n|---|---|---|---|\n"
             "|18:30|Ignoto X|programma tv|90 min|\n|20:00|Tg La7|programma tv|35 min|",
             "La7. Ora: 18:30. Programma: Ignoto X. Genere: programma tv. Durata: 90 min. "
             "Ora: 20:00. Programma: Tg La7. Genere: programma tv. Durata: 35 min."),
            ("## Offerte\nProdotto | Prezzo | Variazione\n:--- | ---: | :---:\n"
             "A | 9,99 € | -1,5%\nB | 15,20 € | +2%",
             "Offerte. Prodotto: A. Prezzo: 9,99 €. Variazione: -1,5%. "
             "Prodotto: B. Prezzo: 15,20 €. Variazione: +2%."),
            ("| Nome | Valore |\n|---|---|\n|A\\|B|12 kg|\n\nNote: misure confermate.",
             "Nome: A|B. Valore: 12 kg. Note: misure confermate."),
            ("## Misure\n- Temperatura: 21,5 °C\n- Umidità: 48%\n---\nFine.",
             "Misure. Temperatura: 21,5 °C. Umidità: 48%. Fine."),
            ("A | B\nQuesta non è una tabella.", "A | B Questa non è una tabella."),
            ("**Costo**: 9,99 €. Operazione 2*3; identificatore item_42.",
             "Costo: 9,99 €. Operazione 2*3; identificatore item_42."),
            ("```\n| Segreto | Valore |\n|---|---|\n|no|42|\n```\nCiao.", "Ciao."),
        ]
        for raw, expected in fixtures:
            with self.subTest(raw=raw):
                result = self.run_voice("prepare", raw).stdout.strip()
                self.assertEqual(result, expected)

    def dictionary(self, replacements):
        path = self.tmp / "pronunciation.it.json"
        path.write_text(json.dumps({"version": 1, "replacements": replacements}, ensure_ascii=False))
        return path

    def test_pronunciation_literal_words_phrases_case_and_no_cascade(self):
        self.dictionary({"principi": "princìpi", "principi generali": "prìncipi generali",
                         "princìpi": "non deve essere applicato", "C++": "ci più più",
                         "caffè": "càffè"})
        cfg = self.config(pronunciation_dictionary="pronunciation.it.json")
        fixtures = [
            ("principi, PRINCIPI; Principi!", "princìpi, PRINCÌPI; Princìpi!"),
            ("principi generali e principi.", "prìncipi generali e princìpi."),
            ("principiante principi2 Xprincipi principi_42", "principiante principi2 Xprincipi principi_42"),
            ("C++ e C+++", "CI PIÙ PIÙ e CI PIÙ PIÙ+"),
            (unicodedata.normalize("NFD", "Un caffè."), "Un càffè."),
        ]
        for raw, expected in fixtures:
            with self.subTest(raw=raw):
                self.assertEqual(self.run_voice("--config", cfg, "pronounce", raw).stdout.strip(), expected)
        # Only the speech payload changes; the request text is passed literally,
        # without interpreting replacement strings as regular-expression templates.
        self.dictionary({"principi": "princìpi $1"})
        self.run_voice("--config", cfg, "speak", "principi", self.tmp / "corrected.wav")
        self.assertEqual(json.loads(Server.requests[-1][2])["input"], "princìpi $1")

    def test_pronunciation_empty_disabled_and_invalid_files(self):
        self.dictionary({})
        raw = "Ciao, perché sì?"
        for settings in [{}, {"pronunciation_dictionary": ""},
                         {"pronunciation_dictionary": "pronunciation.it.json"}]:
            cfg = self.config(**settings)
            self.assertEqual(self.run_voice("--config", cfg, "pronounce", raw).stdout.strip(), raw)
        cfg = self.config(pronunciation_dictionary="pronunciation.it.json")
        for document in [{"version": 2, "replacements": {}},
                         {"version": True, "replacements": {}},
                         {"version": 1, "replacements": []},
                         {"version": 1, "replacements": {"": "ciao"}},
                         {"version": 1, "replacements": {"ciao": 3}},
                         {"version": 1, "replacements": {"ciao": ""}},
                         {"version": 1, "replacements": {"ciao": "a\nb"}},
                         {"version": 1, "replacements": {"ciao": "ciaò", "CIAO": "ciào"}},
                         {"version": 1, "replacements": {}, "typo": True}]:
            with self.subTest(document=document):
                (self.tmp / "pronunciation.it.json").write_text(json.dumps(document))
                self.run_voice("--config", cfg, "pronounce", raw, ok=False)
        (self.tmp / "pronunciation.it.json").write_text("{bad json")
        self.run_voice("--config", cfg, "pronounce", raw, ok=False)
        (self.tmp / "pronunciation.it.json").unlink()
        self.run_voice("--config", cfg, "pronounce", raw, ok=False)

    def test_pronunciation_reload_in_same_client(self):
        self.dictionary({"principi": "princìpi"})
        cfg = self.config(pronunciation_dictionary="pronunciation.it.json")
        # A tiny native caller keeps one client open while Python edits its file.
        source = self.tmp / "reload.c"
        source.write_text('''
#include "qw3_voice.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    (void)argc;
    qw3_voice *voice = NULL;
    char error[512] = {0};
    if (qw3_voice_open(&voice, argv[1], error, sizeof(error))) return 1;
    for (int i = 0; i < 2; i++) {
        char *text = qw3_voice_apply_pronunciation(voice, "principi", error, sizeof(error));
        if (!text) { fprintf(stderr, "%s", error); return 2; }
        puts(text); free(text); fflush(stdout);
        if (!i && getchar() == EOF) return 3;
    }
    qw3_voice_close(voice);
    return 0;
}
''')
        executable = self.tmp / "reload"
        subprocess.run(["cc", "-I", str(ROOT), str(source), str(ROOT / "qw3_voice.o"),
                        "-framework", "Foundation", "-framework", "AVFoundation",
                        "-o", str(executable)], capture_output=True, check=True, timeout=30)
        process = subprocess.Popen([str(executable), str(cfg)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "princìpi")
            self.dictionary({"principi": "prìncipi"})
            output, error = process.communicate("\n", timeout=10)
            self.assertEqual(process.returncode, 0, error)
            self.assertEqual(output.strip(), "prìncipi")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_service_and_payload_failures(self):
        for mode, command, args, expected in [
            ("/missing", "check", [], "no configured model"),
            ("/bad-json", "check", [], "invalid JSON"),
            ("/unready", "check", [], "not ready"),
            ("/failure", "transcribe", [self.wav], "HTTP 503"),
            ("/empty", "transcribe", [self.wav], "no speech"),
            ("/bad-wav", "speak", ["Ciao", self.tmp / "bad.wav"], "did not return WAV"),
        ]:
            with self.subTest(mode=mode):
                result = self.run_voice("--config", self.config(mode), command, *args, ok=False)
                self.assertIn(expected, result.stderr)
        self.assertFalse((self.tmp / "bad.wav").exists())

    def test_config_validation(self):
        for settings in [{"max_record_seconds": 0}, {"request_timeout_seconds": True},
                         {"request_timeout_seconds": "100"}, {"voice": None},
                         {"server_url": "file:///tmp/test"}, {"unknown_key": "value"}]:
            with self.subTest(settings=settings):
                self.run_voice("--config", self.config(**settings), "check", ok=False)
        for speed in [0, -1, 0.49, 2.01, True, "0.85", None]:
            with self.subTest(tts_speed=speed):
                self.run_voice("--config", self.config(tts_speed=speed), "check", ok=False)

    def endpoint_audio(self, segments):
        """20 ms-aligned silence/constant-energy fixtures; not speech classifiers."""
        path = self.tmp / "endpoint.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            for seconds, db in segments:
                frames = round(seconds * 16000)
                amplitude = 0 if db <= -160 else 32767 * math.sqrt(2) * 10 ** (db / 20)
                period = struct.pack("<80h", *[round(amplitude * math.sin(2 * math.pi * i / 80)) for i in range(80)])
                wav.writeframes((period * ((frames + 79) // 80))[:frames * 2])
        return path

    def test_endpoint_silence_activity_and_hysteresis(self):
        cases = [
            ([(0.6, -160), (0.4, -25), (1, -160)], "silence", True, 1.4),
            ([(0.4, -25), (0.2, -160), (0.4, -25), (1, -160)], "silence", True, 1.4),
            ([(0.2, -25), (0.4, -43), (1, -160)], "silence", True, 1.0),
            ([(2, -160)], "no_speech", False, 1.0),
            ([(0.04, -25), (2, -160)], "no_speech", False, 1.0),
            ([(2, -55)], "no_speech", False, 1.0),
        ]
        cfg = self.config(silence_seconds=0.4, speech_start_timeout_seconds=1)
        for segments, reason, detected, elapsed in cases:
            with self.subTest(segments=segments):
                audio = self.endpoint_audio(segments)
                report = json.loads(self.run_voice("--config", cfg, "endpoint", audio).stdout)
                self.assertEqual(report["stop_reason"], reason)
                self.assertEqual(report["speech_detected"], detected)
                self.assertAlmostEqual(report["stop_seconds"], elapsed, delta=0.021)
        self.assertFalse(Server.requests, "file analysis must not contact audio.cpp")

    def test_endpoint_limits_disabled_and_short_gaps(self):
        audio = self.endpoint_audio([(3, -25)])
        cfg = self.config(max_record_seconds=1)
        report = json.loads(self.run_voice("--config", cfg, "endpoint", audio).stdout)
        self.assertEqual(report["stop_reason"], "max_duration")
        self.assertEqual(report["stop_seconds"], 1)
        cfg = self.config(auto_stop=False, max_record_seconds=3)
        audio = self.endpoint_audio([(2, -160)])
        report = json.loads(self.run_voice("--config", cfg, "endpoint", audio).stdout)
        self.assertFalse(report["automatic"])
        self.assertEqual(report["stop_reason"], "end_of_file")
        cfg = self.config(silence_seconds=0.4, speech_start_timeout_seconds=1)
        audio = self.endpoint_audio([(0.1, -25), (0.04, -160), (0.1, -25), (1, -160)])
        report = json.loads(self.run_voice("--config", cfg, "endpoint", audio).stdout)
        self.assertEqual(report["stop_reason"], "silence")
        self.assertTrue(report["speech_detected"])
        self.assertAlmostEqual(report["stop_seconds"], 0.64, delta=0.021)

    def test_endpoint_config_validation(self):
        for settings in [{"auto_stop": "true"}, {"auto_stop": 1},
                         {"speech_threshold_db": -90}, {"speech_threshold_db": 0},
                         {"silence_seconds": 0}, {"silence_seconds": True},
                         {"min_speech_seconds": 0.001},
                         {"speech_start_timeout_seconds": 0.2, "min_speech_seconds": 0.5},
                         {"max_record_seconds": 1, "min_speech_seconds": 2}]:
            with self.subTest(settings=settings):
                self.run_voice("--config", self.config(**settings), "endpoint", self.wav, ok=False)

    def test_timeout_and_interrupt(self):
        cfg = self.config("/delay", request_timeout_seconds=1)
        before = time.monotonic()
        self.run_voice("--config", cfg, "speak", "Ciao", self.tmp / "timeout.wav", ok=False)
        self.assertLess(time.monotonic() - before, 2)
        cfg = self.config("/delay", request_timeout_seconds=10)
        process = subprocess.Popen([str(VOICE), "--config", str(cfg), "speak", "Ciao", str(self.tmp / "cancel.wav")],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.3)
            process.send_signal(signal.SIGINT)
            _, error = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 130, error)
            self.assertIn("interrupted", error)
            self.assertFalse((self.tmp / "cancel.wav").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    @unittest.skipUnless(os.environ.get("QW3_VOICE_TEST_MODEL"), "set QW3_VOICE_TEST_MODEL for a real LLM turn")
    def test_real_agent_spoken_data_style_across_domains(self):
        cfg = self.config()
        fixtures = [
            ("Confronta in italiano due offerte internet: Base costa 20 euro al mese "
             "per 100 megabit al secondo; Plus costa 30 euro al mese per 300 megabit al secondo. "
             "Usa solo questi dati, senza consultare il web.",
             [r"\bbase\b", r"\bplus\b", r"\b(?:20|venti)\b", r"\b(?:30|trenta)\b",
              r"\b(?:100|cento)\b", r"\b(?:300|trecento)\b", r"euro|€", r"megabit|mbps"]),
            ("Riassumi in italiano queste misure: in cucina 21,5 gradi Celsius e umidità 48%; "
             "in camera 19 gradi Celsius e umidità 55%. Usa solo questi dati.",
             [r"cucina", r"camera", r"21[,.]5|ventuno virgola cinque", r"\b(?:19|diciannove)\b",
              r"\b(?:48|quarantotto)\b", r"\b(?:55|cinquantacinque)\b", r"gradi|celsius|°", r"percento|per cento|%"]),
        ]
        for prompt, facts in fixtures:
            with self.subTest(prompt=prompt):
                with Server.lock:
                    Server.requests.clear()
                result = subprocess.run([
                    str(AGENT), "--voice-config", str(cfg), "-m", os.environ["QW3_VOICE_TEST_MODEL"],
                    "--store-dir", str(self.tmp / "store"), "--ctx", "2048", "--nothink", "--no-tools",
                    "--temp", "0", "--no-prefix-cache", "-n", "256", "-p", prompt
                ], capture_output=True, text=True, timeout=180, cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stderr)
                # Check the model's raw answer, not only the cleaned TTS payload.
                answer = result.stdout.strip()
                self.assertNotIn("|", answer)
                self.assertNotIn("```", answer)
                self.assertNotRegex(answer, r"(?m)^\s*#")
                for fact in facts:
                    self.assertRegex(answer.lower(), fact)
                speeches = [json.loads(body) for path, _, body in Server.requests if path.endswith("/speech")]
                self.assertTrue(speeches, result.stderr)
                spoken = " ".join(speech["input"] for speech in speeches)
                for fact in facts:
                    self.assertRegex(spoken.lower(), fact)

    @unittest.skipUnless(os.environ.get("QW3_VOICE_TEST_MODEL"), "set QW3_VOICE_TEST_MODEL for a real LLM turn")
    def test_real_agent_continuous_turns_and_cancel_without_microphone(self):
        executable = self.tmp / "voice-loop"
        subprocess.run(["cc", "-O1", "-fobjc-arc", "-I", str(ROOT),
                        str(ROOT / "tests/test_voice_loop.m"),
                        *[str(ROOT / name) for name in ["qw3_voice.o", "qw3_metal_core.o", "qw3_metal.o", "linenoise_qw3.o"]],
                        "-framework", "Foundation", "-framework", "Metal", "-framework", "AVFoundation",
                        "-lm", "-pthread", "-o", str(executable)],
                       capture_output=True, check=True, timeout=60)
        cfg = self.config()
        master, slave = pty.openpty()
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "QW3_VOICE_LOOP_WAV": str(self.wav)}
        process = subprocess.Popen([
            str(executable), "--voice-config", str(cfg), "-m", os.environ["QW3_VOICE_TEST_MODEL"],
            "--store-dir", str(self.tmp / "store"), "--ctx", "2048", "--nothink", "--no-tools",
            "--temp", "0", "--no-prefix-cache", "-n", "64"
        ], stdin=slave, stdout=slave, stderr=slave, env=env, cwd=ROOT)
        os.close(slave)
        output = b""
        quit_sent = False
        try:
            os.write(master, b"/speed 0.85\n/speed nan\n/speed\n/listen\n")
            deadline = time.monotonic() + 120
            while process.poll() is None and time.monotonic() < deadline:
                if select.select([master], [], [], 0.2)[0]:
                    try:
                        output += os.read(master, 65536)
                    except OSError:
                        break
                if b"TEST_VOICE_RECORD 3" in output and not quit_sent:
                    os.write(master, b"/quit\n")
                    quit_sent = True
            process.wait(timeout=5)
            while select.select([master], [], [], 0)[0]:
                try:
                    chunk = os.read(master, 65536)
                    if not chunk:
                        break
                    output += chunk
                except OSError:
                    break
            text = output.decode("utf-8", errors="replace")
            self.assertEqual(process.returncode, 0, text)
            self.assertTrue(quit_sent, text)
            self.assertEqual(text.count("TEST_VOICE_PLAY"), 2, text)
            self.assertEqual(text.count("TEST_VOICE_RECORD"), 3, text)
            self.assertIn("fixture recording cancelled", text)
            self.assertIn("speech speed 0.85", text)
            self.assertIn("use /speed 0.5..2 or /speed default", text)
            self.assertEqual(sum(path.endswith("/transcriptions") for path, _, _ in Server.requests), 2)
            self.assertEqual(sum(path.endswith("/speech") for path, _, _ in Server.requests), 2)
            speeches = [json.loads(body) for path, _, body in Server.requests if path.endswith("/speech")]
            self.assertTrue(all(payload["options"]["speed"] == 0.85 for payload in speeches))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)

    @unittest.skipUnless(os.environ.get("QW3_VOICE_TEST_MODEL"), "set QW3_VOICE_TEST_MODEL for a real LLM turn")
    def test_real_agent_final_answer_reaches_tts(self):
        cfg = self.config()
        result = subprocess.run([
            str(AGENT), "--voice-config", str(cfg), "-m", os.environ["QW3_VOICE_TEST_MODEL"],
            "--store-dir", str(self.tmp / "store"), "--conversation", "voice-fixture",
            "--trace", str(self.tmp / "trace.jsonl"),
            "--ctx", "2048", "--nothink", "--no-tools",
            "--temp", "0", "--no-prefix-cache", "-n", "32", "-p", "Rispondi soltanto: Ciao, sono QW3."
        ], capture_output=True, text=True, timeout=180, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("voice: audio.cpp", result.stderr)
        self.assertNotIn("cannot play", result.stderr)
        self.assertEqual([line for line in result.stderr.splitlines() if line.startswith("voice:")],
                         ["voice: synthesizing final answer..."], result.stderr)
        speeches = [json.loads(body) for path, _, body in Server.requests if path.endswith("/v1/audio/speech")]
        self.assertEqual(len(speeches), 1, result.stderr)
        self.assertIn("Ciao", speeches[0]["input"])
        self.assertNotIn("<think>", speeches[0]["input"])
        self.assertTrue((self.tmp / "store" / "voice-fixture.txt").exists())
        self.assertIn("Spoken interaction:", (self.tmp / "store" / "voice-fixture.txt").read_text())
        trace = [json.loads(line) for line in (self.tmp / "trace.jsonl").read_text().splitlines()]
        audio = next(event for event in trace if event["event"] == "voice_audio")
        self.assertTrue(audio["ok"])
        self.assertEqual(audio["chunks_total"], audio["chunks_played"])
        self.assertGreater(audio["first_chunk_ready_ms"], 0)
        self.assertGreaterEqual(audio["wired_sample_max_bytes"], audio["wired_before_bytes"])
        unloaded = [json.loads(body) for path, _, body in Server.requests if path.endswith("/unload_models")]
        self.assertTrue(unloaded)
        self.assertTrue(all(request == {"model_ids": ["qwen3-asr", "kokoro"]} for request in unloaded))


if __name__ == "__main__":
    unittest.main(verbosity=2)
