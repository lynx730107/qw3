#include <signal.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qw3_voice.h"

static volatile sig_atomic_t cancelled;
static void cancel(int sig) { (void)sig; cancelled = 1; }
static bool interrupted(void *ud) { (void)ud; return cancelled != 0; }

static int usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  qw3-voice --config PATH check\n"
        "  qw3-voice --config PATH record OUTPUT.wav\n"
        "  qw3-voice --config PATH transcribe INPUT.wav\n"
        "  qw3-voice --config PATH speak TEXT OUTPUT.wav\n"
        "  qw3-voice --config PATH say TEXT\n"
        "  qw3-voice --config PATH plan TEXT\n"
        "  qw3-voice --config PATH pronounce TEXT\n"
        "  qw3-voice --config PATH endpoint INPUT.wav\n"
        "  qw3-voice play INPUT.wav\n"
        "  qw3-voice prepare [--thinking] TEXT\n"
        "record stops on silence or Enter; endpoint analyzes a file without a mic.\n"
        "speak writes a WAV without playing it; say uses progressive playback and reports timing.\n"
        "plan prints the speech chunks without generating or playing audio.\n");
    return 1;
}

int main(int argc, char **argv) {
    signal(SIGINT, cancel);
    char error[512] = {0};
    if (argc >= 3 && !strcmp(argv[1], "prepare")) {
        bool thinking = argc == 4 && !strcmp(argv[2], "--thinking");
        if (argc != (thinking ? 4 : 3)) return usage();
        char *text = qw3_voice_prepare_text(argv[thinking ? 3 : 2], thinking);
        if (!text) return 1;
        puts(text);
        free(text);
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "play")) {
        int rc = qw3_voice_play(argv[2], interrupted, NULL, error, sizeof(error));
        if (rc) fprintf(stderr, "voice: %s\n", error);
        return cancelled ? 130 : rc != 0;
    }
    if (argc < 4 || strcmp(argv[1], "--config")) return usage();
    const char *command = argv[3];
    if ((!strcmp(command, "check") && argc != 4) ||
        (!strcmp(command, "speak") && argc != 6) ||
        ((!strcmp(command, "record") || !strcmp(command, "transcribe") ||
          !strcmp(command, "pronounce") || !strcmp(command, "endpoint") ||
          !strcmp(command, "say") || !strcmp(command, "plan")) && argc != 5))
        return usage();
    qw3_voice *voice = NULL;
    if (qw3_voice_open(&voice, argv[2], error, sizeof(error)) != 0) {
        fprintf(stderr, "voice: %s\n", error);
        return 1;
    }
    int rc;
    if (!strcmp(command, "check")) {
        rc = qw3_voice_check(voice, error, sizeof(error));
        if (!rc) puts("audio.cpp ready; ASR/TTS IDs configured (inference not checked)");
    } else if (!strcmp(command, "record")) {
        rc = qw3_voice_record(voice, argv[4], interrupted, NULL, error, sizeof(error));
    } else if (!strcmp(command, "transcribe")) {
        char *text = NULL;
        rc = qw3_voice_transcribe(voice, argv[4], &text, interrupted, NULL, error, sizeof(error));
        if (!rc) puts(text);
        free(text);
    } else if (!strcmp(command, "endpoint")) {
        char *report = NULL;
        rc = qw3_voice_analyze_endpoint(voice, argv[4], &report, error, sizeof(error));
        if (!rc) puts(report);
        free(report);
    } else if (!strcmp(command, "plan")) {
        char *prose = qw3_voice_prepare_text(argv[4], false);
        char *plan = qw3_voice_plan(voice, prose, error, sizeof(error));
        free(prose);
        rc = plan ? 0 : -1;
        if (plan) puts(plan);
        free(plan);
    } else if (!strcmp(command, "say")) {
        char *prose = qw3_voice_prepare_text(argv[4], false);
        qw3_voice_speech_stats stats = {0};
        rc = qw3_voice_speak(voice, prose, NULL, interrupted, NULL, &stats, error, sizeof(error));
        free(prose);
        printf("{\"chunks_total\":%zu,\"chunks_played\":%zu,\"first_chunk_ready_ms\":%.3f,"
               "\"synthesis_ms\":%.3f,\"playback_ms\":%.3f,\"total_ms\":%.3f,"
               "\"wired_before_bytes\":%" PRIu64 ",\"wired_sample_max_bytes\":%" PRIu64 "}\n",
               stats.chunks_total, stats.chunks_played, stats.first_chunk_ready_ms,
               stats.synthesis_ms, stats.playback_ms, stats.total_ms,
               stats.wired_before_bytes, stats.wired_sample_max_bytes);
    } else if (!strcmp(command, "pronounce")) {
        char *prose = qw3_voice_prepare_text(argv[4], false);
        char *text = qw3_voice_apply_pronunciation(voice, prose, error, sizeof(error));
        free(prose);
        rc = text ? 0 : -1;
        if (text) puts(text);
        free(text);
    } else if (!strcmp(command, "speak")) {
        char *text = qw3_voice_prepare_text(argv[4], false);
        rc = qw3_voice_synthesize(voice, text, argv[5], interrupted, NULL, error, sizeof(error));
        free(text);
    } else {
        qw3_voice_close(voice);
        return usage();
    }
    if (rc) fprintf(stderr, "voice: %s\n", error);
    qw3_voice_close(voice);
    return cancelled ? 130 : rc != 0;
}
