/* Real native HTTP/prefetch controller; replace only the playback device.
 * The mock WAV has the spoken text appended after its ordinary PCM payload. */
#include "qw3_voice.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const char *mode;
static bool cancelled;
static unsigned plays;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static bool interrupted(void *ud) { (void)ud; return cancelled; }

static int play(const char *wav, qw3_voice_interrupt_fn interrupt, void *ud,
                char *err, size_t cap) {
    FILE *input = fopen(wav, "rb");
    if (!input) return -1;
    char text[8192] = {0};
    fseek(input, 32044, SEEK_SET);
    fread(text, 1, sizeof(text) - 1, input);
    fclose(input);
    plays++;
    printf("PATH:%s\nBEGIN:%.9f:%s\n", wav, now(), text);
    fflush(stdout);
    if (plays == 1 && !strcmp(mode, "dictionary")) {
        FILE *dictionary = fopen(getenv("QW3_PIPELINE_DICTIONARY"), "w");
        if (!dictionary) return -1;
        fputs("{\"version\":1,\"replacements\":{\"misura\":\"MODIFICATO\"}}", dictionary);
        fclose(dictionary);
    }
    for (int i = 0; i < 15; i++) {
        usleep(10000);
        if (!strcmp(mode, "cancel") && i == 3) cancelled = true;
        if (interrupt && interrupt(ud)) {
            snprintf(err, cap, "fixture playback interrupted");
            return -1;
        }
    }
    printf("END:%.9f\n", now());
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    mode = argv[3];
    qw3_voice *voice = NULL;
    char err[512] = {0};
    if (qw3_voice_open(&voice, argv[1], err, sizeof(err))) return 3;
    qw3_voice_speech_stats stats;
    int rc = qw3_voice_speak(voice, argv[2], play, interrupted, NULL, &stats, err, sizeof(err));
    printf("STATS:%zu,%zu,%.3f,%.3f,%.3f,%.3f\n", stats.chunks_total, stats.chunks_played,
           stats.first_chunk_ready_ms, stats.synthesis_ms, stats.playback_ms, stats.total_ms);
    if (rc) fprintf(stderr, "%s\n", err);
    qw3_voice_close(voice);
    return rc ? 1 : 0;
}
