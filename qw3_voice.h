#ifndef QW3_VOICE_H
#define QW3_VOICE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Optional macOS audio.cpp client. The caller owns returned strings (free()). */
typedef struct qw3_voice qw3_voice;
typedef bool (*qw3_voice_interrupt_fn)(void *ud);
typedef int (*qw3_voice_player_fn)(const char *wav,
    qw3_voice_interrupt_fn interrupt, void *ud, char *err, size_t cap);

typedef struct {
    size_t chunks_total;
    size_t chunks_played;
    double first_chunk_ready_ms; /* Not a device/audio-onset measurement. */
    double synthesis_ms;         /* Sum; overlaps playback. */
    double playback_ms;
    double total_ms;
    uint64_t wired_before_bytes;     /* System-wide, not process RSS. */
    uint64_t wired_sample_max_bytes; /* Samples at chunk boundaries, not a peak. */
} qw3_voice_speech_stats;

int qw3_voice_open(qw3_voice **out, const char *config, char *err, size_t cap);
void qw3_voice_close(qw3_voice *voice);
/* 0 means use the server default. Explicit rates must be between 0.5 and 2.
 * Changes affect subsequent syntheses on this client, without restarting. */
double qw3_voice_get_speed(qw3_voice *voice);
int qw3_voice_set_speed(qw3_voice *voice, double speed, char *err, size_t cap);
int qw3_voice_check(qw3_voice *voice, char *err, size_t cap);
int qw3_voice_release_models(qw3_voice *voice,
                             qw3_voice_interrupt_fn interrupt, void *ud,
                             char *err, size_t cap);
int qw3_voice_record(qw3_voice *voice, const char *wav,
                     qw3_voice_interrupt_fn interrupt, void *ud,
                     char *err, size_t cap);
/* record: 0 = captured utterance, 1 = no speech-level activity (WAV removed),
 * -1 = error/cancel. Analyze runs the same energy gate on a file, without mic. */
int qw3_voice_analyze_endpoint(qw3_voice *voice, const char *wav, char **report,
                               char *err, size_t cap);
int qw3_voice_transcribe(qw3_voice *voice, const char *wav, char **text,
                         qw3_voice_interrupt_fn interrupt, void *ud,
                         char *err, size_t cap);
int qw3_voice_synthesize(qw3_voice *voice, const char *text, const char *wav,
                         qw3_voice_interrupt_fn interrupt, void *ud,
                         char *err, size_t cap);
int qw3_voice_play(const char *wav, qw3_voice_interrupt_fn interrupt, void *ud,
                   char *err, size_t cap);
/* Plan/speak prepared final prose. Dictionary rules are snapshotted once per
 * reply, before splitting. Plan returns a malloc-owned JSON array of strings.
 * Speak keeps at most one synthesis ahead of playback, cleans all temporary
 * WAVs and joins pending work on error/cancel. Call on the main thread with
 * LLM Metal weights inactive. NULL player selects native AVAudioPlayer. */
char *qw3_voice_plan(qw3_voice *voice, const char *text, char *err, size_t cap);
int qw3_voice_speak(qw3_voice *voice, const char *text,
    qw3_voice_player_fn player, qw3_voice_interrupt_fn interrupt, void *ud,
    qw3_voice_speech_stats *stats, char *err, size_t cap);
/* Returns final prose, preserving inline data; excludes thinking/code fences. */
char *qw3_voice_prepare_text(const char *text, bool thinking_enabled);
/* Reload the configured dictionary and return a malloc-owned speech-only copy.
 * Literal whole-word/phrase matches, case-insensitive, longest first; no cascade.
 * Returns NULL and sets err on an invalid/unreadable configured dictionary. */
char *qw3_voice_apply_pronunciation(qw3_voice *voice, const char *text,
                                    char *err, size_t cap);

#endif
