#ifndef QW3_VOICE_ENDPOINT_H
#define QW3_VOICE_ENDPOINT_H

#include <stdbool.h>
#include <math.h>

/* Energy-based end-of-utterance gate. This does not classify speech content.
 * Feed monotonic elapsed seconds and RMS dBFS; no audio/model allocations. */
typedef enum {
    QW3_ENDPOINT_RECORDING,
    QW3_ENDPOINT_SILENCE,
    QW3_ENDPOINT_NO_SPEECH,
    QW3_ENDPOINT_LIMIT
} qw3_endpoint_result;

typedef struct {
    bool automatic;
    double threshold_db;
    double min_speech_seconds;
    double silence_seconds;
    double start_timeout_seconds;
    double max_seconds;
} qw3_endpoint_options;

typedef struct {
    qw3_endpoint_options options;
    bool has_speech;
    double candidate_seconds;
    double candidate_gap;
    double last_active;
    double elapsed;
    qw3_endpoint_result result;
} qw3_endpoint;

static inline qw3_endpoint_result qw3_endpoint_feed(qw3_endpoint *s,
                                                     double elapsed, double db) {
    if (s->result != QW3_ENDPOINT_RECORDING) return s->result;
    if (!isfinite(elapsed) || elapsed < s->elapsed) return s->result;
    double dt = elapsed - s->elapsed;
    s->elapsed = elapsed;
    if (!isfinite(db)) db = -160;
    if (s->options.automatic) {
        bool started = s->has_speech || s->candidate_seconds > 0;
        bool active = db >= s->options.threshold_db - (started ? 6 : 0);
        if (active) {
            /* A delayed meter poll is not evidence of sustained activity. */
            s->candidate_seconds += dt > 0.1 ? 0.1 : dt;
            s->candidate_gap = 0;
            s->last_active = elapsed;
            if (s->candidate_seconds + 1e-9 >= s->options.min_speech_seconds)
                s->has_speech = true;
        } else {
            s->candidate_gap += dt;
            if (!s->has_speech && s->candidate_gap > 0.1)
                s->candidate_seconds = 0;
        }
        if (s->has_speech && elapsed - s->last_active + 1e-9 >= s->options.silence_seconds)
            return s->result = QW3_ENDPOINT_SILENCE;
        if (!s->has_speech && elapsed + 1e-9 >= s->options.start_timeout_seconds)
            return s->result = QW3_ENDPOINT_NO_SPEECH;
    }
    if (elapsed + 1e-9 >= s->options.max_seconds)
        return s->result = s->options.automatic && !s->has_speech ?
            QW3_ENDPOINT_NO_SPEECH : QW3_ENDPOINT_LIMIT;
    return s->result;
}

#endif
