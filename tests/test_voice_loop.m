/* Exercise the real interactive controller and model, replacing only devices.
 * HTTP, conversation state, model generation and the TTS payload stay real. */
#define QW3_ENABLE_VOICE 1
#define qw3_voice_record fixture_voice_record
#define qw3_voice_play fixture_voice_play
#define main qw3_voice_agent_main
#include "../qw3_agent.c"
#undef main
#undef qw3_voice_record
#undef qw3_voice_play

int fixture_voice_record(qw3_voice *voice, const char *wav,
                          qw3_voice_interrupt_fn interrupt, void *ud,
                          char *err, size_t cap) {
    (void)voice; (void)interrupt; (void)ud;
    static int calls = 0;
    fprintf(stderr, "TEST_VOICE_RECORD %d\n", ++calls);
    if (calls == 3) {
        raise(SIGINT);
        snprintf(err, cap, "fixture recording cancelled");
        return -1;
    }
    const char *source = getenv("QW3_VOICE_LOOP_WAV");
    FILE *input = source ? fopen(source, "rb") : NULL;
    FILE *output = fopen(wav, "wb");
    if (!input || !output) {
        if (input) fclose(input);
        if (output) fclose(output);
        snprintf(err, cap, "cannot open fixture WAV");
        return -1;
    }
    unsigned char bytes[4096];
    size_t n;
    while ((n = fread(bytes, 1, sizeof(bytes), input)) > 0) {
        if (fwrite(bytes, 1, n, output) != n) abort();
    }
    fclose(input);
    fclose(output);
    return 0;
}

int fixture_voice_play(const char *wav, qw3_voice_interrupt_fn interrupt, void *ud,
                        char *err, size_t cap) {
    (void)wav; (void)interrupt; (void)ud; (void)err; (void)cap;
    fprintf(stderr, "TEST_VOICE_PLAY\n");
    return 0;
}

int main(int argc, char **argv) { return qw3_voice_agent_main(argc, argv); }
