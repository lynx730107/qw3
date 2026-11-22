#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <mach/mach.h>
#include <poll.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "qw3_voice.h"
#include "qw3_voice_endpoint.h"

struct qw3_voice {
    void *config; /* Retained NSDictionary, hidden from C callers. */
    double speed; /* 0 preserves audio.cpp's configured default. */
};

static int fail(char *err, size_t cap, NSString *message) {
    if (err && cap) snprintf(err, cap, "%s", message.UTF8String ?: "voice error");
    return -1;
}

static NSDictionary *config(qw3_voice *v) {
    return (__bridge NSDictionary *)v->config;
}

static BOOL interrupted(qw3_voice_interrupt_fn fn, void *ud) {
    return fn && fn(ud);
}

static void tick(void) {
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.025]];
}

static uint64_t wired_memory_bytes(void) {
    vm_statistics64_data_t stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_t host = mach_host_self();
    kern_return_t result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&stats, &count);
    mach_port_deallocate(mach_task_self(), host);
    return result == KERN_SUCCESS ? (uint64_t)stats.wire_count * (uint64_t)getpagesize() : 0;
}

/* Metal's driver can unwire released no-copy views asynchronously. Starting
 * CoreAudio before that happens fails on the 24 GB test machine. Wait only
 * while wired pages exceed 75% of physical RAM, with a bounded, cancellable wait. */
static int wait_for_audio_memory(qw3_voice_interrupt_fn interrupt, void *ud,
                                  char *err, size_t cap) {
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:5];
    for (;;) {
        if (interrupted(interrupt, ud)) return fail(err, cap, @"audio interrupted");
        vm_statistics64_data_t stats;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        host_t host = mach_host_self();
        kern_return_t result = host_statistics64(host, HOST_VM_INFO64,
                                                  (host_info64_t)&stats, &count);
        mach_port_deallocate(mach_task_self(), host);
        if (result != KERN_SUCCESS) return 0;
        uint64_t wired = (uint64_t)stats.wire_count * (uint64_t)getpagesize();
        if (wired < NSProcessInfo.processInfo.physicalMemory * 3 / 4) return 0;
        if (deadline.timeIntervalSinceNow <= 0)
            return fail(err, cap, @"too much wired memory remains for audio; reduce other GPU workloads");
        tick();
    }
}

static NSDictionary *load_pronunciation(NSString *path, char *err, size_t cap) {
    if (!path.length) return @{};
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    if (!data) {
        fail(err, cap, [NSString stringWithFormat:@"cannot read pronunciation dictionary %@: %@", path, error.localizedDescription]);
        return nil;
    }
    id input = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![input isKindOfClass:NSDictionary.class] ||
        ![input[@"version"] isKindOfClass:NSNumber.class] ||
        CFGetTypeID((__bridge CFTypeRef)input[@"version"]) == CFBooleanGetTypeID() ||
        ![input[@"version"] isEqual:@1] ||
        ![input[@"replacements"] isKindOfClass:NSDictionary.class]) {
        fail(err, cap, @"pronunciation dictionary must contain version: 1 and a replacements object");
        return nil;
    }
    for (NSString *key in input) {
        if (![@[@"version", @"replacements"] containsObject:key]) {
            fail(err, cap, [@"unknown pronunciation dictionary field: " stringByAppendingString:key]);
            return nil;
        }
    }
    NSMutableDictionary *rules = [NSMutableDictionary dictionary];
    for (NSString *key in input[@"replacements"]) {
        id value = input[@"replacements"][key];
        if (![key stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].length ||
            ![value isKindOfClass:NSString.class] ||
            ![value stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].length ||
            [key rangeOfCharacterFromSet:NSCharacterSet.controlCharacterSet].location != NSNotFound ||
            [value rangeOfCharacterFromSet:NSCharacterSet.controlCharacterSet].location != NSNotFound) {
            fail(err, cap, @"pronunciation replacements must have nonempty text keys/values without control characters");
            return nil;
        }
        NSString *normalized = key.precomposedStringWithCanonicalMapping.lowercaseString;
        if (rules[normalized]) {
            fail(err, cap, [@"duplicate pronunciation match after case/Unicode normalization: " stringByAppendingString:key]);
            return nil;
        }
        rules[normalized] = [value precomposedStringWithCanonicalMapping];
    }
    return rules;
}

char *qw3_voice_apply_pronunciation(qw3_voice *v, const char *raw, char *err, size_t cap) {
    @autoreleasepool {
        NSString *text = raw ? [NSString stringWithUTF8String:raw] : @"";
        if (!text) { fail(err, cap, @"speech text is not valid UTF-8"); return NULL; }
        NSDictionary *rules = load_pronunciation(config(v)[@"pronunciation_dictionary"], err, cap);
        if (!rules) return NULL;
        if (!rules.count) return strdup(text.UTF8String);
        text = text.precomposedStringWithCanonicalMapping;
        NSArray *keys = [[rules allKeys] sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
            if (a.length != b.length) return a.length > b.length ? NSOrderedAscending : NSOrderedDescending;
            return [a compare:b];
        }];
        NSMutableArray *alternatives = [NSMutableArray array];
        for (NSString *key in keys) [alternatives addObject:[NSRegularExpression escapedPatternForString:key]];
        NSString *pattern = [NSString stringWithFormat:@"(?<![\\p{L}\\p{M}\\p{N}_])(?:%@)(?![\\p{L}\\p{M}\\p{N}_])",
                             [alternatives componentsJoinedByString:@"|"]];
        NSError *error = nil;
        NSRegularExpression *rx = [NSRegularExpression regularExpressionWithPattern:pattern options:NSRegularExpressionCaseInsensitive error:&error];
        if (!rx) { fail(err, cap, error.localizedDescription); return NULL; }
        NSMutableString *spoken = [NSMutableString string];
        NSUInteger offset = 0;
        for (NSTextCheckingResult *hit in [rx matchesInString:text options:0 range:NSMakeRange(0, text.length)]) {
            [spoken appendString:[text substringWithRange:NSMakeRange(offset, hit.range.location - offset)]];
            NSString *matched = [text substringWithRange:hit.range];
            NSString *replacement = rules[matched.lowercaseString] ?: matched;
            if (![matched isEqualToString:matched.lowercaseString]) {
                if ([matched isEqualToString:matched.uppercaseString]) {
                    replacement = replacement.uppercaseString;
                } else {
                    NSRange first = [matched rangeOfComposedCharacterSequenceAtIndex:0];
                    if ([[matched substringWithRange:first] rangeOfCharacterFromSet:NSCharacterSet.uppercaseLetterCharacterSet].location != NSNotFound) {
                        NSRange start = [replacement rangeOfComposedCharacterSequenceAtIndex:0];
                        replacement = [[[replacement substringWithRange:start] uppercaseString]
                            stringByAppendingString:[replacement substringFromIndex:NSMaxRange(start)]];
                    }
                }
            }
            [spoken appendString:replacement];
            offset = NSMaxRange(hit.range);
        }
        [spoken appendString:[text substringFromIndex:offset]];
        return strdup(spoken.UTF8String);
    }
}

int qw3_voice_open(qw3_voice **out, const char *path, char *err, size_t cap) {
    @autoreleasepool {
        *out = NULL;
        NSError *error = nil;
        NSData *data = [NSData dataWithContentsOfFile:@(path) options:0 error:&error];
        if (!data) return fail(err, cap, error.localizedDescription);
        id input = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (![input isKindOfClass:NSDictionary.class])
            return fail(err, cap, @"voice config must be a JSON object");
        NSMutableDictionary *c = [@{
            @"server_url": @"http://127.0.0.1:8080",
            @"asr_model": @"qwen3-asr", @"tts_model": @"kokoro",
            @"asr_language": @"Italian", @"tts_language": @"it",
            @"voice": @"if_sara", @"request_timeout_seconds": @120,
            @"pronunciation_dictionary": @"",
            @"tts_speed": @0,
            @"speech_chunk_chars": @320,
            @"auto_stop": @YES, @"silence_seconds": @1.2,
            @"speech_threshold_db": @-40, @"min_speech_seconds": @0.18,
            @"speech_start_timeout_seconds": @8,
            @"max_record_seconds": @30
        } mutableCopy];
        for (NSString *key in input) {
            id value = input[key];
            if (!c[key]) return fail(err, cap, [@"unknown voice setting: " stringByAppendingString:key]);
            BOOL number = [c[key] isKindOfClass:NSNumber.class];
            if ([key isEqualToString:@"auto_stop"]) {
                if (![value isKindOfClass:NSNumber.class] ||
                    CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID())
                    return fail(err, cap, @"auto_stop must be true or false");
            } else if (number) {
                double minimum = 1, maximum = 600;
                if ([key isEqualToString:@"max_record_seconds"]) maximum = 300;
                else if ([key isEqualToString:@"silence_seconds"]) { minimum = 0.2; maximum = 10; }
                else if ([key isEqualToString:@"speech_threshold_db"]) { minimum = -80; maximum = -5; }
                else if ([key isEqualToString:@"min_speech_seconds"]) { minimum = 0.05; maximum = 2; }
                else if ([key isEqualToString:@"speech_start_timeout_seconds"]) { minimum = 0.2; maximum = 60; }
                else if ([key isEqualToString:@"tts_speed"]) { minimum = 0.5; maximum = 2; }
                else if ([key isEqualToString:@"speech_chunk_chars"]) { minimum = 80; maximum = 1000; }
                if (![value isKindOfClass:NSNumber.class] ||
                    CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() ||
                    !isfinite([value doubleValue]) || [value doubleValue] < minimum ||
                    [value doubleValue] > maximum)
                    return fail(err, cap, [@"invalid numeric voice setting: " stringByAppendingString:key]);
                if ([key isEqualToString:@"speech_chunk_chars"] && floor([value doubleValue]) != [value doubleValue])
                    return fail(err, cap, @"speech_chunk_chars must be an integer");
            } else if (![value isKindOfClass:NSString.class] ||
                       (![value length] && ![key isEqualToString:@"pronunciation_dictionary"])) {
                return fail(err, cap, [@"voice setting must be a nonempty string: " stringByAppendingString:key]);
            }
            c[key] = value;
        }
        if ([c[@"min_speech_seconds"] doubleValue] > [c[@"speech_start_timeout_seconds"] doubleValue] ||
            [c[@"min_speech_seconds"] doubleValue] > [c[@"max_record_seconds"] doubleValue])
            return fail(err, cap, @"min_speech_seconds must fit within recording/start timeouts");
        NSString *urlText = c[@"server_url"];
        while ([urlText hasSuffix:@"/"]) urlText = [urlText substringToIndex:urlText.length - 1];
        NSURL *url = [NSURL URLWithString:urlText];
        if (!url.host.length || ![@[@"http", @"https"] containsObject:url.scheme] ||
            url.query || url.fragment || url.user || url.password)
            return fail(err, cap, @"server_url must be an HTTP(S) base URL without credentials, query or fragment");
        c[@"server_url"] = urlText;
        NSString *dictionary = [c[@"pronunciation_dictionary"] stringByExpandingTildeInPath];
        if (dictionary.length) {
            if (!dictionary.isAbsolutePath) {
                NSString *base = [[NSURL fileURLWithPath:@(path)] URLByDeletingLastPathComponent].path;
                dictionary = [base stringByAppendingPathComponent:dictionary];
            }
            c[@"pronunciation_dictionary"] = dictionary.stringByStandardizingPath;
        }
        if (!load_pronunciation(c[@"pronunciation_dictionary"], err, cap)) return -1;
        qw3_voice *v = calloc(1, sizeof(*v));
        if (!v) return fail(err, cap, @"out of memory");
        v->config = (void *)CFBridgingRetain([c copy]);
        v->speed = [c[@"tts_speed"] doubleValue];
        *out = v;
        return 0;
    }
}

void qw3_voice_close(qw3_voice *v) {
    if (!v) return;
    CFRelease(v->config);
    free(v);
}

double qw3_voice_get_speed(qw3_voice *v) { return v->speed; }

int qw3_voice_set_speed(qw3_voice *v, double speed, char *err, size_t cap) {
    if (!isfinite(speed) || (speed != 0 && (speed < 0.5 || speed > 2)))
        return fail(err, cap, @"speech speed must be between 0.5 and 2, or default");
    v->speed = speed;
    return 0;
}

static NSData *request(qw3_voice *v, NSString *route, NSData *body,
                       NSString *contentType, qw3_voice_interrupt_fn interrupt,
                       void *ud, char *err, size_t cap) {
    NSDictionary *c = config(v);
    NSURL *url = [NSURL URLWithString:[c[@"server_url"] stringByAppendingString:route]];
    NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:url];
    req.timeoutInterval = [c[@"request_timeout_seconds"] doubleValue];
    req.HTTPMethod = body ? @"POST" : @"GET";
    req.HTTPBody = body;
    if (contentType) [req setValue:contentType forHTTPHeaderField:@"Content-Type"];
    NSURLSessionConfiguration *sc = [NSURLSessionConfiguration ephemeralSessionConfiguration];
    sc.timeoutIntervalForRequest = req.timeoutInterval;
    sc.timeoutIntervalForResource = req.timeoutInterval;
    NSURLSession *session = [NSURLSession sessionWithConfiguration:sc];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block NSData *result = nil;
    __block NSURLResponse *response = nil;
    __block NSError *error = nil;
    NSURLSessionDataTask *task = [session dataTaskWithRequest:req completionHandler:
        ^(NSData *data, NSURLResponse *resp, NSError *e) {
            result = data; response = resp; error = e;
            dispatch_semaphore_signal(done);
        }];
    [task resume];
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:req.timeoutInterval];
    while (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 50000000)) != 0) {
        if (interrupted(interrupt, ud) || deadline.timeIntervalSinceNow <= 0) {
            [session invalidateAndCancel];
            fail(err, cap, interrupted(interrupt, ud) ? @"voice request interrupted" : @"audio.cpp request timed out");
            return nil;
        }
    }
    [session finishTasksAndInvalidate];
    if (error) { fail(err, cap, error.localizedDescription); return nil; }
    NSInteger status = [(NSHTTPURLResponse *)response statusCode];
    if (status < 200 || status >= 300) {
        NSString *detail = [[NSString alloc] initWithData:result encoding:NSUTF8StringEncoding] ?: @"";
        if (detail.length > 300) detail = [detail substringToIndex:300];
        fail(err, cap, [NSString stringWithFormat:@"audio.cpp HTTP %ld: %@", (long)status, detail]);
        return nil;
    }
    return result;
}

static id json(NSData *data, char *err, size_t cap) {
    if (!data) return nil;
    NSError *error = nil;
    id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![value isKindOfClass:NSDictionary.class]) {
        fail(err, cap, @"audio.cpp returned an invalid JSON object");
        return nil;
    }
    return value;
}

int qw3_voice_check(qw3_voice *v, char *err, size_t cap) {
    @autoreleasepool {
        NSDictionary *health = json(request(v, @"/health", nil, nil, NULL, NULL, err, cap), err, cap);
        if (!health) return -1;
        if (![health[@"status"] isEqual:@"ok"]) return fail(err, cap, @"audio.cpp is not ready");
        NSDictionary *models = json(request(v, @"/v1/models", nil, nil, NULL, NULL, err, cap), err, cap);
        if (!models) return -1;
        if (![models[@"data"] isKindOfClass:NSArray.class])
            return fail(err, cap, @"audio.cpp model list is missing data[]");
        NSMutableSet *ids = [NSMutableSet set];
        for (id entry in models[@"data"]) {
            if ([entry isKindOfClass:NSDictionary.class] && [entry[@"id"] isKindOfClass:NSString.class])
                [ids addObject:entry[@"id"]];
        }
        for (NSString *key in @[@"asr_model", @"tts_model"]) {
            if (![ids containsObject:config(v)[key]])
                return fail(err, cap, [NSString stringWithFormat:@"audio.cpp has no configured model '%@'", config(v)[key]]);
        }
        return 0;
    }
}

int qw3_voice_release_models(qw3_voice *v, qw3_voice_interrupt_fn interrupt,
                             void *ud, char *err, size_t cap) {
    @autoreleasepool {
        NSDictionary *c = config(v);
        NSData *body = [NSJSONSerialization dataWithJSONObject:
            @{@"model_ids": @[c[@"asr_model"], c[@"tts_model"]]} options:0 error:NULL];
        NSDictionary *reply = json(request(v, @"/v1/tasks/unload_models", body,
            @"application/json", interrupt, ud, err, cap), err, cap);
        if (!reply) return -1;
        if (![reply[@"unloaded"] isKindOfClass:NSArray.class] ||
            ![reply[@"not_found"] isKindOfClass:NSArray.class] || [reply[@"not_found"] count])
            return fail(err, cap, @"audio.cpp could not release the configured audio models");
        return 0;
    }
}

static qw3_endpoint endpoint(qw3_voice *v) {
    NSDictionary *c = config(v);
    qw3_endpoint state = { .options = {
        .automatic = [c[@"auto_stop"] boolValue],
        .threshold_db = [c[@"speech_threshold_db"] doubleValue],
        .min_speech_seconds = [c[@"min_speech_seconds"] doubleValue],
        .silence_seconds = [c[@"silence_seconds"] doubleValue],
        .start_timeout_seconds = [c[@"speech_start_timeout_seconds"] doubleValue],
        .max_seconds = [c[@"max_record_seconds"] doubleValue]
    }};
    return state;
}

int qw3_voice_analyze_endpoint(qw3_voice *v, const char *wav, char **report,
                               char *err, size_t cap) {
    @autoreleasepool {
        *report = NULL;
        NSError *error = nil;
        AVAudioFile *file = [[AVAudioFile alloc] initForReading:[NSURL fileURLWithPath:@(wav)]
            commonFormat:AVAudioPCMFormatFloat32 interleaved:NO error:&error];
        if (!file) return fail(err, cap, error.localizedDescription);
        AVAudioFormat *format = file.processingFormat;
        AVAudioFrameCount frames = (AVAudioFrameCount)(format.sampleRate * 0.02);
        if (!frames || !format.channelCount) return fail(err, cap, @"invalid audio format");
        AVAudioPCMBuffer *buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format frameCapacity:frames];
        if (!buffer) return fail(err, cap, @"cannot allocate endpoint analysis buffer");
        qw3_endpoint state = endpoint(v);
        double peak = -160;
        uint64_t readFrames = 0;
        while (state.result == QW3_ENDPOINT_RECORDING && file.framePosition < file.length) {
            if (![file readIntoBuffer:buffer error:&error]) return fail(err, cap, error.localizedDescription);
            if (!buffer.frameLength) break;
            double energy = 0;
            for (AVAudioChannelCount channel = 0; channel < format.channelCount; channel++)
                for (AVAudioFrameCount i = 0; i < buffer.frameLength; i++) {
                    double x = buffer.floatChannelData[channel][i];
                    if (isfinite(x)) energy += x * x;
                }
            double rms2 = energy / ((double)buffer.frameLength * format.channelCount);
            double db = rms2 > 1e-16 ? 10 * log10(rms2) : -160;
            if (db > peak) peak = db;
            readFrames += buffer.frameLength;
            qw3_endpoint_feed(&state, (double)readFrames / format.sampleRate, db);
        }
        NSString *reason = @"end_of_file";
        if (state.result == QW3_ENDPOINT_SILENCE) reason = @"silence";
        else if (state.result == QW3_ENDPOINT_NO_SPEECH) reason = @"no_speech";
        else if (state.result == QW3_ENDPOINT_LIMIT) reason = @"max_duration";
        NSData *data = [NSJSONSerialization dataWithJSONObject:@{
            @"automatic": @(state.options.automatic), @"speech_detected": @(state.has_speech),
            @"stop_reason": reason, @"stop_seconds": @(state.elapsed), @"peak_db": @(peak)
        } options:0 error:NULL];
        *report = strdup([[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding].UTF8String);
        return *report ? 0 : fail(err, cap, @"out of memory");
    }
}

int qw3_voice_record(qw3_voice *v, const char *wav,
                     qw3_voice_interrupt_fn interrupt, void *ud, char *err, size_t cap) {
    @autoreleasepool {
        if (!isatty(STDIN_FILENO)) return fail(err, cap, @"microphone capture requires a terminal");
        if (wait_for_audio_memory(interrupt, ud, err, cap) != 0) return -1;
        AVAuthorizationStatus permission = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
        if (permission == AVAuthorizationStatusNotDetermined) {
            __block BOOL answered = NO, allowed = NO;
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
                /* Deliver state changes on this thread's run loop via the main queue. */
                dispatch_async(dispatch_get_main_queue(), ^{ allowed = granted; answered = YES; });
            }];
            while (!answered) {
                if (interrupted(interrupt, ud)) return fail(err, cap, @"recording interrupted");
                tick();
            }
            if (!allowed) return fail(err, cap, @"microphone access denied; enable it in macOS Privacy & Security > Microphone");
        } else if (permission != AVAuthorizationStatusAuthorized) {
            return fail(err, cap, @"microphone access denied; enable it in macOS Privacy & Security > Microphone");
        }
        NSError *error = nil;
        NSDictionary *settings = @{
            AVFormatIDKey: @(kAudioFormatLinearPCM), AVSampleRateKey: @16000,
            AVNumberOfChannelsKey: @1, AVLinearPCMBitDepthKey: @16,
            AVLinearPCMIsFloatKey: @NO, AVLinearPCMIsBigEndianKey: @NO
        };
        AVAudioRecorder *recorder = [[AVAudioRecorder alloc]
            initWithURL:[NSURL fileURLWithPath:@(wav)] settings:settings error:&error];
        recorder.meteringEnabled = YES;
        if (!recorder || ![recorder record])
            return fail(err, cap, error.localizedDescription ?: @"cannot start microphone recording");
        qw3_endpoint state = endpoint(v);
        fprintf(stderr, "voice: recording; %s (maximum %.0f s). Ctrl-C cancels.\n",
                state.options.automatic ? "silence stops automatically; Enter also stops" : "press Enter to stop",
                state.options.max_seconds);
        double started = NSProcessInfo.processInfo.systemUptime;
        BOOL cancelled = NO, manual = NO;
        while (state.result == QW3_ENDPOINT_RECORDING && recorder.recording) {
            if (interrupted(interrupt, ud)) { cancelled = YES; break; }
            struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
            if (poll(&pfd, 1, 0) > 0 && (pfd.revents & (POLLIN | POLLHUP))) {
                char ch;
                ssize_t n = read(STDIN_FILENO, &ch, 1);
                if (n <= 0) { cancelled = YES; break; }
                if (ch == '\n') { manual = YES; break; }
            }
            [recorder updateMeters];
            bool hadSpeech = state.has_speech;
            qw3_endpoint_feed(&state, NSProcessInfo.processInfo.systemUptime - started,
                               [recorder averagePowerForChannel:0]);
            if (!hadSpeech && state.has_speech) fprintf(stderr, "voice: input activity detected...\n");
            tick();
        }
        [recorder stop];
        if (cancelled) return fail(err, cap, @"recording cancelled");
        if (!manual && state.result == QW3_ENDPOINT_RECORDING)
            return fail(err, cap, @"microphone recording stopped unexpectedly");
        if (state.options.automatic && !state.has_speech) {
            unlink(wav);
            if (err && cap) snprintf(err, cap, "no input activity detected; recording discarded");
            return 1;
        }
        if (state.result == QW3_ENDPOINT_SILENCE) fprintf(stderr, "voice: end of utterance detected.\n");
        else if (state.result == QW3_ENDPOINT_LIMIT) fprintf(stderr, "voice: recording duration limit reached.\n");
        return 0;
    }
}

int qw3_voice_transcribe(qw3_voice *v, const char *wav, char **text,
                         qw3_voice_interrupt_fn interrupt, void *ud, char *err, size_t cap) {
    @autoreleasepool {
        *text = NULL;
        NSError *error = nil;
        NSData *audio = [NSData dataWithContentsOfFile:@(wav) options:0 error:&error];
        if (!audio) return fail(err, cap, error.localizedDescription);
        if (audio.length < 12 || memcmp(audio.bytes, "RIFF", 4) || memcmp((const char *)audio.bytes + 8, "WAVE", 4))
            return fail(err, cap, @"transcription input must be a WAV file (recorded input is 16 kHz mono PCM16)");
        NSString *boundary = [@"qw3-" stringByAppendingString:NSUUID.UUID.UUIDString];
        NSMutableData *body = [NSMutableData data];
        NSDictionary *fields = @{@"model": config(v)[@"asr_model"], @"language": config(v)[@"asr_language"]};
        for (NSString *key in @[@"model", @"language"]) {
            NSString *part = [NSString stringWithFormat:@"--%@\r\nContent-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", boundary, key, fields[key]];
            [body appendData:[part dataUsingEncoding:NSUTF8StringEncoding]];
        }
        [body appendData:[[NSString stringWithFormat:@"--%@\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.wav\"\r\nContent-Type: audio/wav\r\n\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        [body appendData:audio];
        [body appendData:[[NSString stringWithFormat:@"\r\n--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
        NSDictionary *reply = json(request(v, @"/v1/audio/transcriptions", body,
            [@"multipart/form-data; boundary=" stringByAppendingString:boundary], interrupt, ud, err, cap), err, cap);
        if (!reply) return -1;
        if (![reply[@"text"] isKindOfClass:NSString.class]) return fail(err, cap, @"transcription response is missing text");
        NSString *result = [reply[@"text"] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        if (!result.length) return fail(err, cap, @"no speech recognized");
        if ([result rangeOfCharacterFromSet:[NSCharacterSet characterSetWithRange:NSMakeRange(0, 1)]].location != NSNotFound)
            return fail(err, cap, @"transcript contains a NUL character");
        *text = strdup(result.UTF8String);
        return *text ? 0 : fail(err, cap, @"out of memory");
    }
}

static int synthesize_spoken(qw3_voice *v, NSString *spoken, const char *wav,
    qw3_voice_interrupt_fn interrupt, void *ud, char *err, size_t cap) {
    @autoreleasepool {
        if (wait_for_audio_memory(interrupt, ud, err, cap) != 0) return -1;
        NSDictionary *c = config(v);
        NSMutableDictionary *options = [@{@"language": c[@"tts_language"]} mutableCopy];
        if (v->speed > 0) options[@"speed"] = @(v->speed);
        NSData *body = [NSJSONSerialization dataWithJSONObject:@{
            @"model": c[@"tts_model"], @"input": spoken, @"voice": c[@"voice"],
            @"response_format": @"wav", @"options": options
        } options:0 error:NULL];
        NSData *audio = request(v, @"/v1/audio/speech", body, @"application/json", interrupt, ud, err, cap);
        if (!audio) return -1;
        if (audio.length < 12 || memcmp(audio.bytes, "RIFF", 4) || memcmp((const char *)audio.bytes + 8, "WAVE", 4))
            return fail(err, cap, @"audio.cpp did not return WAV audio");
        NSError *error = nil;
        if (![audio writeToFile:@(wav) options:NSDataWritingAtomic error:&error])
            return fail(err, cap, error.localizedDescription);
        return 0;
    }
}

int qw3_voice_synthesize(qw3_voice *v, const char *text, const char *wav,
                         qw3_voice_interrupt_fn interrupt, void *ud, char *err, size_t cap) {
    @autoreleasepool {
        if (!text || !*text) return fail(err, cap, @"no speakable text");
        char *corrected = qw3_voice_apply_pronunciation(v, text, err, cap);
        if (!corrected) return -1;
        NSString *spoken = [NSString stringWithUTF8String:corrected];
        free(corrected);
        return synthesize_spoken(v, spoken, wav, interrupt, ud, err, cap);
    }
}

int qw3_voice_play(const char *wav, qw3_voice_interrupt_fn interrupt, void *ud,
                   char *err, size_t cap) {
    @autoreleasepool {
        if (wait_for_audio_memory(interrupt, ud, err, cap) != 0) return -1;
        NSError *error = nil;
        AVAudioPlayer *player = [[AVAudioPlayer alloc] initWithContentsOfURL:[NSURL fileURLWithPath:@(wav)] error:&error];
        if (!player) return fail(err, cap, error.localizedDescription ?: @"cannot decode synthesized audio");
        if (![player prepareToPlay] || ![player play]) return fail(err, cap, @"cannot play synthesized audio");
        while (player.playing) {
            if (interrupted(interrupt, ud)) {
                [player stop];
                return fail(err, cap, @"playback interrupted");
            }
            tick();
        }
        return 0;
    }
}

static NSString *replace(NSString *text, NSString *pattern, NSString *replacement) {
    NSRegularExpression *rx = [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:NULL];
    return [rx stringByReplacingMatchesInString:text options:0 range:NSMakeRange(0, text.length) withTemplate:replacement];
}

static NSArray<NSString *> *speech_chunks(qw3_voice *v, const char *text, char *err, size_t cap) {
    char *corrected = qw3_voice_apply_pronunciation(v, text, err, cap);
    if (!corrected) return nil;
    NSString *prose = [NSString stringWithUTF8String:corrected];
    free(corrected);
    prose = [replace(prose, @"\\s+", @" ") stringByTrimmingCharactersInSet:
        NSCharacterSet.whitespaceAndNewlineCharacterSet];
    NSMutableIndexSet *sentences = [NSMutableIndexSet indexSet];
    [prose enumerateSubstringsInRange:NSMakeRange(0, prose.length)
        options:NSStringEnumerationBySentences usingBlock:
        ^(NSString *part, NSRange range, NSRange enclosing, BOOL *stop) {
            (void)part; (void)enclosing; (void)stop;
            NSUInteger end = NSMaxRange(range);
            while (end && [prose characterAtIndex:end - 1] == ' ') end--;
            if (end == prose.length || [prose characterAtIndex:end] == ' ') [sentences addIndex:end];
        }];
    NSUInteger limit = [config(v)[@"speech_chunk_chars"] unsignedIntegerValue];
    NSMutableArray *chunks = [NSMutableArray array];
    for (NSUInteger start = 0; start < prose.length;) {
        NSUInteger target = MIN(prose.length, start + limit);
        NSUInteger end = [sentences indexLessThanOrEqualToIndex:target];
        if (end == NSNotFound || end <= start) {
            end = target;
            if (end < prose.length) {
                while (end > start && [prose characterAtIndex:end] != ' ') end--;
                if (end == start) {
                    end = target;
                    while (end < prose.length && [prose characterAtIndex:end] != ' ') end++;
                }
            }
        }
        [chunks addObject:[prose substringWithRange:NSMakeRange(start, end - start)]];
        start = end;
        while (start < prose.length && [prose characterAtIndex:start] == ' ') start++;
    }
    return chunks;
}

char *qw3_voice_plan(qw3_voice *v, const char *text, char *err, size_t cap) {
    @autoreleasepool {
        NSArray *chunks = speech_chunks(v, text, err, cap);
        if (!chunks) return NULL;
        NSData *data = [NSJSONSerialization dataWithJSONObject:chunks options:0 error:NULL];
        return data ? strndup(data.bytes, data.length) : NULL;
    }
}

typedef struct {
    atomic_bool cancelled;
    qw3_voice_interrupt_fn interrupt;
    void *ud;
} speech_cancel;

static bool speech_worker_interrupted(void *ud) {
    return atomic_load(&((speech_cancel *)ud)->cancelled);
}

/* Only the main thread invokes the caller's interrupt callback. Workers share
 * an atomic flag, so a terminal SIGINT callback need not be thread-safe. */
static bool speech_main_interrupted(void *ud) {
    speech_cancel *state = ud;
    if (interrupted(state->interrupt, state->ud)) atomic_store(&state->cancelled, true);
    return speech_worker_interrupted(ud);
}

int qw3_voice_speak(qw3_voice *v, const char *text,
    qw3_voice_player_fn player, qw3_voice_interrupt_fn interrupt, void *ud,
    qw3_voice_speech_stats *stats, char *err, size_t cap) {
    @autoreleasepool {
        double started = NSProcessInfo.processInfo.systemUptime;
        qw3_voice_speech_stats result = { .first_chunk_ready_ms = -1 };
        result.wired_before_bytes = wired_memory_bytes();
        result.wired_sample_max_bytes = result.wired_before_bytes;
        if (stats) *stats = result;
        NSArray<NSString *> *chunks = speech_chunks(v, text, err, cap);
        if (!chunks) return -1;
        result.chunks_total = chunks.count;
        if (!chunks.count) return fail(err, cap, @"no speakable text");
        char directory[] = "/tmp/qw3-speech-XXXXXX";
        if (!mkdtemp(directory)) return fail(err, cap, @"cannot create temporary audio directory");
        NSArray<NSString *> *paths = @[
            [@(directory) stringByAppendingPathComponent:@"0.wav"],
            [@(directory) stringByAppendingPathComponent:@"1.wav"]];
        speech_cancel state = { .interrupt = interrupt, .ud = ud };
        atomic_init(&state.cancelled, false);
        speech_cancel *cancel = &state;
        dispatch_queue_t queue = dispatch_queue_create("qw3.voice.prefetch", DISPATCH_QUEUE_SERIAL);
        if (!player) player = qw3_voice_play;
        double before = NSProcessInfo.processInfo.systemUptime;
        int rc = synthesize_spoken(v, chunks[0], paths[0].fileSystemRepresentation,
            speech_main_interrupted, cancel, err, cap);
        result.synthesis_ms += (NSProcessInfo.processInfo.systemUptime - before) * 1000;
        if (!rc) result.first_chunk_ready_ms = (NSProcessInfo.processInfo.systemUptime - started) * 1000;
        result.wired_sample_max_bytes = MAX(result.wired_sample_max_bytes, wired_memory_bytes());
        for (NSUInteger i = 0; !rc && i < chunks.count; i++) {
            if (speech_main_interrupted(cancel)) { rc = fail(err, cap, @"speech interrupted"); break; }
            BOOL hasNext = i + 1 < chunks.count;
            dispatch_semaphore_t ready = dispatch_semaphore_create(0);
            __block int nextRc = 0;
            __block NSString *nextError = nil;
            __block double nextMs = 0;
            if (hasNext) dispatch_async(queue, ^{
                @autoreleasepool {
                    char message[512] = {0};
                    double t = NSProcessInfo.processInfo.systemUptime;
                    nextRc = synthesize_spoken(v, chunks[i + 1], paths[(i + 1) % 2].fileSystemRepresentation,
                        speech_worker_interrupted, cancel, message, sizeof(message));
                    nextMs = (NSProcessInfo.processInfo.systemUptime - t) * 1000;
                    if (nextRc) nextError = @(message);
                    dispatch_semaphore_signal(ready);
                }
            });
            before = NSProcessInfo.processInfo.systemUptime;
            rc = player(paths[i % 2].fileSystemRepresentation, speech_main_interrupted, cancel, err, cap);
            result.playback_ms += (NSProcessInfo.processInfo.systemUptime - before) * 1000;
            if (!rc) result.chunks_played++;
            unlink(paths[i % 2].fileSystemRepresentation);
            if (rc || speech_main_interrupted(cancel)) {
                atomic_store(&state.cancelled, true);
                if (!rc) rc = fail(err, cap, @"speech interrupted");
            }
            if (hasNext) {
                /* Join even on cancel/failure: no worker can outlive the client,
                 * its stack cancellation state or the private WAV directory. */
                while (dispatch_semaphore_wait(ready, dispatch_time(DISPATCH_TIME_NOW, 25000000)) != 0)
                    speech_main_interrupted(cancel);
                result.synthesis_ms += nextMs;
                if (!rc && speech_main_interrupted(cancel)) rc = fail(err, cap, @"speech interrupted");
                if (!rc && nextRc) rc = fail(err, cap, nextError ?: @"speech synthesis failed");
            }
            result.wired_sample_max_bytes = MAX(result.wired_sample_max_bytes, wired_memory_bytes());
        }
        for (NSString *path in paths) unlink(path.fileSystemRepresentation);
        rmdir(directory);
        result.total_ms = (NSProcessInfo.processInfo.systemUptime - started) * 1000;
        if (stats) *stats = result;
        return rc;
    }
}

/* Inline spans often contain prices, measurements, model names or identifiers.
 * Remove the delimiters, not the data. A matching run must have the same size. */
static NSString *spoken_inline(NSString *text) {
    NSMutableString *out = [NSMutableString string];
    for (NSUInteger i = 0; i < text.length;) {
        if ([text characterAtIndex:i] != '`') {
            [out appendString:[text substringWithRange:NSMakeRange(i++, 1)]];
            continue;
        }
        NSUInteger start = i;
        while (i < text.length && [text characterAtIndex:i] == '`') i++;
        NSUInteger count = i - start, end = i;
        BOOL matched = NO;
        while (end < text.length && [text characterAtIndex:end] != '\n') {
            if ([text characterAtIndex:end] != '`') { end++; continue; }
            NSUInteger closing = end;
            while (end < text.length && [text characterAtIndex:end] == '`') end++;
            if (end - closing == count) {
                [out appendString:[text substringWithRange:NSMakeRange(i, closing - i)]];
                i = end;
                matched = YES;
                break;
            }
        }
        if (!matched) [out appendString:[text substringWithRange:NSMakeRange(start, count)]];
    }
    return out;
}

static NSString *reference_key(NSString *label) {
    return [replace(label, @"\\s+", @" ") stringByTrimmingCharactersInSet:
        NSCharacterSet.whitespaceAndNewlineCharacterSet].lowercaseString;
}

/* Keep source names, omit Markdown syntax and URL paths/tracking parameters.
 * NSDataDetector parses links locally; this performs no network requests. */
static NSString *spoken_links(NSString *text) {
    NSMutableString *result = [text mutableCopy];
    NSMutableDictionary *references = [NSMutableDictionary dictionary];
    NSRegularExpression *definitions = [NSRegularExpression regularExpressionWithPattern:
        @"(?m)^ {0,3}\\[([^\\]\\n]+)\\]:[ \\t]*<?(https?://[^\\s<>]+)>?[^\\n]*$"
        options:NSRegularExpressionCaseInsensitive error:NULL];
    for (NSTextCheckingResult *hit in [[definitions matchesInString:result options:0
        range:NSMakeRange(0, result.length)] reverseObjectEnumerator]) {
        NSString *key = reference_key([result substringWithRange:[hit rangeAtIndex:1]]);
        references[key] = [result substringWithRange:[hit rangeAtIndex:2]];
        [result replaceCharactersInRange:hit.range withString:@""];
    }
    NSRegularExpression *inlineLinks = [NSRegularExpression regularExpressionWithPattern:
        @"(!?)\\[([^\\]\\n]*)\\]\\((?:[^()\\n]|\\([^()\\n]*\\))*\\)" options:0 error:NULL];
    for (NSTextCheckingResult *hit in [[inlineLinks matchesInString:result options:0
        range:NSMakeRange(0, result.length)] reverseObjectEnumerator]) {
        BOOL image = [hit rangeAtIndex:1].length > 0;
        NSString *label = [result substringWithRange:[hit rangeAtIndex:2]];
        [result replaceCharactersInRange:hit.range withString:image ? @"" : label];
    }
    NSRegularExpression *referenceLinks = [NSRegularExpression regularExpressionWithPattern:
        @"(!?)\\[([^\\]\\n]+)\\](?:\\[([^\\]\\n]*)\\])?" options:0 error:NULL];
    for (NSTextCheckingResult *hit in [[referenceLinks matchesInString:result options:0
        range:NSMakeRange(0, result.length)] reverseObjectEnumerator]) {
        NSString *label = [result substringWithRange:[hit rangeAtIndex:2]];
        NSRange keyRange = [hit rangeAtIndex:3];
        NSString *key = keyRange.location != NSNotFound && keyRange.length ?
            [result substringWithRange:keyRange] : label;
        if (!references[reference_key(key)]) continue;
        BOOL image = [hit rangeAtIndex:1].length > 0;
        [result replaceCharactersInRange:hit.range withString:image ? @"" : label];
    }
    result = [replace(result, @"<(https?://[^<>\\s]+)>", @"$1") mutableCopy];
    NSDataDetector *urls = [NSDataDetector dataDetectorWithTypes:NSTextCheckingTypeLink error:NULL];
    for (NSTextCheckingResult *hit in [[urls matchesInString:result options:0
        range:NSMakeRange(0, result.length)] reverseObjectEnumerator]) {
        NSURL *url = hit.URL;
        if (![@[@"http", @"https"] containsObject:url.scheme.lowercaseString] || !url.host.length) continue;
        [result replaceCharactersInRange:hit.range withString:url.host];
    }
    return result;
}

static NSArray<NSString *> *table_cells(NSString *line) {
    NSString *trimmed = [line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
    NSMutableArray *cells = [NSMutableArray array];
    NSMutableString *cell = [NSMutableString string];
    for (NSUInteger i = 0; i < trimmed.length; i++) {
        unichar ch = [trimmed characterAtIndex:i];
        if (ch == '\\' && i + 1 < trimmed.length &&
            ([trimmed characterAtIndex:i + 1] == '|' || [trimmed characterAtIndex:i + 1] == '\\')) {
            [cell appendFormat:@"%C", [trimmed characterAtIndex:++i]];
        } else if (ch == '|') {
            [cells addObject:[cell stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet]];
            [cell setString:@""];
        } else [cell appendFormat:@"%C", ch];
    }
    [cells addObject:[cell stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet]];
    if ([trimmed hasPrefix:@"|"] && cells.count && ![cells[0] length]) [cells removeObjectAtIndex:0];
    if (cells.count && [trimmed hasSuffix:@"|"] && ![[cells lastObject] length]) [cells removeLastObject];
    return cells;
}

static BOOL table_separator(NSArray<NSString *> *cells) {
    if (!cells.count) return NO;
    for (NSString *cell in cells) {
        NSString *dashes = [cell stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@":"]];
        if (dashes.length < 3 || [dashes stringByTrimmingCharactersInSet:
            [NSCharacterSet characterSetWithCharactersInString:@"-"]].length) return NO;
    }
    return YES;
}

static NSString *sentence(NSString *text) {
    NSString *trimmed = [text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    if (!trimmed.length) return @"";
    unichar last = [trimmed characterAtIndex:trimmed.length - 1];
    if (last == '.' || last == '!' || last == '?' || last == ':' || last == ';') return trimmed;
    return [trimmed stringByAppendingString:@"."];
}

/* Generic fallback for explicitly requested visual answers. The model's voice
 * profile should produce natural prose first; this preserves table data/labels
 * without assuming a domain or changing numeric values. */
static NSString *spoken_blocks(NSString *text) {
    NSArray<NSString *> *lines = [text componentsSeparatedByString:@"\n"];
    NSMutableString *out = [NSMutableString string];
    NSRegularExpression *list = [NSRegularExpression regularExpressionWithPattern:
        @"^\\s{0,3}(?:#{1,6}\\s+|[-*+>]\\s+|[0-9]+[.)]\\s+)" options:0 error:NULL];
    for (NSUInteger i = 0; i < lines.count; i++) {
        NSString *line = lines[i];
        if ([line containsString:@"|"] && i + 1 < lines.count) {
            NSArray *headers = table_cells(line);
            NSArray *separator = table_cells(lines[i + 1]);
            if (headers.count && separator.count == headers.count && table_separator(separator)) {
                i += 2;
                for (; i < lines.count && [lines[i] containsString:@"|"]; i++) {
                    NSArray *values = table_cells(lines[i]);
                    if (table_separator(values)) continue;
                    NSMutableArray *fields = [NSMutableArray array];
                    for (NSUInteger j = 0; j < values.count; j++) {
                        NSString *value = values[j];
                        if (!value.length) continue;
                        NSString *header = j < headers.count ? headers[j] : @"";
                        NSString *field = header.length ?
                            [NSString stringWithFormat:@"%@: %@", header, value] : value;
                        [fields addObject:sentence(field)];
                    }
                    if (fields.count) [out appendFormat:@"%@\n", [fields componentsJoinedByString:@" "]];
                }
                i--;
                continue;
            }
        }
        NSString *trimmed = [line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        if (trimmed.length >= 3 &&
            (![trimmed stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"-"]].length ||
             ![trimmed stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"*"]].length))
            continue;
        if ([list firstMatchInString:line options:0 range:NSMakeRange(0, line.length)]) {
            line = replace(line, @"[\\x{1F000}-\\x{1FAFF}\\x{2600}-\\x{27BF}\\uFE0F]", @"");
            line = sentence(line);
        }
        [out appendFormat:@"%@\n", line];
    }
    return out;
}

char *qw3_voice_prepare_text(const char *raw, bool thinking_enabled) {
    @autoreleasepool {
        NSString *text = raw ? [NSString stringWithUTF8String:raw] : @"";
        if (!text) return strdup("");
        if (thinking_enabled) {
            NSRange end = [text rangeOfString:@"</think>"];
            if (end.location == NSNotFound) return strdup("");
            text = [text substringFromIndex:NSMaxRange(end)];
        }
        text = replace(text, @"(?s)<think>.*?(?:</think>|$)", @"");
        /* Defensive truncation: this function also serves the standalone tool. */
        for (NSString *marker in @[@"<tool_call>", @"<｜DSML｜tool_calls>", @"<tool_response>", @"<|im_end|>"]) {
            NSRange hit = [text rangeOfString:marker];
            if (hit.location != NSNotFound) text = [text substringToIndex:hit.location];
        }
        NSMutableString *prose = [NSMutableString string];
        unichar fence = 0;
        NSUInteger fenceLength = 0;
        for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
            NSString *trimmed = [line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
            NSUInteger run = 0;
            unichar first = trimmed.length ? [trimmed characterAtIndex:0] : 0;
            if (first == '`' || first == '~') {
                while (run < trimmed.length && [trimmed characterAtIndex:run] == first) run++;
            }
            if (!fence && run >= 3) { fence = first; fenceLength = run; continue; }
            if (fence) {
                if (first == fence && run >= fenceLength &&
                    ![[trimmed substringFromIndex:run] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet].length)
                    fence = 0;
                continue;
            }
            if ([line hasPrefix:@"    "] || [line hasPrefix:@"\t"]) continue;
            [prose appendFormat:@"%@\n", line];
        }
        text = spoken_links(prose);
        text = spoken_blocks(text);
        text = spoken_inline(text);
        text = replace(text, @"(?m)^\\s{0,3}(?:#{1,6}\\s+|[-*+>]\\s+|[0-9]+[.)]\\s+)", @"");
        text = replace(text, @"\\x1b\\[[0-?]*[ -/]*[@-~]", @"");
        /* Remove paired Markdown emphasis, preserving literal operators and
         * identifiers such as 2*3 and item_42 instead of joining their values. */
        for (int nesting = 0; nesting < 3; nesting++)
            text = replace(text, @"(?<!\\w)(\\*\\*\\*|___|\\*\\*|__|\\*|_)(?=\\S)(.+?)(?<=\\S)\\1(?!\\w)", @"$2");
        text = replace(text, @"\\s+", @" ");
        return strdup([text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].UTF8String);
    }
}
