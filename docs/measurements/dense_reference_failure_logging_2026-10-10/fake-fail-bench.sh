#!/bin/sh
printf "runner_failure_stdout\n"
printf "runner_failure_stderr\n" >&2
exit 127
