#!/bin/sh
printf 'fake benchmark stdout preserved\n'
printf 'fake benchmark stderr preserved\n' >&2
exit 127
