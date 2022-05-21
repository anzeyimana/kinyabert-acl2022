# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

def read_lines(file_name):
    f = open(file_name, 'r')
    lines = [line.rstrip('\n') for line in f]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    f.close()
    return lines

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
