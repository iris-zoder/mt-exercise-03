import sys
import re

for line in sys.stdin:
    # Get rid of speaker assignments and scene information
    line = re.sub(r'[-\[\]]', '', line)
    if line.strip() == "" or re.search(r'[A-Z]{2,}', line):
        continue
    else:
        line = line.replace(u'\ufeff', '')
        line = " ".join(line.split())
        sys.stdout.write(line + "\n")
