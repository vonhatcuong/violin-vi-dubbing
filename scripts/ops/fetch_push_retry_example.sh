#!/bin/bash
# After the first fetch+push pass, re-run it up to 3 times so transient 403s (e.g. lecture 1) get retried.
S="/private/tmp/claude-501/-Users-riley-Nhat-Cuong-code-00-personal/de56e961-708c-4a0e-aa19-971aa07b72fe/scratchpad"
for pass in 1 2 3; do
  while pgrep -f "cme295_fetch_push.s[h]" >/dev/null; do sleep 60; done
  sleep 120
  missing=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -p 41620 root@83.10.114.74 'for id in Ub3GoFaUcds yT84Y5zCnaA Q5baLehv5So VlA_jt_3Qc4 PmW_TMQ3l0I k5Fh-UgTuCo h-7S6HNq0Vg 8fNP4N46RRo Q86qzJ1K1Ss; do [ -s /workspace/samples/cme295/$id.mp4 ] || echo $id; done' 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%FT%T) retry pass $pass: $missing missing on server"
  [ "$missing" = "0" ] && exit 0
  "$S/cme295_fetch_push.sh" >> "$S/cme295_fetch_push.log" 2>&1
done
