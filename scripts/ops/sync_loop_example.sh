#!/bin/bash
# Example Mac-side puller (see docs/ops/gpu-server-runbook.md §6). Pairs with scripts/ops/encode720_daemon_example.sh on the server. Edit BASE/H/port before reuse.
BASE="/Users/riley/Nhat Cuong/code/00-personal/violin/output/e2e_server"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ConnectionAttempts=2 -p 41620"
H=root@83.10.114.74
END=$(( $(date +%s) + 20*3600 ))
while [ $(date +%s) -lt $END ]; do
  ready=$($SSH $H 'ls /workspace/out_18_06/*/*_720p.mp4.ok /workspace/out_cs336/*/*_720p.mp4.ok /workspace/out12/*_720p.mp4.ok /workspace/out13/*_720p.mp4.ok /workspace/out14/*_720p.mp4.ok 2>/dev/null' 2>/dev/null)
  if [ -z "$ready" ]; then sleep 120; continue; fi
  for okf in $ready; do
    f720=${okf%.ok}; rdir=$(dirname "$f720"); fname=$(basename "$f720"); stem=${fname%_720p.mp4}
    case "$rdir" in
      /workspace/out_18_06/*) tag=$(basename "$rdir"); ldir="$BASE/mit_18_06/$tag"; extra="$rdir/${stem}.srt $rdir/${stem}.fit.units.json $rdir/${stem}.transcript.txt $rdir/run.log" ;;
      /workspace/out_cs336/*)  tag=$(basename "$rdir"); ldir="$BASE/cs336/$tag";     extra="$rdir/${stem}.srt $rdir/${stem}.fit.units.json $rdir/${stem}.transcript.txt $rdir/run.log" ;;
      /workspace/out12) ldir="$BASE/docs_llm_run12"; extra="$rdir/${stem}.srt $rdir/${stem}.voices.json $rdir/${stem}.fit.units.json $rdir/${stem}.transcript.txt" ;;
      /workspace/out13) ldir="$BASE/envs_run13";     extra="$rdir/${stem}.srt $rdir/${stem}.voices.json $rdir/${stem}.fit.units.json $rdir/${stem}.transcript.txt" ;;
      /workspace/out14) ldir="$BASE/video3_run14";   extra="$rdir/${stem}.srt $rdir/${stem}.voices.json $rdir/${stem}.fit.units.json $rdir/${stem}.transcript.txt" ;;
      *) continue ;;
    esac
    [ -f "$ldir/.synced" ] && continue
    mkdir -p "$ldir"
    srcs="$H:$f720"; for e in $extra; do srcs="$srcs :$e"; done
    if rsync -a --partial -e "$SSH" $srcs "$ldir/" 2>>"$BASE/.sync_errors.log" && [ -s "$ldir/$fname" ]; then
      touch "$ldir/.synced"; echo "$(date +%FT%T) synced $(basename "$ldir")"
    else
      echo "$(date +%FT%T) retry-later $(basename "$ldir")"; sleep 30
    fi
  done
  sleep 90
done
