#!/bin/bash
# Example (2026-09-03): batch for a playlist whose sources are pushed from a personal machine (YouTube blocks the server IP). Processes any lecture whose source exists, loops until all DONE. Edit IDS/paths.
# Stanford CME295: sources are pushed from the Mac. Process whichever lecture has a source, in order; keep looping until all DONE.
IDS="Ub3GoFaUcds yT84Y5zCnaA Q5baLehv5So VlA_jt_3Qc4 PmW_TMQ3l0I k5Fh-UgTuCo h-7S6HNq0Vg 8fNP4N46RRo Q86qzJ1K1Ss"
OUT=/workspace/out_cme295; mkdir -p $OUT /workspace/samples/cme295
while true; do
  n=0; remaining=0; ran=0
  for id in $IDS; do
    n=$((n+1)); tag=$(printf "%02d_%s" $n "$id"); d=$OUT/$tag; src=/workspace/samples/cme295/$id.mp4
    [ -f "$d/DONE" ] && continue
    remaining=$((remaining+1))
    [ -f "$src" ] || continue                                   # not pushed yet: try the next lecture
    mkdir -p "$d"
    while pgrep -f "^\.venv/bin/python main\.py" >/dev/null; do sleep 20; done
    echo "$(date +%FT%T) $tag START" >> $OUT/batch.log; date +%s > "$d/start_epoch"
    ( cd /workspace/violin && CUDA_VISIBLE_DEVICES=GPU-8ff58980-c6f9-dbd4-096a-2b14d3c6414a .venv/bin/python main.py "$src" "$d/${id}_vi.mp4" --language Vietnamese --config /workspace/config_run10.yaml > "$d/run.log" 2>&1 )
    if [ -f "$d/${id}_vi.mp4" ]; then touch "$d/DONE"; echo "$(date +%FT%T) $tag DONE $(( $(date +%s) - $(cat "$d/start_epoch") )) s" >> $OUT/batch.log; else echo "$(date +%FT%T) $tag FAILED" >> $OUT/batch.log; fi
    ran=1
  done
  [ $remaining = 0 ] && { echo "$(date +%FT%T) BATCH COMPLETE" >> $OUT/batch.log; exit 0; }
  [ $ran = 0 ] && sleep 60
done
