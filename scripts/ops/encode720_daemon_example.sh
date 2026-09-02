#!/bin/bash
# Make the compact 720p copy as soon as an output exists (NVENC), so the Mac puller never waits on encoding.
enc() { # $1=src $2=dst $3=cq
  [ -f "$2.ok" ] && return 0
  [ -f "$1" ] || return 0
  ffmpeg -hide_banner -loglevel error -y -hwaccel cuda -i "$1" -vf "scale=-2:'min(720,ih)'" -c:v h264_nvenc -preset p4 -cq "$3" -c:a aac -b:a 96k "$2.tmp.mp4" \
    && ffprobe -v error -show_entries format=duration -of csv=p=0 "$2.tmp.mp4" | grep -qE '^[0-9]' && mv -f "$2.tmp.mp4" "$2" && touch "$2.ok" && echo "$(date +%FT%T) encoded $2"
}
while true; do
  for root in out_18_06 out_cs336; do
    for d in /workspace/$root/*/; do
      [ -f "$d/DONE" ] || continue; t=$(basename "$d"); id=${t#*_}
      enc "$d/${id}_vi.mp4" "$d/${id}_vi_720p.mp4" $([ "$root" = out_18_06 ] && echo 32 || echo 30)
    done
  done
  for spec in "out12:docs_llm_vi:docs_llm" "out13:envs_vi:envs" "out14:video3_vi:video3"; do
    IFS=: read -r dir stem log <<< "$spec"
    grep -q "Done!" /workspace/$dir/$log.log 2>/dev/null && enc "/workspace/$dir/$stem.mp4" "/workspace/$dir/${stem}_720p.mp4" 30
  done
  sleep 30
done
