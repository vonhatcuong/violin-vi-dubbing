#!/bin/bash
# Example Mac-side puller used with the GPU server (see docs/ops/gpu-server-runbook.md §6). Edit BASE/H/port before reuse.
# Pull finished dubbing outputs from the GPU server every 10 min for up to 14 h. 720p only for HD sources; MIT (<=360p) as is.
BASE="/Users/riley/Nhat Cuong/code/00-personal/violin/output/e2e_server"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 41620"
H=root@83.10.114.74
END=$(( $(date +%s) + 14*3600 ))
while [ $(date +%s) -lt $END ]; do
  for pair in "out13:envs_run13:envs_vi:envs" "out14:video3_run14:video3_vi:video3"; do
    IFS=: read -r remote local stem log <<< "$pair"
    [ -f "$BASE/$local/.synced" ] && continue
    if $SSH $H "grep -q 'Done!' /workspace/$remote/$log.log 2>/dev/null"; then
      # make the 720p on the server if missing (NVENC), then pull only that
      # skip while an encoder is still writing the 720p; encode only if the file is absent; verify it is a complete mp4 before pulling
      $SSH $H "pgrep -f \"ffmpeg.*${stem}_720[p]\" >/dev/null" && continue
      $SSH $H "cd /workspace/$remote && ([ -f ${stem}_720p.mp4 ] || ffmpeg -hide_banner -loglevel error -y -hwaccel cuda -i ${stem}.mp4 -vf scale=-2:720 -c:v h264_nvenc -preset p4 -cq 30 -c:a aac -b:a 96k ${stem}_720p.mp4) && ffprobe -v error -show_entries format=duration -of csv=p=0 ${stem}_720p.mp4 | grep -qE '^[0-9]'" || continue
      mkdir -p "$BASE/$local"
      rsync -a --partial -e "$SSH" $H:/workspace/$remote/${stem}_720p.mp4 :/workspace/$remote/${stem}.srt :/workspace/$remote/${stem}.voices.json :/workspace/$remote/${stem}.fit.units.json :/workspace/$remote/${stem}.transcript.txt :/workspace/$remote/$log.log "$BASE/$local/" && [ -s "$BASE/$local/${stem}_720p.mp4" ] && touch "$BASE/$local/.synced" && echo "$(date +%FT%T) synced $local"
    fi
  done
  for tag in $($SSH $H "cd /workspace/out_18_06 2>/dev/null && ls -d */DONE 2>/dev/null | cut -d/ -f1"); do
    [ -f "$BASE/mit_18_06/$tag/.synced" ] && continue
    id=${tag#*_}
    # compact copy on the server first (NVENC, native resolution capped at 720p, ~300-500 kbps) — the merger's own encode is ~1.6 Mbps
    $SSH $H "pgrep -f \"ffmpeg.*${id}_vi_720[p]\" >/dev/null" && continue
    $SSH $H "cd /workspace/out_18_06/$tag && ([ -f ${id}_vi_720p.mp4 ] || ffmpeg -hide_banner -loglevel error -y -hwaccel cuda -i ${id}_vi.mp4 -vf \"scale=-2:'min(720,ih)'\" -c:v h264_nvenc -preset p4 -cq 32 -c:a aac -b:a 96k ${id}_vi_720p.mp4) && ffprobe -v error -show_entries format=duration -of csv=p=0 ${id}_vi_720p.mp4 | grep -qE '^[0-9]'" || continue
    mkdir -p "$BASE/mit_18_06/$tag"
    rsync -a --partial -e "$SSH" $H:/workspace/out_18_06/$tag/${id}_vi_720p.mp4 :/workspace/out_18_06/$tag/${id}_vi.srt :/workspace/out_18_06/$tag/${id}_vi.fit.units.json :/workspace/out_18_06/$tag/${id}_vi.transcript.txt :/workspace/out_18_06/$tag/run.log "$BASE/mit_18_06/$tag/" && [ -s "$BASE/mit_18_06/$tag/${id}_vi_720p.mp4" ] && touch "$BASE/mit_18_06/$tag/.synced" && echo "$(date +%FT%T) synced $tag"
  done
  sleep 600
done
