#!/bin/bash
# Example (2026-09-03): download sources on a personal machine when YouTube blocks the server IP, then push them. Edit IDS/paths/port.
# Download the CS336 lectures YouTube refuses to serve to the server, then push each to /workspace/samples/cs336/.
D="/private/tmp/claude-501/-Users-riley-Nhat-Cuong-code-00-personal/de56e961-708c-4a0e-aa19-971aa07b72fe/scratchpad/cs336_src"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 41620"; H=root@83.10.114.74
IDS="EfM546A79aM vTfEyOyzV9E JpAxdTWQJxM -qm0ln33G24 5sxHosTLPF8 2oH6PWPrYFo dIFAi87Ws4E 26FtD08ZpOU 9EEm4iMAF5s"
mkdir -p "$D"
for id in $IDS; do
  f="$D/$id.mp4"
  if [ ! -s "$f" ]; then
    uvx --from "yt-dlp>=2026.8.19" yt-dlp -f "bv*[ext=mp4][vcodec^=avc1][height<=1080]+ba[ext=m4a]/bv*[height<=1080]+ba/b" --merge-output-format mp4 --extractor-args "youtube:player_client=web_safari,android,default" -o "$f" --no-warnings -- "https://www.youtube.com/watch?v=$id" >> "$D/fetch.log" 2>&1
    [ -s "$f" ] || { echo "$(date +%FT%T) $id DOWNLOAD FAILED"; continue; }
  fi
  if $SSH $H "[ -s /workspace/samples/cs336/$id.mp4 ]"; then echo "$(date +%FT%T) $id already on server"; continue; fi
  if rsync -a --partial -e "$SSH" "$f" "$H:/workspace/samples/cs336/$id.mp4.part" && $SSH $H "mv /workspace/samples/cs336/$id.mp4.part /workspace/samples/cs336/$id.mp4"; then
    echo "$(date +%FT%T) $id pushed ($(stat -f %z "$f") bytes)"; rm -f "$f"
  else
    echo "$(date +%FT%T) $id PUSH FAILED"
  fi
done
echo "$(date +%FT%T) fetch+push finished"
