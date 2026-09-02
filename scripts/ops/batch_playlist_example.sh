#!/bin/bash
# Example used for MIT 18.06 (see docs/ops/gpu-server-runbook.md §5-6). Edit IDS/paths before reuse.
# MIT 18.06 (PLE7DDD91010BC51F8): sequential dubbing queue; idempotent (skips items with DONE marker); runs after run 14.
IDS="J7DzL2_Na80 QVKj3LADCnA FX4C-JpTFgY MsIvs_6vC38 JibVXBElKL0 8o5Cmfpeo6g VqP2tREMvt0 9Q1q7s1jTzU yjBerM5jWsc nHlE7EgJFds 2IdtqGM6KWU 6-wh6yvk6uc l88D4r74gtM YzZUIYRCE38 Y_Ac6KiQ1t0 osh80YCg_GM 0MtwqhIwdrI srxexLishgY 23LLB9mNJvc QNpj-gOXW9M cdZnhQjJu4I 13r9QY6cmjc IZqwi0wJovM lGGDIGizcQ0 QuZL5IKpO_U UCc9q_cAhho M0Sa8fLOajA vF7eyJ2g3kU TSdXJw83kyA TX_vooSnhm8 Ts3o2I8_Mxc 0h43aV4aH7I HgC1l_6ySkc Go2aLo7ZOlU RWvi4Vx4CDc 7UJ4CFRGd-U"
YT=/workspace/violin/.venv/bin/yt-dlp
OUT=/workspace/out_18_06; mkdir -p $OUT /workspace/samples/18_06
while [ ! -f /workspace/out13/DONE ]; do sleep 30; done
while pgrep -f "^\.venv/bin/python main\.py" >/dev/null; do sleep 20; done
n=0
for id in $IDS; do
  n=$((n+1)); tag=$(printf "%02d_%s" $n $id); d=$OUT/$tag
  [ -f $d/DONE ] && continue
  mkdir -p $d
  src=/workspace/samples/18_06/$id.mp4
  if [ ! -f $src ]; then
    $YT -f "bv*[ext=mp4][vcodec^=avc1][height<=1080]+ba[ext=m4a]/bv*[height<=1080]+ba/b" --merge-output-format mp4 --extractor-args "youtube:player_client=web_safari,android,default" -o "$src" --no-warnings "https://www.youtube.com/watch?v=$id" > $d/download.log 2>&1
    [ -f $src ] || { echo "$(date +%FT%T) $tag DOWNLOAD FAILED" >> $OUT/batch.log; continue; }
  fi
  echo "$(date +%FT%T) $tag START" >> $OUT/batch.log; date +%s > $d/start_epoch
  ( cd /workspace/violin && CUDA_VISIBLE_DEVICES=GPU-8ff58980-c6f9-dbd4-096a-2b14d3c6414a .venv/bin/python main.py "$src" "$d/${id}_vi.mp4" --language Vietnamese --config /workspace/config_run10.yaml > $d/run.log 2>&1 )
  if [ -f "$d/${id}_vi.mp4" ]; then touch $d/DONE; echo "$(date +%FT%T) $tag DONE $(( $(date +%s) - $(cat $d/start_epoch) )) s" >> $OUT/batch.log; else echo "$(date +%FT%T) $tag FAILED" >> $OUT/batch.log; fi
done
echo "$(date +%FT%T) BATCH COMPLETE" >> $OUT/batch.log
