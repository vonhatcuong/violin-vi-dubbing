#!/bin/bash
# Example (2026-09-03): push finished 720p outputs + srt to Google Drive with `gws` (folder ids, readable names, modifiedTime by lecture). Edit BASE/S/titles.
# Upload synced 720p outputs (+ .srt) to Google Drive via gws. Waits for auth; idempotent (.uploaded markers); readable names.
BASE="/Users/riley/Nhat Cuong/code/00-personal/violin/output/e2e_server"
S="/private/tmp/claude-501/-Users-riley-Nhat-Cuong-code-00-personal/de56e961-708c-4a0e-aa19-971aa07b72fe/scratchpad"
TITLES="$S/titles.tsv"
END=$(( $(date +%s) + 24*3600 ))
jid() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null; }
folder() { # $1=name $2=parent-id-or-empty → prints id (find or create)
  local q="name = '$1' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
  [ -n "$2" ] && q="$q and '$2' in parents"
  local id=$(gws drive files list --params "{\"q\":\"$q\",\"fields\":\"files(id)\",\"pageSize\":1}" 2>/dev/null | python3 -c "import json,sys; f=json.load(sys.stdin).get('files',[]); print(f[0]['id'] if f else '')" 2>/dev/null)
  if [ -z "$id" ]; then
    local body="{\"name\":\"$1\",\"mimeType\":\"application/vnd.google-apps.folder\"$( [ -n "$2" ] && echo ",\"parents\":[\"$2\"]" )}"
    id=$(gws drive files create --json "$body" 2>/dev/null | jid)
  fi
  echo "$id"
}
title_for() { grep -m1 "^$1	" "$TITLES" | cut -f2; }
until gws drive about get --params '{"fields":"user(emailAddress)"}' >/dev/null 2>&1; do
  [ $(date +%s) -gt $END ] && exit 0; sleep 60
done
echo "$(date +%FT%T) auth ok"
ROOT=$(folder "Violin Dubbing EN-VI" ""); MIT=$(folder "MIT 18.06 Linear Algebra (Gilbert Strang)" "$ROOT"); CS=$(folder "Stanford CS336 Language Modeling from Scratch" "$ROOT"); HAM=$(folder "Hamel Husain talks" "$ROOT")
echo "$(date +%FT%T) folders root=$ROOT mit=$MIT cs336=$CS hamel=$HAM"
while [ $(date +%s) -lt $END ]; do
  for d in "$BASE/MIT 18.06 Linear Algebra (Gilbert Strang)"/*/ "$BASE/Stanford CS336 Language Modeling from Scratch"/*/; do
    [ -d "$d" ] && [ -f "$d/.synced" ] && [ ! -f "$d/.uploaded" ] || continue
    mp4=$(ls "$d"/*_720p.mp4 2>/dev/null | head -1); [ -n "$mp4" ] || continue
    srt=$(ls "$d"/*.srt 2>/dev/null | head -1)
    case "$d" in
      */MIT\ 18.06*/*) tag=$(basename "$d"); [[ "$tag" == [0-9][0-9]_* ]] || continue; n=${tag%%_*}; id=${tag#*_}; t=$(title_for "$id"); name="MIT 18.06 - ${n} - ${t:-$id}"; parent=$MIT ;;
      */Stanford\ CS336*/*) tag=$(basename "$d"); [[ "$tag" == [0-9][0-9]_* ]] || continue; n=${tag%%_*}; id=${tag#*_}; t=$(title_for "$id"); name="CS336 - ${n} - ${t:-$id}"; parent=$CS ;;
      */docs_llm_run12/) name="Hamel - How to Process Documents at Scale with LLMs (Shreya Shankar)"; parent=$HAM ;;
      */envs_run13/)     name="Hamel - Don't Build Agents, Build Environments Instead"; parent=$HAM ;;
      */video3_run14/)   name="Hamel - How To Use Open Models Effectively"; parent=$HAM ;;
      *) continue ;;
    esac
    name=$(printf "%s" "$name" | sed -e "s#/#-#g" -e "s#:# -#g" -e "s#|#-#g" -e "s#\"#'#g" -e "s#  *# #g" -e "s#^[ .]*##" -e "s#[ .]*\$##")   # same cleaning as rename_local2.py
    # stamp modifiedTime by lecture number so Drive's "last modified" sort equals lecture order (MIT: Jan, CS336: Feb, Hamel: Mar)
    case "$d" in */MIT\ 18.06*/*) mon=01; k=$((10#$n)) ;; */Stanford\ CS336*/*) mon=02; k=$((10#$n)) ;; esac
    stamp() { local fid=$(echo "$1" | jid); [ -n "$fid" ] && gws drive files update --params "{\"fileId\":\"$fid\",\"fields\":\"id\"}" --json "{\"modifiedTime\":\"2026-${mon}-$(printf %02d $((1 + k / 24)))T$(printf %02d $((k % 24))):$2:00Z\"}" >/dev/null 2>>"$S/drive_upload_errors.log"; }
    out=$(gws drive +upload "$mp4" --parent "$parent" --name "${name} [vi dub 720p].mp4" 2>>"$S/drive_upload_errors.log")
    if echo "$out" | jid | grep -q .; then
      stamp "$out" 00
      [ -n "$srt" ] && stamp "$(gws drive +upload "$srt" --parent "$parent" --name "${name} [en].srt" 2>>"$S/drive_upload_errors.log")" 01
      touch "$d/.uploaded"; echo "$(date +%FT%T) uploaded $name"
    else
      echo "$(date +%FT%T) upload failed $name"; sleep 60
    fi
  done
  python3 "$S/rename_local2.py" "$S/titles.tsv" "$S/rename_manifest.tsv" >/dev/null 2>&1   # readable local names after upload
  sleep 120
done
