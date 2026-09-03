"""Rename the local dubbing tree to readable names (courses, per-video folders, every file). Same-directory
renames only, never overwrite, never delete; appends to the manifest (old \t new). Idempotent."""
import os, re, sys, datetime
from pathlib import Path
BASE = Path("/Users/riley/Nhat Cuong/code/00-personal/violin/output/e2e_server")
TITLES = Path(sys.argv[1]); MANIFEST = Path(sys.argv[2])
titles = {l.split("\t", 1)[0]: l.split("\t", 1)[1].strip() for l in TITLES.read_text().splitlines() if "\t" in l}
COURSES = {"mit_18_06": ("MIT 18.06 Linear Algebra (Gilbert Strang)", "MIT 18.06"),
           "cs336": ("Stanford CS336 Language Modeling from Scratch", "CS336")}
HAMEL = {"docs_llm_run12": "How to Process Documents at Scale with LLMs",
         "envs_run13": "Don't Build Agents, Build Environments Instead",
         "video3_run14": "How To Use Open Models Effectively"}
TAIL_RE = re.compile(r"^(?P<id>.+?)_vi(?P<tail>_720p\.mp4|\.srt|\.transcript\.txt|\.fit\.units\.json|\.voices\.json)$")
TAILS = {"_720p.mp4": " [vi dub 720p].mp4", ".srt": " [en].srt", ".transcript.txt": " [vi transcript].txt",
         ".fit.units.json": " [fit units].json", ".voices.json": " [voices].json"}
LOGS = {"run.log", "docs_llm.log", "envs.log", "video3.log"}
def clean(s):
    s = s.replace("/", "-").replace(":", " -").replace("|", "-").replace('"', "'")
    return re.sub(r"\s+", " ", s).strip(" .")[:150]
mf = MANIFEST.open("a"); n_done = 0
def mv(src: Path, dst: Path):
    global n_done
    if src == dst or not src.exists() or dst.exists(): return
    os.rename(src, dst); mf.write(f"{datetime.datetime.now():%FT%T}\t{src}\t{dst}\n"); n_done += 1
def rename_files(d: Path, base: str):
    for f in list(d.iterdir()):
        if f.name.startswith(".") or f.name.startswith(base): continue
        m = TAIL_RE.match(f.name)
        if m: mv(f, d / (base + TAILS[m.group("tail")]))
        elif f.name in LOGS: mv(f, d / (base + " [run].log"))
for course_key, (course_name, prefix) in COURSES.items():
    if (BASE / course_key).is_dir(): mv(BASE / course_key, BASE / course_name)
    cdir = BASE / course_name
    if not cdir.is_dir(): continue
    for d in sorted(cdir.iterdir()):
        if not d.is_dir() or not (d / ".uploaded").exists(): continue   # in-flight items keep NN_<id> for the loops
        m = re.match(r"^(\d{2})_(.+)$", d.name)
        if m:
            n, vid = m.groups(); title = titles.get(vid, vid); new_dir = cdir / clean(f"{n} - {title}")
        else:
            n = d.name[:2]; title = d.name[5:]; new_dir = d
        base = clean(f"{prefix} - {n} - {title}")
        rename_files(d, base); mv(d, new_dir)
for key, title in HAMEL.items():
    base = clean(f"Hamel Husain - {title}")
    for d in (BASE / key, BASE / base):
        if d.is_dir() and (d / ".uploaded").exists():
            rename_files(d, base); mv(d, BASE / base)
mf.close(); print(f"renamed {n_done}")
