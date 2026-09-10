"""Fetch every qualifying SuiteSparse matrix the corpus does not already hold.

The corpus was assembled by hand and is a sample rather than a census: of the 1,184
square real matrices at n <= 10,000, it holds 830 (plus 100 self-generated random ones).
Six more are present but truncated -- their headers declare far more nonzeros than the
files contain, which is why they never load.

Every download is verified before it is kept: the header's nonzero count is compared
against the number of entry lines actually present. That is exactly the check that would
have caught the six bad files at the time they were fetched instead of months later,
inside a benchmark run.

    python tools/download_corpus.py --dry-run      # list what is missing
    python tools/download_corpus.py                # fetch matrices in existing domains
    python tools/download_corpus.py --all-kinds    # fetch every qualifying matrix
"""
import argparse
import csv
import io
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://suitesparse-collection-website.herokuapp.com/MM"
MAX_N = 10_000

#: Present locally but truncated: header nnz far exceeds the entry lines in the file.
BROKEN = ("cavity20", "nemeth17", "nemeth22", "nemeth23", "nemeth25", "psmigr_2")


def normalise(kind):
    return kind.strip().lower().replace("/", "_").replace(" ", "_")


def verify(path):
    """Return (ok, declared, present). A Matrix Market file must contain as many entry
    lines as its header declares; a short one is a truncated download."""
    declared = present = 0
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("%"):
                continue
            if declared == 0:
                parts = line.split()
                if len(parts) < 3:
                    return False, 0, 0
                declared = int(parts[2])
                continue
            if line.strip():
                present += 1
    return present >= declared, declared, present


def fetch(group, name, dest_dir, retries=3):
    """Download one matrix, verify it, and keep it only if it is complete."""
    url = f"{BASE}/{group}/{name}.tar.gz"
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{name}.mtx"
    for attempt in range(1, retries + 1):
        tmp = Path(tempfile.mkdtemp())
        try:
            with urllib.request.urlopen(url, timeout=120) as r, \
                    io.open(tmp / "m.tar.gz", "wb") as out:
                shutil.copyfileobj(r, out)
            with tarfile.open(tmp / "m.tar.gz", "r:gz") as tf:
                member = next((m for m in tf.getmembers()
                               if m.name.endswith(f"{name}.mtx")), None)
                if member is None:
                    return False, "no .mtx in archive"
                member.name = Path(member.name).name
                tf.extract(member, tmp)
            ok, declared, present = verify(tmp / f"{name}.mtx")
            if not ok:
                if attempt < retries:
                    time.sleep(2 * attempt)
                    continue
                return False, f"truncated: header {declared}, file {present}"
            shutil.move(str(tmp / f"{name}.mtx"), str(target))
            return True, f"{declared} nnz"
        except (urllib.error.URLError, tarfile.TarError, OSError) as e:
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            return False, f"{type(e).__name__}: {e}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return False, "exhausted retries"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset_large"))
    ap.add_argument("--all-kinds", action="store_true",
                    help="also fetch kinds the corpus does not yet contain")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    stats = data / "ssstats.csv"
    if not stats.exists():
        sys.exit(f"no index at {stats}")

    have = {p.stem for p in data.rglob("*.mtx")}
    local_kinds = {p.parent.name for p in data.rglob("*.mtx")}

    wanted = []
    for row in list(csv.reader(io.open(stats, encoding="utf-8")))[2:]:
        if len(row) < 12:
            continue
        group, name = row[0], row[1]
        try:
            nr, nc, nnz, real = int(row[2]), int(row[3]), int(row[4]), int(row[5])
        except ValueError:
            continue
        if nr != nc or nr > MAX_N or real != 1:
            continue
        kind = normalise(row[11])
        if name in have and name not in BROKEN:
            continue
        if not args.all_kinds and kind not in local_kinds:
            continue
        wanted.append((group, name, nr, nnz, kind))

    wanted.sort(key=lambda w: w[3])
    if args.limit:
        wanted = wanted[:args.limit]

    redownload = [w for w in wanted if w[1] in BROKEN]
    print(f"corpus at {data}: {len(have)} matrices in {len(local_kinds)} domains")
    print(f"to fetch: {len(wanted)}  ({len(redownload)} of them re-downloads of "
          f"truncated files)")
    print(f"estimated size: {sum(w[3] for w in wanted) * 20 / 1e6:.0f} MB")
    if args.dry_run:
        for g, n, nr, nnz, k in wanted[:40]:
            print(f"   {g}/{n:<24} n={nr:<7} nnz={nnz:<10} {k}")
        if len(wanted) > 40:
            print(f"   ... and {len(wanted) - 40} more")
        return

    ok = failed = 0
    t0 = time.perf_counter()
    for i, (group, name, nr, nnz, kind) in enumerate(wanted, 1):
        elapsed = (time.perf_counter() - t0) / 60
        rate = i / max(elapsed, 1e-9)
        eta = (len(wanted) - i) / rate if rate else float("nan")
        good, note = fetch(group, name, data / kind)
        ok += good
        failed += not good
        print(f"[{i:>4}/{len(wanted)}] {name:<24} n={nr:<7} {kind:<34} "
              f"{'OK ' if good else 'FAIL'} {note:<28} "
              f"| {elapsed:5.1f} min, ETA {eta:5.1f}", flush=True)

    print(f"\ndownloaded {ok}, failed {failed}, in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"corpus now: {len(list(data.rglob('*.mtx')))} matrices")


if __name__ == "__main__":
    main()
