"""Upload the IRIS checkpoint archives to Zenodo and mint a DOI.

Zenodo publication is PERMANENT: a published record cannot be deleted, only
superseded by a new version. This script therefore stops after uploading and
setting metadata, leaving the deposition as a draft for you to review in the
browser. Publishing is a separate, explicit step (--publish, or the button on
the Zenodo page).

    # 1. dry run against the sandbox (throwaway DOIs, safe to experiment)
    python src/zenodo_upload.py --sandbox --token-file ~/.zenodo_sandbox_token

    # 2. the real thing -- uploads and stops at draft
    python src/zenodo_upload.py --token-file ~/.zenodo_token

    # 3. publish once you are happy with the draft
    python src/zenodo_upload.py --token-file ~/.zenodo_token --publish-id 1234567
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(ROOT, "zenodo_upload")
REPO = "https://github.com/samim-reza/IRIS"

DESCRIPTION = """<p>Trained model checkpoints supporting the paper
<em>IRIS: reliability-gated label acceptance and introspective instance
selection for human-in-the-loop deep active learning under imperfect
annotators</em>.</p>

<p>One checkpoint per experimental run: the final trained model after the last
acquisition round, for every combination of dataset, acquisition method,
oracle noise level and seed (285 runs in total). Each file stores the backbone
and auxiliary head weights together with the audit trail needed to re-measure
its reported accuracy independently &mdash; the indices of the examples it was
trained on, the labels the (possibly noisy) synthetic oracle returned for them,
the training configuration and the PyTorch version.</p>

<p>Archives are split by dataset so a specific claim can be checked without
downloading the whole set:</p>
<ul>
<li><code>iris-checkpoints-fmnist.tar</code> &mdash; Fashion-MNIST, 114 models</li>
<li><code>iris-checkpoints-bloodmnist.tar</code> &mdash; BloodMNIST, 114 models</li>
<li><code>iris-checkpoints-cifar10.tar</code> &mdash; CIFAR-10, 57 models</li>
<li><code>MANIFEST.csv</code> &mdash; every model with its recorded test accuracy</li>
<li><code>SHA256SUMS.txt</code> &mdash; checksums for all of the above</li>
</ul>

<p>Training was deterministic (cuDNN autotuning disabled, dataloader seeded),
so re-running the code reproduces these numbers exactly rather than
approximately. To verify: download an archive, unpack it into
<code>checkpoints/</code> in a clone of the code repository, and run
<code>python src/verify_checkpoint.py</code>, which reloads each model and
re-measures it on the untouched test set.</p>

<p>Code, logs and the scripts that regenerate every number, table and figure:
<a href="%s">%s</a></p>""" % (REPO, REPO)

METADATA = {
    "metadata": {
        "upload_type": "dataset",
        "title": ("IRIS: trained model checkpoints for reliability-gated "
                  "label acceptance in human-in-the-loop deep active learning"),
        "description": DESCRIPTION,
        "creators": [
            {"name": "Samim"},
            {"name": "Shihavuddin, ASM"},
        ],
        "license": "cc-by-4.0",
        "access_right": "open",
        "keywords": [
            "active learning", "human-in-the-loop", "noisy labels",
            "deep learning", "model checkpoints", "reproducibility",
            "BloodMNIST", "Fashion-MNIST", "CIFAR-10",
        ],
        "related_identifiers": [
            {"identifier": REPO, "relation": "isSupplementTo",
             "resource_type": "software", "scheme": "url"},
        ],
    }
}


def api(url, token, method="GET", data=None, headers=None):
    h = {"Authorization": f"Bearer {token}"}
    body = None
    if data is not None:
        h["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"\nZenodo returned HTTP {e.code} for {method} {url}\n"
                 f"{e.read().decode()[:500]}\n")


def put_file(bucket, path, token):
    size = os.path.getsize(path)
    name = os.path.basename(path)
    print(f"  uploading {name} ({size / 1e6:.0f} MB) ...", end="", flush=True)
    with open(path, "rb") as f:
        req = urllib.request.Request(
            f"{bucket}/{name}", data=f, method="PUT",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/octet-stream",
                     "Content-Length": str(size)})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                r.read()
        except urllib.error.HTTPError as e:
            sys.exit(f"\n  FAILED: HTTP {e.code}\n{e.read().decode()[:400]}\n")
    print(" ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", required=True,
                    help="file containing only your Zenodo access token")
    ap.add_argument("--sandbox", action="store_true",
                    help="use sandbox.zenodo.org (throwaway DOIs)")
    ap.add_argument("--publish-id", type=int,
                    help="publish an existing draft by deposition id")
    args = ap.parse_args()

    base = ("https://sandbox.zenodo.org/api" if args.sandbox
            else "https://zenodo.org/api")
    token = open(os.path.expanduser(args.token_file)).read().strip()
    if not token:
        sys.exit("token file is empty")

    if args.publish_id:
        r = api(f"{base}/deposit/depositions/{args.publish_id}/actions/publish",
                token, method="POST")
        print(f"\nPUBLISHED\n  DOI: {r.get('doi')}\n  URL: "
              f"{r.get('links', {}).get('record_html')}\n")
        return 0

    files = [os.path.join(UPLOAD_DIR, f) for f in sorted(os.listdir(UPLOAD_DIR))]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        sys.exit(f"nothing to upload in {UPLOAD_DIR}")
    total = sum(os.path.getsize(f) for f in files)
    print(f"{'SANDBOX' if args.sandbox else 'ZENODO'}: {len(files)} files, "
          f"{total / 1e9:.2f} GB\n")

    dep = api(f"{base}/deposit/depositions", token, method="POST", data={})
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"draft deposition {dep_id} created")

    for f in files:
        put_file(bucket, f, token)

    api(f"{base}/deposit/depositions/{dep_id}", token, method="PUT",
        data=METADATA)
    print("\nmetadata set. DRAFT ONLY -- nothing is public yet.")
    print(f"  review:  {dep['links']['html']}")
    print(f"  publish: python src/zenodo_upload.py --token-file {args.token_file}"
          f"{' --sandbox' if args.sandbox else ''} --publish-id {dep_id}")
    print("\nPublishing is permanent: a published record cannot be deleted, "
          "only superseded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
