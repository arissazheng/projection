"""Assemble Room_Window_Colab.ipynb with colab_room.py embedded."""
import json
from pathlib import Path

here = Path(__file__).parent
script = (here / 'colab_room.py').read_text()

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')})


def code(text, title=None):
    meta = {'cellView': 'form'} if title else {}
    cells.append({'cell_type': 'code', 'metadata': meta, 'execution_count': None, 'outputs': [],
                  'source': text.strip('\n')})


md('''
# Room Window: video → 3D walkthrough (Colab)

Runs [arissazheng/projection](https://github.com/arissazheng/projection) on a Colab GPU.
It uses the repo's own frame extraction and COLMAP alignment. The training step uses **gsplat**
(CUDA) instead of Brush, which needs Vulkan graphics drivers that Colab doesn't reliably have.
The result is the same `scene.ply` and `settings.json`, opened in the repo's own walkthrough viewer.

**Before you start:** Runtime → Change runtime type → **T4 GPU** → Save.
Then run the cells top to bottom. Every cell ends with a ✅ or ❌ line.

| Step | Time on a T4 |
|---|---|
| 1–3 Setup + compile gsplat | ~8–12 min (first run only) |
| 4 Add video | ~1 min |
| 5 Align + train | ~10 min alignment, ~10–15 min training |
| 6 Save to Drive | ~1 min |
| 7 Find objects (SAM 3) | ~5–15 min |
| 8 Open viewer | ~1 min |
''')

code('''
#@title 1 · Check GPU
import subprocess, torch
if not torch.cuda.is_available():
    print('❌ STEP 1 FAILED: no GPU. Runtime → Change runtime type → T4 GPU, then rerun.')
else:
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f'✅ STEP 1 DONE: {name}, {mem:.0f} GB, torch {torch.__version__}, CUDA {torch.version.cuda}')
''', title=True)

code('''
#@title 2 · Install the repo and dependencies (~2 min)
#@markdown Rerun this cell to pull the newest code. Set BRANCH if the faster version lives on another branch.
REPO_URL = "https://github.com/arissazheng/projection" #@param {type:"string"}
BRANCH = "main" #@param {type:"string"}
%cd /content
!rm -rf /content/projection && git clone -q --depth 1 -b "$BRANCH" "$REPO_URL" /content/projection && git -C /content/projection log -1 --format='Repo commit: %h %s (%cr)'
!pip install -q "pycolmap==4.2.1" "imageio-ffmpeg==0.8.1" "pillow-heif>=1,<2" "Flask>=3.1,<4" ninja gsplat==1.5.3 "transformers>=5.24,<6" accelerate 2>&1 | grep -v "already satisfied" | tail -3
# The repo's walkthrough viewer (same pinned SuperSplat viewer as scripts/setup_walkthrough.sh)
!mkdir -p /content/projection/.walkthrough-tools/supersplat && cd /content/projection/.walkthrough-tools/supersplat && curl -fsSL https://registry.npmjs.org/@playcanvas/supersplat-viewer/-/supersplat-viewer-1.43.3.tgz | tar -xz
import os, importlib
ok = os.path.isfile('/content/projection/pipeline/reconstruct_room.py') and os.path.isfile('/content/projection/.walkthrough-tools/supersplat/package/public/index.html')
# Extra requirements the updated repo may add (Brush-specific setup is skipped on Colab).
!pip install -q -r /content/projection/requirements-walkthrough.txt 2>&1 | grep -v "already satisfied" | tail -3
# The runner below calls these repo functions; warn clearly if an update renamed any of them.
import sys, inspect
sys.path.insert(0, '/content/projection')
for mod in [m for m in list(sys.modules) if m == 'pipeline' or m.startswith('pipeline.')]:
    del sys.modules[mod]
try:
    import pycolmap, gsplat
    from pipeline import reconstruct_room as rr, walkthrough as wt
    needed = {rr: ['input_kind', 'prepare_images', 'align_images', 'prepare_training_data', 'orient_model'],
              wt: ['create_scene', 'publish_scene', 'create_app']}
    missing = [f'{m.__name__}.{f}' for m, fs in needed.items() for f in fs if not hasattr(m, f)]
    if not ok:
        print('❌ STEP 2 FAILED: repo or viewer files missing')
    elif missing:
        print(f'❌ STEP 2 FAILED: the repo update changed functions this notebook uses: {missing}. The runner needs updating.')
    else:
        print(f'✅ STEP 2 DONE: pycolmap {pycolmap.__version__}, gsplat {gsplat.__version__}, repo API compatible')
except Exception as e:
    print(f'❌ STEP 2 FAILED: {e!r}')
''', title=True)

code('''
#@title 3 · Compile gsplat's CUDA kernels (~5–10 min, first run only)
import os, subprocess, torch
major, minor = torch.cuda.get_device_capability(0)
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
os.environ.setdefault('MAX_JOBS', str(os.cpu_count() or 2))
r = subprocess.run(['python', '-c', 'from gsplat.cuda._backend import _C; print(_C is not None)'],
                   capture_output=True, text=True, env=os.environ)
open('/content/gsplat_build.log', 'w').write(r.stdout + r.stderr)
if r.returncode == 0 and 'True' in r.stdout:
    print('✅ STEP 3 DONE: gsplat CUDA kernels ready')
else:
    print((r.stdout + r.stderr)[-2500:])
    print('❌ STEP 3 FAILED: gsplat did not compile (full log: /content/gsplat_build.log)')
''', title=True)

code('%%writefile /content/colab_room.py\n' + script)
code('%%writefile /content/segment_objects.py\n' + (here / 'segment_objects.py').read_text())

code('''
#@title 4 · Add your room video
#@markdown Paste a Google Drive **link** (also works for files shared with you) or a path like `MyDrive/room.mov`.<br>
#@markdown Leave blank to pick a file from your computer instead.
VIDEO = "" #@param {type:"string"}
import os, re, glob, shutil
os.makedirs('/content/input', exist_ok=True)
for old in glob.glob('/content/input/*'):
    os.remove(old)
v = VIDEO.strip()
m = re.search(r'/d/([-\\w]{20,})|[?&]id=([-\\w]{20,})', v) or re.fullmatch(r'()([-\\w]{25,})', v)
try:
    if m:
        from google.colab import auth
        auth.authenticate_user()
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        drive_api = build('drive', 'v3', cache_discovery=False)
        fid = m.group(1) or m.group(2)
        meta = drive_api.files().get(fileId=fid, fields='name,size', supportsAllDrives=True).execute()
        dest = os.path.join('/content/input', meta['name'])
        with open(dest, 'wb') as f:
            dl = MediaIoBaseDownload(f, drive_api.files().get_media(fileId=fid, supportsAllDrives=True), chunksize=64 * 2**20)
            finished = False
            while not finished:
                _, finished = dl.next_chunk()
    elif v:
        from google.colab import drive
        drive.mount('/content/drive')
        rel = v.removeprefix('/content/drive/').lstrip('/')
        src = os.path.join('/content/drive', rel if rel.startswith(('MyDrive', 'Shareddrives')) else 'MyDrive/' + rel)
        shutil.copy(src, '/content/input/')
    else:
        from google.colab import files
        for name, data in files.upload().items():
            open(os.path.join('/content/input', name), 'wb').write(data)
except Exception as e:
    print(f'download error: {e!r}')
inputs = sorted(glob.glob('/content/input/*'))
size = sum(os.path.getsize(p) for p in inputs) / 2**20
print(f'✅ STEP 4 DONE: {len(inputs)} file(s), {size:.0f} MB: {[os.path.basename(p) for p in inputs]}' if inputs
      else '❌ STEP 4 FAILED: no file added')
''', title=True)

code('''
#@title 5 · Align the views and train the 3D scene (~20–30 min)
STEPS = 7000 #@param {type:"integer"}
import subprocess, time, os, glob, torch
major, minor = torch.cuda.get_device_capability(0)
env = {**os.environ, 'TORCH_CUDA_ARCH_LIST': f'{major}.{minor}', 'PYTHONUNBUFFERED': '1'}
open('/content/progress.log', 'w').close()
inputs = sorted(glob.glob('/content/input/*'))
with open('/content/run.log', 'w') as full_log:
    proc = subprocess.Popen(['python', '/content/colab_room.py', *inputs, '--repo', '/content/projection',
                             '--out', '/content/scene', '--steps', str(STEPS)],
                            stdout=full_log, stderr=subprocess.STDOUT, env=env)
    shown = 0
    while True:
        done = proc.poll() is not None
        lines = open('/content/progress.log').read().splitlines()
        for line in lines[shown:]:
            print(line, flush=True)
        shown = len(lines)
        if done:
            break
        time.sleep(10)
if proc.returncode == 0 and os.path.isfile('/content/scene/scene.ply'):
    mb = os.path.getsize('/content/scene/scene.ply') / 2**20
    print(f'✅ STEP 5 DONE: scene.ply {mb:.0f} MB')
else:
    tail = open('/content/run.log').read().splitlines()
    print('\\n'.join(l for l in tail if not l.startswith(('I20', 'W20')))[-3000:])
    print(f'❌ STEP 5 FAILED (exit {proc.returncode}). Full log: /content/run.log')
''', title=True)

code('''
#@title 6 · Save to / load from Google Drive
#@markdown **save** after step 5 so a Colab reset doesn't cost you alignment + training.<br>
#@markdown **load** in a fresh session (after steps 1–3) to skip steps 4–5.
ACTION = "save" #@param ["save", "load"]
RUN_NAME = "room1" #@param {type:"string"}
import os, subprocess
from google.colab import drive
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')
folder = f'/content/drive/MyDrive/room_window/{RUN_NAME}'
archive = f'{folder}/run.tar'
paths = ['content/work/camera.json', 'content/work/dataset/undistorted', 'content/scene', 'content/objects']
if ACTION == 'save':
    os.makedirs(folder, exist_ok=True)
    present = [p for p in paths if os.path.exists('/' + p)]
    r = subprocess.run(['tar', '-cf', archive, '-C', '/', *present], capture_output=True, text=True)
    ok = r.returncode == 0
    msg = f'saved {", ".join(os.path.basename(p) for p in present)} → {archive} ({os.path.getsize(archive) / 2**20:.0f} MB)' if ok else r.stderr[-500:]
else:
    r = subprocess.run(['tar', '-xf', archive, '-C', '/'], capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.isfile('/content/scene/scene.ply')
    msg = f'loaded {archive}' if ok else (r.stderr[-500:] or 'scene.ply missing in archive')
print(f'✅ STEP 6 DONE: {msg}' if ok else f'❌ STEP 6 FAILED: {msg}')
''', title=True)

code('''
#@title 7 · Find interactable objects with SAM 3 (~5–15 min)
#@markdown Needs a Colab secret named `HF_TOKEN` (key icon, left sidebar) from a Hugging Face account with SAM 3 access.<br>
#@markdown Edit the list to change what counts as interactable (comma-separated).
PROMPTS = "mug, cup, bottle, bowl, plate, book, laptop, keyboard, phone, remote control, lamp, pillow, blanket, chair, stool, backpack, bag, box, basket, plant, vase, picture frame, clock, speaker, shoe, toy" #@param {type:"string"}
MIN_SCORE = 0.5 #@param {type:"number"}
import os, subprocess, time
from google.colab import userdata
try:
    token = userdata.get('HF_TOKEN')
except Exception as e:
    token = None
    print(f'Could not read HF_TOKEN secret: {e!r}')
if token:
    env = {**os.environ, 'HF_TOKEN': token, 'PYTHONUNBUFFERED': '1'}
    open('/content/progress.log', 'w').close()
    with open('/content/segment.log', 'w') as full_log:
        proc = subprocess.Popen(['python', '/content/segment_objects.py', '--work', '/content/work',
                                 '--scene', '/content/scene/scene.ply', '--out', '/content/objects',
                                 '--prompts', PROMPTS, '--min-score', str(MIN_SCORE)],
                                stdout=full_log, stderr=subprocess.STDOUT, env=env)
        shown = 0
        while True:
            done = proc.poll() is not None
            lines = open('/content/progress.log').read().splitlines()
            for line in lines[shown:]:
                print(line, flush=True)
            shown = len(lines)
            if done:
                break
            time.sleep(10)
    if proc.returncode == 0 and os.path.isfile('/content/objects/objects.json'):
        import json
        from IPython.display import Image, display
        objs = json.load(open('/content/objects/objects.json'))['objects']
        for o in objs:
            print(f"  {o['id']:02d} {o['label']:<16} score {o['score']:.2f}  {o['splats']:>7,} splats  seen in {o['frames_seen']} frames")
        if os.path.isfile('/content/objects/preview.jpg'):
            display(Image('/content/objects/preview.jpg'))
        print(f'✅ STEP 7 DONE: {len(objs)} objects in /content/objects (rerun step 6 "save" to keep them)')
    else:
        tail = open('/content/segment.log').read()
        print(tail[-3000:])
        hint = ' (403/gated: accept SAM 3 terms on huggingface.co/facebook/sam3 with the same account as the token)' if ('403' in tail or 'gated' in tail.lower()) else ''
        print(f'❌ STEP 7 FAILED (exit {proc.returncode}){hint}. Full log: /content/segment.log')
else:
    print('❌ STEP 7 FAILED: add a Colab secret named HF_TOKEN and allow this notebook to access it')
''', title=True)

code('''
#@title 8 · Open the walkthrough viewer
import sys, threading, socket, logging
sys.path.insert(0, '/content/projection')
from pipeline import walkthrough as wt
scene_id = open('/content/scene/scene_id.txt').read().strip()

def port_open(port):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

if not port_open(8181):
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app = wt.create_app(7000)
    app.config['TRUSTED_HOSTS'] = None  # Colab's proxy uses its own host name
    threading.Thread(target=lambda: app.run(host='127.0.0.1', port=8181, use_reloader=False), daemon=True).start()
    import time; time.sleep(2)

from google.colab.output import eval_js
base = eval_js('google.colab.kernel.proxyPort(8181)')
url = f"{base.rstrip('/')}/room/{scene_id}"
from IPython.display import HTML, display
display(HTML(f'<a href="{url}" target="_blank" style="font-size:18px">▶ Open walkthrough</a>'))
print('Controls: 2 = Fly mode, WASD = move, drag/G = look, Q/E = up/down, R = reset')
print(f'✅ STEP 8 DONE: {url}' if port_open(8181) else '❌ STEP 8 FAILED: viewer server did not start')
''', title=True)

code('''
#@title 9 · (Optional) Download the scene + objects
import shutil
from google.colab import files
import os
os.makedirs('/content/export', exist_ok=True)
for d in ('scene', 'objects'):
    if os.path.isdir(f'/content/{d}'):
        shutil.copytree(f'/content/{d}', f'/content/export/{d}', dirs_exist_ok=True)
shutil.make_archive('/content/room_scene', 'zip', '/content/export')
files.download('/content/room_scene.zip')
print('✅ STEP 9 DONE: room_scene.zip (scene/ + objects/). You can also drag any .ply into https://superspl.at/editor')
''', title=True)

nb = {
    'nbformat': 4, 'nbformat_minor': 0,
    'metadata': {'colab': {'provenance': [], 'gpuType': 'T4'}, 'accelerator': 'GPU',
                 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                 'language_info': {'name': 'python'}},
    'cells': cells,
}
out = here / 'Room_Window_Colab.ipynb'
out.write_text(json.dumps(nb, indent=1))
print('wrote', out)
