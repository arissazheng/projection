"""Serve the local room importer and walkthrough library."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from flask import Flask, abort, jsonify, redirect, request, send_from_directory, url_for

from .reconstruct_room import DEFAULT_STEPS, REPO, TOOLS, brush_binary, input_kind, reconstruct


JOBS = REPO / '.walkthrough-data'
VIEWER = TOOLS / 'supersplat' / 'package' / 'public'
LOG = logging.getLogger(__name__)
ACTIVE_STATES = {'queued', 'preparing', 'features', 'matching', 'alignment', 'training'}


def create_scene(title: str) -> tuple[Path, dict]:
    scene = JOBS / str(uuid4())
    scene.mkdir(parents=True)
    return scene, {'id': scene.name, 'title': title, 'createdAt': datetime.now(timezone.utc).isoformat()}


def write_status(scene: Path, status: dict) -> None:
    temporary = scene / 'status.tmp'
    temporary.write_text(json.dumps(status), encoding='utf-8')
    temporary.replace(scene / 'status.json')


def publish_scene(scene: Path, base: dict, result: Path, camera: dict, registered: int, count: int) -> None:
    # The pinned SuperSplat v2 settings schema requires every post-effect even when disabled.
    settings = {
        'version': 2,
        'tonemapping': 'none',
        'highPrecisionRendering': False,
        'background': {'color': [0, 0, 0]},
        'postEffectSettings': {
            'sharpness': {'enabled': False, 'amount': 0},
            'bloom': {'enabled': False, 'intensity': 0.1, 'blurLevel': 2},
            'grading': {'enabled': False, 'brightness': 1, 'contrast': 1, 'saturation': 1, 'tint': [1, 1, 1]},
            'vignette': {'enabled': False, 'intensity': 0.5, 'inner': 0.3, 'outer': 0.75, 'curvature': 1},
            'fringing': {'enabled': False, 'intensity': 0.5},
        },
        'cameras': [{'initial': camera}],
        'animTracks': [],
        'annotations': [],
        'startMode': 'default',
    }
    (result / 'settings.json').write_text(json.dumps(settings), encoding='utf-8')
    result.rename(scene / 'result')
    write_status(scene, {**base, 'state': 'ready', 'message': f'{registered} of {count} views aligned',
                         'registered': registered, 'views': count})


def run_scene(scene: Path, base: dict, paths: list[Path], steps: int) -> None:
    def report(state: str, message: str) -> None:
        write_status(scene, {**base, 'state': state, 'message': message})

    try:
        with TemporaryDirectory(dir=scene) as workspace:
            work = Path(workspace)
            result = work / 'result'
            camera, registered, count = reconstruct(paths, work / 'dataset', result, steps, report)
            publish_scene(scene, base, result, camera, registered, count)
    except ValueError as error:
        report('failed', str(error))
    except Exception:
        LOG.exception('Room reconstruction failed')
        report('failed', 'Reconstruction stopped unexpectedly. See the terminal for details.')


def create_app(steps: int) -> Flask:
    app = Flask(__name__, static_folder=str(REPO / 'viewer'), static_url_path='/viewer')
    app.config.update(MAX_CONTENT_LENGTH=1024**3, TRUSTED_HOSTS=['localhost', '127.0.0.1'])
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='room-reconstruction')
    active: set[str] = set()

    def scene_status(scene: Path) -> dict:
        status = json.loads((scene / 'status.json').read_text(encoding='utf-8'))
        if status['state'] in ACTIVE_STATES and scene.name not in active:
            status.update(state='failed', message='Processing was interrupted. Import the capture again.')
        if status['state'] == 'ready':
            status['url'] = url_for('open_scene', scene_id=scene.name)
        return status

    @app.get('/')
    def home():
        return send_from_directory(REPO / 'viewer' / 'walkthrough', 'index.html')

    @app.get('/api/scenes')
    def list_scenes():
        JOBS.mkdir(exist_ok=True)
        scenes = [scene_status(path.parent) for path in JOBS.glob('*/status.json')]
        return jsonify(sorted(scenes, key=lambda status: status['createdAt'], reverse=True))

    @app.post('/api/scenes')
    def upload_scene():
        origin = request.headers.get('Origin')
        if origin and origin != request.host_url.rstrip('/'):
            abort(403)
        uploads = request.files.getlist('files')
        filenames = [Path(upload.filename or '').name for upload in uploads]
        try:
            kind = input_kind(filenames)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        title = request.form.get('title', '').strip()
        if not title:
            title = Path(filenames[0]).stem if kind == 'video' else 'Room photos'
        scene, base = create_scene(title[:80])

        # Only this job owns its upload directory. Clean it up after processing.
        intake = TemporaryDirectory(dir=scene, prefix='upload-')
        try:
            paths = []
            for index, (upload, filename) in enumerate(zip(uploads, filenames), start=1):
                path = Path(intake.name) / f'{index:05d}{Path(filename).suffix.lower()}'
                upload.save(path)
                paths.append(path)
            active.add(scene.name)
            write_status(scene, {**base, 'state': 'queued', 'message': 'Waiting to reconstruct…'})
            task = worker.submit(run_scene, scene, base, paths, steps)
        except Exception:
            active.discard(scene.name)
            intake.cleanup()
            raise

        def finish(_):
            active.discard(scene.name)
            intake.cleanup()

        task.add_done_callback(finish)
        return jsonify(scene_status(scene)), 202

    @app.errorhandler(413)
    def upload_too_large(_):
        return jsonify(error='Choose a capture smaller than 1 GB.'), 413

    @app.get('/room/<uuid:scene_id>')
    def open_scene(scene_id):
        scene = JOBS / str(scene_id)
        if not (scene / 'result' / 'scene.ply').is_file():
            abort(404)
        return redirect(url_for('viewer_files', filename='index.html', noanim='', reticle='',
                                content=url_for('room_files', scene_id=scene_id, filename='scene.ply'),
                                settings=url_for('room_files', scene_id=scene_id, filename='settings.json')))

    @app.get('/room/<uuid:scene_id>/<filename>')
    def room_files(scene_id, filename):
        if filename not in {'scene.ply', 'settings.json'}:
            abort(404)
        return send_from_directory(JOBS / str(scene_id) / 'result', filename)

    @app.get('/splat-viewer/<path:filename>')
    def viewer_files(filename):
        return send_from_directory(VIEWER, filename)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8181)
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS, help='Brush training steps per import')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if not (VIEWER / 'index.html').exists():
        parser.error('Run bash scripts/setup_walkthrough.sh to install the browser viewer.')
    try:
        brush_binary()
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO)
    create_app(args.steps).run(host='127.0.0.1', port=args.port, use_reloader=False)


if __name__ == '__main__':
    main()
