"""Reconstruct a room from one video or a set of overlapping photos."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import numpy as np
import pycolmap
from PIL import Image, ImageOps


REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / '.walkthrough-tools'
VIDEO_EXTENSIONS = {'.mov', '.mp4', '.m4v', '.avi', '.mkv', '.webm'}
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.tif', '.tiff'}
MIN_VIEWS = 12
MAX_PHOTOS = 500
FRAME_SIZE = 1280
TRAINING_IMAGE_SIZE = 960
BRUSH_HEADER_BYTES = 16_387
VIDEO_FRAME_LIMIT = 160
DEFAULT_STEPS = 5000
COLMAP_THREADS = min(8, os.cpu_count() or 1)
ProgressCallback = Callable[[str, str], None]
LOG = logging.getLogger(__name__)


def input_kind(paths: list[str] | list[Path]) -> str:
    extensions = [Path(path).suffix.lower() for path in paths]
    if len(extensions) == 1 and extensions[0] in VIDEO_EXTENSIONS:
        return 'video'
    if MIN_VIEWS <= len(extensions) <= MAX_PHOTOS and all(ext in PHOTO_EXTENSIONS for ext in extensions):
        return 'photos'
    raise ValueError(f'Select one video or {MIN_VIEWS}–{MAX_PHOTOS} overlapping photos of the same room.')


def brush_binary() -> str:
    if override := os.environ.get('BRUSH_BIN'):
        path = Path(override).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise ValueError('BRUSH_BIN must point to an executable Brush 0.3.0 binary.')
    for path in TOOLS.glob('brush/*/brush_app'):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    if path := shutil.which('brush_app'):
        return str(Path(path).resolve())
    raise ValueError('Brush 0.3.0 is missing. Run scripts/setup_walkthrough.sh or set BRUSH_BIN.')


def _extract_video_frames(video: Path, output: Path) -> None:
    frames = imageio_ffmpeg.read_frames(str(video))
    try:
        duration = float(next(frames).get('duration') or 0)
    except (OSError, RuntimeError, StopIteration) as error:
        raise ValueError('Could not decode the video. Try exporting it as a MOV or MP4 file.') from error
    finally:
        frames.close()
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Could not determine the video duration.')

    sample_rate = min(6, VIDEO_FRAME_LIMIT / duration)
    filters = f'fps={sample_rate:.8f},scale={FRAME_SIZE}:{FRAME_SIZE}:force_original_aspect_ratio=decrease'
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-i', str(video), '-map', '0:v:0', '-vf', filters, '-frames:v', str(VIDEO_FRAME_LIMIT),
        '-q:v', '2', str(output / '%05d.jpg'),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        LOG.error('FFmpeg: %s', result.stderr.strip())
        raise ValueError('Could not extract frames from the video. Try exporting it as a MOV or MP4 file.')


def _prepare_photos(paths: list[Path], output: Path) -> None:
    if any(path.suffix.lower() in {'.heic', '.heif'} for path in paths):
        from pillow_heif import register_heif_opener

        register_heif_opener()
    for index, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS)
                image.save(output / f'{index:05d}.jpg', quality=93, exif=image.getexif())
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f'Could not open photo {index}. Try exporting it as a JPEG.') from error


def prepare_images(paths: list[Path], kind: str, output: Path) -> int:
    output.mkdir()
    if kind == 'video':
        _extract_video_frames(paths[0], output)
    else:
        _prepare_photos(paths, output)
    count = sum(1 for _ in output.glob('*.jpg'))
    if count < MIN_VIEWS:
        raise ValueError(f'Only {count} usable frames. Record a longer video while moving through the room.')
    return count


def align_images(dataset: Path, images: Path, kind: str, count: int,
                 report: ProgressCallback) -> pycolmap.Reconstruction:
    database = dataset / 'features.db'
    maps = dataset / 'maps'
    maps.mkdir()
    extraction = pycolmap.FeatureExtractionOptions(num_threads=COLMAP_THREADS)
    extraction.sift.max_num_features = 4096
    matching = pycolmap.FeatureMatchingOptions(num_threads=COLMAP_THREADS)
    device = pycolmap.Device.cpu

    report('features', f'Finding visual features in {count} images…')
    camera_mode = pycolmap.CameraMode.SINGLE if kind == 'video' else pycolmap.CameraMode.AUTO
    pycolmap.extract_features(database, images, camera_mode=camera_mode, extraction_options=extraction, device=device)

    def map_views() -> pycolmap.Reconstruction | None:
        report('alignment', 'Estimating where each image was taken…')
        options = pycolmap.IncrementalPipelineOptions(num_threads=COLMAP_THREADS)
        candidates = pycolmap.incremental_mapping(database, images, maps, options=options)
        return max(candidates.values(), default=None, key=lambda model: model.num_reg_images())

    report('matching', 'Matching overlapping images…')
    if kind == 'video':
        pycolmap.match_sequential(database, matching, pycolmap.SequentialPairingOptions(overlap=8), device=device)
        model = map_views()
        if model is None or model.num_reg_images() < count * 0.7:
            report('matching', 'Looking for additional overlapping views…')
            pycolmap.match_exhaustive(database, matching, device=device)
            expanded = map_views()
            if expanded is not None and (model is None or expanded.num_reg_images() > model.num_reg_images()):
                model = expanded
    else:
        pycolmap.match_exhaustive(database, matching, device=device)
        model = map_views()

    registered = model.num_reg_images() if model is not None else 0
    if model is None or registered < max(MIN_VIEWS, math.ceil(count * 0.5)):
        raise ValueError(f'COLMAP aligned {registered} of {count} images into one scene. Capture more overlapping '
                         'views while physically moving the camera, and keep the room still and well lit.')
    return model


def orient_model(model: pycolmap.Reconstruction) -> dict:
    """Level the scene and choose a well-supported camera near the start of the capture."""
    cameras = sorted((model.image(image_id) for image_id in model.reg_image_ids()), key=lambda image: image.name)
    poses = [image.cam_from_world().inverse() for image in cameras]
    rotations = np.array([pose.rotation.matrix() for pose in poses])
    start = poses[0].translation
    # COLMAP's camera axes point right, down and forward. The PLY keeps this
    # leveled world frame. SuperSplat flips PLY x and y when it loads the asset.
    down = np.median(rotations[:, :, 1], axis=0)
    if np.linalg.norm(down) < 0.1:
        down = rotations[0, :, 1]
    down /= np.linalg.norm(down)
    forward = rotations[0, :, 2] - down * np.dot(rotations[0, :, 2], down)
    if np.linalg.norm(forward) < 0.1:
        forward = np.cross(rotations[0, :, 0], down)
    forward /= np.linalg.norm(forward)
    rotation = np.stack((np.cross(down, forward), down, forward))
    distances = np.linalg.norm(np.array([pose.translation for pose in poses]) - start, axis=1)
    scale = 3.0 / max(float(np.percentile(distances, 90)), 1e-6)

    # Brush needs a spatially spread point cloud to initialize splat sizes.
    points = np.array([point.xyz for point in model.points3D.values()])
    if len(points):
        low, high = np.percentile(points, (5, 95), axis=0)
        extent = float(np.max(high - low)) * scale
    else:
        extent = 0.0
    if not math.isfinite(extent) or extent < 1e-3:
        raise ValueError('Camera alignment produced almost no usable depth. Move through the room while keeping '
                         'consecutive views overlapping.')

    model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), -scale * (rotation @ start)))

    candidates = range(max(1, len(cameras) // 8))
    index = max(candidates, key=lambda index: cameras[index].num_points3D)
    viewer_axes = np.array([-1.0, -1.0, 1.0])
    position = scale * (rotation @ (poses[index].translation - start)) * viewer_axes
    direction = (rotation @ rotations[index, :, 2]) * viewer_axes
    camera = model.camera(cameras[index].camera_id)
    focal_length = camera.calibration_matrix()[0, 0]
    fov = math.degrees(2 * math.atan(camera.width / (2 * focal_length)))
    return {'position': position.tolist(), 'target': (position + direction).tolist(), 'fov': min(90, max(30, fov))}


def prepare_training_data(model: pycolmap.Reconstruction, images: Path,
                          dataset: Path) -> tuple[Path, pycolmap.Reconstruction]:
    # Brush uses pinhole cameras and discovers COLMAP files recursively. Export
    # undistorted images and only the chosen model into a separate directory.
    sparse = dataset / 'registered'
    sparse.mkdir()
    model.write(sparse)
    training = dataset / 'undistorted'
    training.mkdir()
    options = pycolmap.UndistortCameraOptions(max_image_size=TRAINING_IMAGE_SIZE)
    pycolmap.undistort_images(training, sparse, images, undistort_options=options,
                             jpeg_quality=94, num_threads=COLMAP_THREADS)

    undistorted = pycolmap.Reconstruction(training / 'sparse')
    invalid_cameras = {
        camera_id for camera_id, camera in undistorted.cameras.items()
        if not all(math.isfinite(focal) and focal > 0 for focal in (camera.focal_length_x, camera.focal_length_y))
    }
    invalid_frames = {image.frame_id for image in undistorted.images.values() if image.camera_id in invalid_cameras}
    for frame_id in invalid_frames:
        undistorted.deregister_frame(frame_id)

    # Brush 0.3.0 reads a fixed-size header chunk. JPEG readers ignore padding after the end marker.
    for photo in (training / 'images').glob('*.jpg'):
        missing = BRUSH_HEADER_BYTES - photo.stat().st_size
        if missing > 0:
            with photo.open('ab') as stream:
                stream.write(bytes(missing))
    return training, undistorted


def train_splat(brush: str, training_input: Path, output: Path, steps: int, registered: int, count: int,
                report: ProgressCallback) -> None:
    report('training', f'Training the 3D scene from {registered} of {count} images…')
    command = [
        brush, str(training_input.resolve()),
        '--total-steps', str(steps), '--export-every', str(steps),
        '--growth-stop-iter', str(int(steps * 0.85)), '--max-resolution', str(TRAINING_IMAGE_SIZE),
        '--refine-every', '150', '--growth-select-fraction', '0.16', '--max-splats', '600000', '--sh-degree', '2',
        '--export-path', str(output.resolve()), '--export-name', 'scene.ply',
    ]
    log_path = training_input.parent / 'brush.log'
    TOOLS.mkdir(exist_ok=True)
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=TOOLS, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not (output / 'scene.ply').is_file():
        details = log_path.read_text(encoding='utf-8', errors='replace')[-3000:]
        LOG.error('Brush log:\n%s', details)
        if 'No possible adapter available' in details:
            raise ValueError('Brush could not access the GPU. Start the server in a terminal with GPU access.')
        raise ValueError('Brush could not train the scene. Check the terminal for details.')


def reconstruct(paths: list[Path], dataset: Path, output: Path, steps: int,
                report: ProgressCallback) -> tuple[dict, int, int]:
    kind = input_kind(paths)
    brush = brush_binary()
    dataset.mkdir()
    images = dataset / 'images'
    report('preparing', 'Extracting video frames…' if kind == 'video' else 'Preparing photos…')
    count = prepare_images(paths, kind, images)
    model = align_images(dataset, images, kind, count, report)
    report('preparing', 'Correcting lens distortion in the aligned images…')
    training_input, model = prepare_training_data(model, images, dataset)
    registered = model.num_reg_images()
    if registered < max(MIN_VIEWS, math.ceil(count * 0.5)):
        raise ValueError(f'Only {registered} of {count} views had usable camera calibration. '
                         'Try capturing more overlapping views with the same lens and zoom.')
    camera = orient_model(model)
    model.write(training_input / 'sparse')
    output.mkdir()
    train_splat(brush, training_input, output, steps, registered, count, report)
    return camera, registered, count
