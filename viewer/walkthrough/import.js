const form = document.querySelector('#import-form');
const submit = form.querySelector('button');
const feedback = document.querySelector('#feedback');
const list = document.querySelector('#scenes');
const libraryStatus = document.querySelector('#library-status');
let renderedScenes = '';
let refreshTimer;
const isActive = ({ state }) => state !== 'ready' && state !== 'failed';

function sceneRow(scene) {
  const row = document.createElement('li');
  const details = document.createElement('div');
  const name = document.createElement('strong');
  name.textContent = scene.title;
  const status = document.createElement('small');
  status.className = scene.state === 'failed' ? 'error' : 'hint';
  status.textContent = scene.message;
  details.append(name, status);
  row.append(details);

  if (scene.state === 'ready') {
    const link = document.createElement('a');
    link.className = 'open';
    link.href = scene.url;
    link.textContent = 'Open walkthrough';
    link.setAttribute('aria-label', `Open walkthrough for ${scene.title}`);
    row.append(link);
  } else if (isActive(scene)) {
    const progress = document.createElement('progress');
    progress.setAttribute('aria-label', `Processing ${scene.title}`);
    details.append(progress);
  }
  return row;
}

async function refreshScenes() {
  clearTimeout(refreshTimer);
  let retry = true;
  try {
    const response = await fetch('/api/scenes', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const scenes = await response.json();
    const snapshot = JSON.stringify(scenes);
    if (snapshot !== renderedScenes) {
      list.replaceChildren(...scenes.map(sceneRow));
      renderedScenes = snapshot;
    }
    libraryStatus.textContent = scenes.length ? '' : 'Ready walkthroughs will appear here.';
    retry = scenes.some(isActive);
  } catch {
    libraryStatus.textContent = 'Could not load scenes. Reconnecting…';
  } finally {
    clearTimeout(refreshTimer);
    if (retry) refreshTimer = setTimeout(refreshScenes, 1500);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submit.disabled = true;
  feedback.className = 'hint';
  feedback.textContent = 'Uploading capture…';
  try {
    const response = await fetch('/api/scenes', { method: 'POST', body: new FormData(form) });
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      throw new Error(result.error || `The upload failed (HTTP ${response.status}).`);
    }
    form.reset();
    feedback.textContent = 'Capture added.';
    await refreshScenes();
  } catch (error) {
    feedback.className = 'error';
    feedback.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});

refreshScenes();
