/**
 * Attendance PWA — Face recognition entirely in the browser.
 * Uses face-api.js + IndexedDB. No server needed.
 */

// ─── Config ──────────────────────────────────────────────
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1/model';
const MATCH_THRESHOLD = 0.55; // Lower = stricter
const DB_NAME = 'AsistenciaDB';
const DB_VER = 1;

// ─── State ───────────────────────────────────────────────
let db = null;
let modelsLoaded = false;
let camStream = null;
let regStream = null;
let regPhotoData = null;
let lastResults = null; // last attendance results

// ─── Init ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Register service worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }

  // Open DB
  db = await openDB();

  // Load face-api models
  try {
    updateLoader('Descargando modelos de IA...');
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ]);
    modelsLoaded = true;
    document.getElementById('statusBadge').textContent = 'Listo';
    document.getElementById('statusBadge').classList.add('ready');
  } catch (e) {
    updateLoader('Error cargando modelos. Verifique su conexion.');
    console.error(e);
    return;
  }
  hideLoader();

  // Bind navigation
  document.querySelectorAll('.nav__btn').forEach(btn => {
    btn.onclick = () => switchTab(btn.dataset.tab);
  });

  // Bind camera placeholder click
  document.getElementById('camPlaceholder').onclick = startMainCamera;

  // Bind attendance
  document.getElementById('btnTakeAttendance').onclick = takeAttendance;
  document.getElementById('btnSaveSession').onclick = saveCurrentSession;
  document.getElementById('btnExportResult').onclick = () => exportResults(lastResults);

  // Bind registration
  document.getElementById('btnRegCapture').onclick = captureRegPhoto;
  document.getElementById('btnRegRetake').onclick = retakeRegPhoto;
  document.getElementById('btnRegSave').onclick = registerStudent;

  // Bind history
  document.getElementById('btnExportAll').onclick = exportAllSessions;

  // Load student list
  refreshStudentList();
  refreshHistory();
});

// ─── IndexedDB ───────────────────────────────────────────
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VER);
    req.onupgradeneeded = e => {
      const d = e.target.result;
      if (!d.objectStoreNames.contains('students'))
        d.createObjectStore('students', { keyPath: 'id', autoIncrement: true });
      if (!d.objectStoreNames.contains('sessions'))
        d.createObjectStore('sessions', { keyPath: 'id', autoIncrement: true });
    };
    req.onsuccess = e => resolve(e.target.result);
    req.onerror = e => reject(e.target.error);
  });
}

function dbTx(store, mode = 'readonly') {
  const tx = db.transaction(store, mode);
  return tx.objectStore(store);
}

function dbGetAll(store) {
  return new Promise((resolve, reject) => {
    const req = dbTx(store).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function dbAdd(store, data) {
  return new Promise((resolve, reject) => {
    const req = dbTx(store, 'readwrite').add(data);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function dbDelete(store, id) {
  return new Promise((resolve, reject) => {
    const req = dbTx(store, 'readwrite').delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// ─── Tab Navigation ──────────────────────────────────────
function switchTab(viewId) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav__btn').forEach(b => b.classList.remove('active'));
  document.getElementById(viewId).classList.add('active');
  document.querySelector(`[data-tab="${viewId}"]`).classList.add('active');

  // Start/stop cameras as needed
  if (viewId === 'viewStudents') {
    startRegCamera();
  } else {
    stopRegCamera();
  }
  if (viewId === 'viewHistory') refreshHistory();
}

// ─── Main Camera ─────────────────────────────────────────
async function startMainCamera() {
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false
    });
    const video = document.getElementById('camVideo');
    video.srcObject = camStream;
    await video.play();
    document.getElementById('camPlaceholder').style.display = 'none';
    document.getElementById('btnTakeAttendance').disabled = false;
    toast('Camara lista', 'ok');
  } catch (e) {
    toast('No se pudo acceder a la camara: ' + e.message, 'err');
  }
}

// ─── Take Attendance ─────────────────────────────────────
async function takeAttendance() {
  if (!modelsLoaded) return toast('Modelos no cargados', 'err');

  const students = await dbGetAll('students');
  if (!students.length) return toast('Registre estudiantes primero', 'err');

  const btn = document.getElementById('btnTakeAttendance');
  btn.disabled = true;
  btn.textContent = '🔍 Analizando rostros...';

  try {
    const video = document.getElementById('camVideo');
    const canvas = document.getElementById('camCanvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Detect all faces
    const detections = await faceapi
      .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.4 }))
      .withFaceLandmarks(true)
      .withFaceDescriptors();

    // Build matcher from stored students
    const labeled = students
      .filter(s => s.descriptor)
      .map(s => new faceapi.LabeledFaceDescriptors(
        String(s.id),
        [new Float32Array(s.descriptor)]
      ));

    if (!labeled.length) {
      toast('No hay descriptores faciales guardados', 'err');
      btn.disabled = false;
      btn.textContent = '📸 Tomar Asistencia';
      return;
    }

    const matcher = new faceapi.FaceMatcher(labeled, MATCH_THRESHOLD);

    // Match each detected face
    const matched = [];
    const matchedIds = new Set();

    for (const det of detections) {
      const best = matcher.findBestMatch(det.descriptor);
      if (best.label !== 'unknown') {
        const sid = parseInt(best.label);
        if (!matchedIds.has(sid)) {
          matchedIds.add(sid);
          const student = students.find(s => s.id === sid);
          matched.push({
            studentId: sid,
            name: student ? student.name : 'ID ' + sid,
            code: student ? student.code : '',
            confidence: +(1 - best.distance).toFixed(3),
            box: det.detection.box,
          });
        }
      }
    }

    // Draw overlays
    drawResults(ctx, detections, matched, students);

    // Build full results (present + absent)
    const present = matched;
    const absentStudents = students.filter(s => !matchedIds.has(s.id));
    const results = {
      date: new Date().toISOString(),
      present,
      absent: absentStudents.map(s => ({ studentId: s.id, name: s.name, code: s.code })),
      totalDetected: detections.length,
    };

    lastResults = results;
    showResults(results, students.length);
    toast(`Detectados ${detections.length} rostros, ${present.length} reconocidos`, 'ok');

  } catch (e) {
    console.error(e);
    toast('Error al procesar: ' + e.message, 'err');
  }

  btn.disabled = false;
  btn.textContent = '📸 Tomar Asistencia';
}

function drawResults(ctx, detections, matched, students) {
  const matchedIds = new Set(matched.map(m => m.studentId));

  for (const det of detections) {
    const box = det.detection.box;
    const bestLabel = matched.find(m => {
      const b = m.box;
      return Math.abs(b.x - box.x) < 5 && Math.abs(b.y - box.y) < 5;
    });

    const known = !!bestLabel;
    const color = known ? '#22c55e' : '#ef4444';

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Label
    const label = known ? bestLabel.name : 'Desconocido';
    ctx.font = 'bold 16px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = known ? 'rgba(34,197,94,.8)' : 'rgba(239,68,68,.8)';
    ctx.fillRect(box.x, box.y - 26, tw + 12, 24);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, box.x + 6, box.y - 8);
  }
}

function showResults(results, totalStudents) {
  document.getElementById('resultsSection').style.display = 'flex';
  document.getElementById('resultsSection').style.flexDirection = 'column';
  document.getElementById('resultsSection').style.gap = '10px';

  const p = results.present.length;
  const a = results.absent.length;
  const pct = totalStudents ? Math.round(p / totalStudents * 100) : 0;

  document.getElementById('rPresent').textContent = p;
  document.getElementById('rAbsent').textContent = a;
  document.getElementById('rTotal').textContent = totalStudents;
  document.getElementById('rProgress').style.width = pct + '%';
  document.getElementById('resultCount').textContent = `${pct}% asistencia`;

  const list = document.getElementById('resultsList');
  list.innerHTML = '';

  // Present
  results.present.forEach(r => {
    list.innerHTML += `
      <div class="result-item result-item--present">
        <div class="student-item__avatar">${initials(r.name)}</div>
        <div class="student-item__info">
          <div class="student-item__name">${r.name}</div>
          <div class="student-item__code">${r.code} - ${(r.confidence * 100).toFixed(0)}% confianza</div>
        </div>
        <div class="result-item__badge">Presente</div>
      </div>`;
  });

  // Absent
  results.absent.forEach(r => {
    list.innerHTML += `
      <div class="result-item result-item--absent">
        <div class="student-item__avatar" style="background:linear-gradient(135deg,var(--red),#dc2626)">${initials(r.name)}</div>
        <div class="student-item__info">
          <div class="student-item__name">${r.name}</div>
          <div class="student-item__code">${r.code}</div>
        </div>
        <div class="result-item__badge">Ausente</div>
      </div>`;
  });
}

async function saveCurrentSession() {
  if (!lastResults) return;
  await dbAdd('sessions', {
    date: lastResults.date,
    present: lastResults.present,
    absent: lastResults.absent,
    totalDetected: lastResults.totalDetected,
  });
  toast('Sesion guardada', 'ok');
  refreshHistory();
}

// ─── Registration Camera ─────────────────────────────────
async function startRegCamera() {
  try {
    regStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 480 } },
      audio: false
    });
    const video = document.getElementById('regVideo');
    video.srcObject = regStream;
    await video.play();
    document.getElementById('regPlaceholder').style.display = 'none';
    document.getElementById('regVideo').style.display = 'block';
    document.getElementById('regPreview').style.display = 'none';
    document.getElementById('btnRegCapture').style.display = 'inline-flex';
    document.getElementById('btnRegRetake').style.display = 'none';
  } catch (e) {
    toast('Camara no disponible', 'err');
  }
}

function stopRegCamera() {
  if (regStream) {
    regStream.getTracks().forEach(t => t.stop());
    regStream = null;
  }
}

function captureRegPhoto() {
  const video = document.getElementById('regVideo');
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  regPhotoData = c.toDataURL('image/jpeg', 0.85);

  document.getElementById('regPreview').src = regPhotoData;
  document.getElementById('regPreview').style.display = 'block';
  document.getElementById('regVideo').style.display = 'none';
  document.getElementById('btnRegCapture').style.display = 'none';
  document.getElementById('btnRegRetake').style.display = 'inline-flex';
  document.getElementById('btnRegSave').disabled = false;

  stopRegCamera();
}

function retakeRegPhoto() {
  regPhotoData = null;
  document.getElementById('btnRegSave').disabled = true;
  startRegCamera();
}

async function registerStudent() {
  const name = document.getElementById('regName').value.trim();
  const code = document.getElementById('regCode').value.trim();
  if (!name || !code) return toast('Complete nombre y codigo', 'err');
  if (!regPhotoData) return toast('Capture una foto', 'err');

  const btn = document.getElementById('btnRegSave');
  btn.disabled = true;
  btn.textContent = 'Procesando...';

  try {
    // Load image into element for face-api
    const img = await loadImage(regPhotoData);

    const det = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.3 }))
      .withFaceLandmarks(true)
      .withFaceDescriptor();

    if (!det) {
      toast('No se detecto un rostro claro. Intente de nuevo.', 'err');
      btn.disabled = false;
      btn.textContent = 'Registrar Estudiante';
      return;
    }

    const descriptor = Array.from(det.descriptor);

    await dbAdd('students', {
      name, code, descriptor,
      photo: regPhotoData,
      createdAt: new Date().toISOString(),
    });

    toast(`${name} registrado exitosamente`, 'ok');
    document.getElementById('regName').value = '';
    document.getElementById('regCode').value = '';
    regPhotoData = null;
    refreshStudentList();
    retakeRegPhoto();

  } catch (e) {
    toast('Error: ' + e.message, 'err');
  }

  btn.disabled = false;
  btn.textContent = '✅ Registrar Estudiante';
}

// ─── Student List ────────────────────────────────────────
async function refreshStudentList() {
  const students = await dbGetAll('students');
  const container = document.getElementById('studentList');
  const countLabel = document.getElementById('studentCountLabel');
  countLabel.textContent = students.length;

  // Update header badge
  const badge = document.getElementById('statusBadge');
  if (modelsLoaded) {
    badge.textContent = students.length + ' estudiantes';
    badge.classList.add('ready');
  }

  if (!students.length) {
    container.innerHTML = '<div class="empty"><div class="empty__icon">👤</div>No hay estudiantes</div>';
    return;
  }

  container.innerHTML = students.map(s => `
    <div class="student-item" id="stu-${s.id}">
      ${s.photo
        ? `<img class="student-item__photo" src="${s.photo}" alt="${s.name}">`
        : `<div class="student-item__avatar">${initials(s.name)}</div>`}
      <div class="student-item__info">
        <div class="student-item__name">${s.name}</div>
        <div class="student-item__code">${s.code}</div>
      </div>
      <button class="btn btn--danger btn--icon" onclick="deleteStudent(${s.id})" title="Eliminar">
        🗑️
      </button>
    </div>`).join('');
}

async function deleteStudent(id) {
  if (!confirm('Eliminar este estudiante?')) return;
  await dbDelete('students', id);
  toast('Estudiante eliminado', 'info');
  refreshStudentList();
}

// ─── History ─────────────────────────────────────────────
async function refreshHistory() {
  const sessions = await dbGetAll('sessions');
  const container = document.getElementById('historyList');

  if (!sessions.length) {
    container.innerHTML = '<div class="empty"><div class="empty__icon">📋</div>No hay sesiones</div>';
    return;
  }

  container.innerHTML = sessions.reverse().map(s => {
    const d = new Date(s.date);
    const dateStr = d.toLocaleDateString('es-CO', { day: '2-digit', month: 'short', year: 'numeric' });
    const timeStr = d.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' });
    const total = s.present.length + s.absent.length;
    return `
      <div class="session-card">
        <div class="session-card__info">
          <div class="session-card__date">${dateStr} - ${timeStr}</div>
          <div class="session-card__meta">${s.present.length} presentes, ${s.absent.length} ausentes</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="session-card__count">${s.present.length}/${total}</div>
          <button class="btn" onclick='exportSession(${s.id})' style="padding:6px 10px;font-size:.7rem">📥</button>
          <button class="btn btn--danger btn--icon" onclick="deleteSession(${s.id})" style="width:32px;height:32px;font-size:.7rem">🗑️</button>
        </div>
      </div>`;
  }).join('');
}

async function deleteSession(id) {
  if (!confirm('Eliminar esta sesion?')) return;
  await dbDelete('sessions', id);
  refreshHistory();
  toast('Sesion eliminada', 'info');
}

// ─── Excel Export ────────────────────────────────────────
function exportResults(results) {
  if (!results) return toast('No hay resultados', 'err');

  const rows = [];
  results.present.forEach(r => {
    rows.push({
      Nombre: r.name, Codigo: r.code, Estado: 'Presente',
      Confianza: (r.confidence * 100).toFixed(1) + '%',
      Fecha: new Date(results.date).toLocaleString('es-CO'),
    });
  });
  results.absent.forEach(r => {
    rows.push({
      Nombre: r.name, Codigo: r.code, Estado: 'Ausente',
      Confianza: '-',
      Fecha: new Date(results.date).toLocaleString('es-CO'),
    });
  });

  downloadExcel(rows, 'asistencia');
}

async function exportSession(sessionId) {
  const sessions = await dbGetAll('sessions');
  const s = sessions.find(x => x.id === sessionId);
  if (!s) return;
  exportResults({ ...s, present: s.present, absent: s.absent });
}

async function exportAllSessions() {
  const sessions = await dbGetAll('sessions');
  if (!sessions.length) return toast('No hay sesiones', 'err');

  const rows = [];
  sessions.forEach(s => {
    const d = new Date(s.date).toLocaleString('es-CO');
    s.present.forEach(r => {
      rows.push({ Fecha: d, Nombre: r.name, Codigo: r.code, Estado: 'Presente', Confianza: (r.confidence * 100).toFixed(1) + '%' });
    });
    s.absent.forEach(r => {
      rows.push({ Fecha: d, Nombre: r.name, Codigo: r.code, Estado: 'Ausente', Confianza: '-' });
    });
  });
  downloadExcel(rows, 'historial_asistencia');
}

function downloadExcel(rows, name) {
  const ws = XLSX.utils.json_to_sheet(rows);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Asistencia');

  // Column widths
  ws['!cols'] = [{ wch: 28 }, { wch: 12 }, { wch: 12 }, { wch: 10 }, { wch: 22 }];

  XLSX.writeFile(wb, `${name}_${new Date().toISOString().slice(0, 10)}.xlsx`);
  toast('Excel descargado', 'ok');
}

// ─── Utilities ───────────────────────────────────────────
function initials(name) {
  return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function toast(msg, type = 'info') {
  const box = document.getElementById('toastBox');
  const icons = { ok: '✅', err: '❌', info: 'ℹ️' };
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${msg}</span>`;
  box.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3500);
}

function updateLoader(text) {
  document.getElementById('loaderText').textContent = text;
}
function hideLoader() {
  document.getElementById('loader').classList.add('hidden');
}
