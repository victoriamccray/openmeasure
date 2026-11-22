// Become the Signal: EEG, from signal to inference.
//
// Two files drive everything here. content.json holds the words, with a
// citation key on every claim taken from a source. data.json holds the
// numbers, every one computed by modules/signal_pipeline/core from a recorded
// public EEG recording (scripts/build_become_the_signal_data.py). This
// file draws them and computes nothing about the signal itself, so the
// 3D scene, the panel, and the text view can never disagree.

import * as THREE from "three";

const [content, data] = await Promise.all([
  fetch("content.json").then((r) => r.json()),
  fetch("data.json").then((r) => r.json()),
]);

const STAGES = content.stages;
if (STAGES.map((s) => s.name).join() !== data.stages.join()) {
  throw new Error("content.json stages do not match the pipeline stages in data.json.");
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const narrow = window.matchMedia("(max-width: 760px)");

const COLORS = {
  closed: 0xff9f6b,
  open: 0x7cc4ff,
  wrong: 0xff5d7a,
  copy: 0xb9f5ff,
  path: 0x5a6bd8,
};

// Each stage's world, as fog and light colours. One hue per world so a
// learner can tell where they are without reading.
const WORLD = [
  { fog: 0x1a0612, light: 0xff6f91 },
  { fog: 0x061417, light: 0x6fe3d0 },
  { fog: 0x04101c, light: 0x5ad1ff },
  { fog: 0x0b0820, light: 0x9a8cff },
  { fog: 0x170f04, light: 0xffc46b },
  { fog: 0x04140b, light: 0x7ef0a0 },
  { fog: 0x100619, light: 0xd08cff },
];

const DECISION = STAGES.find((s) => s.interaction === "safeguard");

const state = {
  stage: 0,
  visited: new Set([0]),
  perspective: "agency",
  eyes: "eyes_closed",
  pass: 1,
  // One entry per pass: whether the device recorded, and the safeguard
  // chosen at the junction with its level. Earlier passes are never
  // cleared, so their copies stay on the map.
  passes: { 1: { device: false, safeguard: null, level: null } },
  textView: false,
  // Neuroanatomy figure: the sensor shown, the region tapped, and the
  // electrode read on the 10-20 map.
  modality: "eeg",
  anatomySystem: "occipital",
  electrode: "O1",
  // Study room: the open tab and the dichotomy axis shown.
  studyTab: "room",
  studyAxis: "state",
};

// ------------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const esc = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (value) => (value === null || value === undefined ? "fails" : `${Math.round(value * 100)}%`);

// The anatomy figure cites from data.json (core/neuroanatomy.py); the rest
// of the journey cites from content.json. A key in both has the same text,
// which shared/tests/test_become_the_signal.py checks.
const REFS = {
  ...Object.fromEntries(data.resting_state.references.map((r) => [r.key, r])),
  ...Object.fromEntries(data.neuroanatomy.references.map((r) => [r.key, r])),
  ...content.references,
};
function anatomyRefs(stage) {
  const a = stage.anatomy;
  if (!a) return [];
  const anat = data.neuroanatomy;
  if (a.view === "electrode") return [...anat.recording_electrode.refs, ...anat.electrode_caveat_refs];
  return [
    ...anat.systems.flatMap((s) => s.refs),
    ...anat.modalities.flatMap((m) => m.refs),
    ...anat.caveat_refs,
  ];
}

// References are numbered by first appearance along the journey, so the
// numbers read in order as the learner moves forward.
const refOrder = [];
for (const stage of STAGES) {
  const keys = [
    ...(stage.refs || []),
    ...(stage.mechanism_refs || []),
    ...(stage.threat?.refs || []),
    ...(stage.threat ? content.about_data.refs : []),
    ...(stage.second_inference?.refs || []),
    ...(stage.future_inference?.refs || []),
    ...anatomyRefs(stage),
  ];
  for (const key of keys) if (!refOrder.includes(key)) refOrder.push(key);
}
// The study room's sources, numbered after the journey's own.
for (const r of data.resting_state.references) if (!refOrder.includes(r.key)) refOrder.push(r.key);
const refNumber = (key) => refOrder.indexOf(key) + 1;
const refMarks = (keys = []) =>
  keys.map((k) => `<button type="button" class="ref" data-ref="${k}" aria-label="Reference ${refNumber(k)}">[${refNumber(k)}]</button>`).join("");

function noiseEntry(sd) {
  return data.noise.find((n) => n.sd_uv === sd);
}
function downsampleEntry(factor) {
  return data.downsample.find((d) => d.factor === factor);
}

const current = () => state.passes[state.pass];

// What each copy supports, for a pass's safeguard. Noise and a lower rate
// change the stored recording, so the device's copy and the legitimate
// system change with it; the other safeguards leave both untouched.
function outcomeFor(record) {
  const base = data.baseline;
  switch (record.safeguard) {
    case "noise": {
      const e = noiseEntry(record.level);
      return { requester: e, legitimate: e };
    }
    case "downsample": {
      const e = downsampleEntry(record.level);
      return { requester: e, legitimate: e };
    }
    case "metadata":
    case "purpose":
      return { requester: base, legitimate: base };
    case "deny":
      return { requester: null, legitimate: base };
    default:
      return { requester: null, legitimate: base };
  }
}
const safeguardOutcome = () => (current().safeguard ? outcomeFor(current()) : null);

// Every copy of the learner, with who holds it, what it still supports,
// and whether the person can take it back. Concrete facts, not a score.
function copies() {
  const list = [];
  for (const [pass, record] of Object.entries(state.passes)) {
    const outcome = outcomeFor(record);
    if (record.device) {
      list.push({
        pass: Number(pass),
        where: "Sensors",
        label: STAGES[1].copy,
        holder: "The recording device",
        accuracy: outcome.legitimate.accuracy,
        labels: true,
        recall: "Deletable by whoever runs the device",
      });
    }
    if (record.safeguard && record.safeguard !== "deny") {
      list.push({
        pass: Number(pass),
        where: "Decision",
        label: DECISION.safeguards[record.safeguard].copy,
        holder: "An outside organization",
        accuracy: outcome.requester.accuracy,
        labels: record.safeguard !== "metadata",
        recall: "Stays with the receiver",
      });
    }
  }
  return list;
}

function misclassified(result) {
  const t = result.threshold;
  return {
    closedAsOpen: result.alpha_power_eyes_closed.filter((p) => p <= t).length,
    openAsClosed: result.alpha_power_eyes_open.filter((p) => p > t).length,
    total: result.alpha_power_eyes_closed.length + result.alpha_power_eyes_open.length,
  };
}

// ------------------------------------------------------------------
// 3D scene
// ------------------------------------------------------------------

const canvas = $("scene");
let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
} catch (error) {
  renderer = null;
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 400);
scene.fog = new THREE.FogExp2(WORLD[0].fog, 0.028);
scene.background = new THREE.Color(WORLD[0].fog);

const ambient = new THREE.AmbientLight(0xffffff, 0.35);
const key = new THREE.PointLight(WORLD[0].light, 60, 60, 1.6);
const fill = new THREE.DirectionalLight(0xffffff, 0.6);
fill.position.set(5, 10, 8);
scene.add(ambient, key, fill);

const stationPos = (i) => new THREE.Vector3(Math.sin(i * 0.9) * 9, 0, -i * 60);

function glow(color, intensity = 0.8) {
  return new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: intensity, roughness: 0.4 });
}
function matte(color) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.75, metalness: 0.1 });
}
function line(points, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
}

// A trace drawn from recorded samples, centred on the origin.
function traceLine(values, width, height, color, maxAbs) {
  const n = values.length;
  const points = values.map((v, i) => new THREE.Vector3((i / (n - 1) - 0.5) * width, (v / maxAbs) * height, 0));
  return line(points, color);
}

// A row of pillars from measured alpha powers. Heights are scaled to the
// largest value in the set, which the panel states.
function pillarSet(result, { spacing = 0.55, height = 6, radius = 0.18, showWrong = true } = {}) {
  const group = new THREE.Group();
  const all = [...result.alpha_power_eyes_closed, ...result.alpha_power_eyes_open];
  const max = Math.max(...all);
  const rows = [
    { values: result.alpha_power_eyes_closed, z: -1.2, truth: "closed" },
    { values: result.alpha_power_eyes_open, z: 1.2, truth: "open" },
  ];
  for (const row of rows) {
    row.values.forEach((power, i) => {
      const h = Math.max((power / max) * height, 0.04);
      const predicted = power > result.threshold ? "closed" : "open";
      const wrong = predicted !== row.truth;
      const pillar = new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius, h, 12),
        glow(predicted === "closed" ? COLORS.closed : COLORS.open, wrong ? 0.25 : 0.6),
      );
      pillar.position.set((i - (row.values.length - 1) / 2) * spacing, h / 2, row.z);
      pillar.userData.inspect =
        `Second ${i + 1}, eyes ${row.truth}. Alpha power ${power.toFixed(1)} µV². ` +
        `Labelled eyes ${predicted}${wrong ? ", which is wrong" : ""}.`;
      group.add(pillar);
      if (wrong && showWrong) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(radius * 2.2, 0.05, 8, 24), glow(COLORS.wrong, 1.2));
        ring.rotation.x = Math.PI / 2;
        ring.position.set(pillar.position.x, 0.06, row.z);
        group.add(ring);
      }
    });
  }
  const width = (Math.max(rows[0].values.length, rows[1].values.length) + 1) * spacing;
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(width, 4.4),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }),
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = (result.threshold / max) * height;
  group.add(plane);
  const edge = line(
    [new THREE.Vector3(-width / 2, plane.position.y, -2.2), new THREE.Vector3(width / 2, plane.position.y, -2.2)],
    0xffffff,
    0.7,
  );
  group.add(edge);
  return group;
}

// A short label in the world, drawn once to a canvas.
function labelSprite(text, color = "#e8ebf7") {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 96;
  const ctx = c.getContext("2d");
  ctx.font = "600 52px system-ui, sans-serif";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 256, 48);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  sprite.scale.set(5.2, 0.98, 1);
  return sprite;
}

// A fingerprint mark: the identifying information carried by the signal
// itself, which travels with every copy whatever happens to its labels.
const fingerprintTexture = (() => {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d");
  ctx.strokeStyle = "#ffcf6b";
  ctx.lineWidth = 5;
  for (let r = 10; r <= 56; r += 11) {
    ctx.beginPath();
    ctx.ellipse(64, 70, r * 0.78, r, 0, Math.PI * 0.95, Math.PI * 2.05);
    ctx.stroke();
  }
  return new THREE.CanvasTexture(c);
})();
function fingerprint() {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: fingerprintTexture, transparent: true, depthWrite: false }));
  sprite.scale.set(0.9, 0.9, 1);
  sprite.userData.inspect = "The signal itself can identify the person. This travels with every copy.";
  return sprite;
}

// A tall column of light over a copy, so copies stay visible from far
// along the pathway. Thin up close, broad in the overview.
const beacons = new Set();
function beacon() {
  const geometry = new THREE.CylinderGeometry(0.18, 0.18, 40, 12, 1, true);
  geometry.translate(0, 20, 0);
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      color: COLORS.copy,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    }),
  );
  beacons.add(mesh);
  return mesh;
}

// An inference landscape, or a flattened one when the inference fails.
function landscape(result, options) {
  if (result.accuracy !== null) return pillarSet(result, options);
  const group = new THREE.Group();
  const flat = new THREE.MeshStandardMaterial({ color: 0x555a66, roughness: 0.9 });
  for (const z of [-1.2, 1.2]) {
    for (let i = 0; i < 15; i++) {
      const disc = new THREE.Mesh(new THREE.CylinderGeometry(options.radius ?? 0.18, options.radius ?? 0.18, 0.05, 12), flat);
      disc.position.set((i - 7) * options.spacing, 0.03, z);
      disc.userData.inspect = "The inference fails here: the signal no longer contains the alpha band.";
      group.add(disc);
    }
  }
  return group;
}

function ghost(scale = 1) {
  const mesh = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.55 * scale, 1),
    new THREE.MeshBasicMaterial({ color: COLORS.copy, wireframe: true, transparent: true, opacity: 0.75 }),
  );
  return mesh;
}

// Explanatory motion. Each entry animates one stage's transformation and
// runs only while that stage is on screen. Under reduced motion every
// entry is drawn once, at its settled state.
const animations = [];
const animate = (stage, update) => animations.push({ stage, update });
let arrivedAt = 0;
let settledStage = -1;
let copyFlightStart = -Infinity;

const stations = STAGES.map((_, i) => {
  const group = new THREE.Group();
  group.position.copy(stationPos(i));
  scene.add(group);
  return group;
});

// Measured average alpha power per condition, from the same windows the
// Inference stage classifies.
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const ALPHA_MEAN = {
  eyes_closed: mean(data.baseline.alpha_power_eyes_closed),
  eyes_open: mean(data.baseline.alpha_power_eyes_open),
};

const waveMax = Math.max(
  ...data.waveform.eyes_closed.map(Math.abs),
  ...data.waveform.eyes_open.map(Math.abs),
);

function seeded(seed) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) / 2147483647);
}

// 1. Signals: neural tissue, with the recorded trace running through it.
(function buildSignals() {
  const g = stations[0];
  const rand = seeded(7);
  const count = 160;
  const cells = new THREE.InstancedMesh(new THREE.SphereGeometry(0.22, 12, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }), count);
  // Each neuron's place in the cycle, from -0.5 to 0.5 of a cycle. How far
  // these spread is what separates in-step from out-of-step firing.
  const offsets = Array.from({ length: count }, () => rand() - 0.5);
  const m = new THREE.Matrix4();
  const positions = [];
  for (let i = 0; i < count; i++) {
    const p = new THREE.Vector3((rand() - 0.5) * 34, (rand() - 0.2) * 14, (rand() - 0.5) * 26 - 6);
    positions.push(p);
    const s = 0.5 + rand() * 1.4;
    m.compose(p, new THREE.Quaternion(), new THREE.Vector3(s, s, s));
    cells.setMatrixAt(i, m);
    cells.setColorAt(i, new THREE.Color(0xff6f91));
  }
  g.add(cells);
  const links = [];
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      if (positions[i].distanceTo(positions[j]) < 3.6) links.push(positions[i], positions[j]);
    }
  }
  g.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(links),
    new THREE.LineBasicMaterial({ color: 0xff8fb0, transparent: true, opacity: 0.25 }),
  ));
  // One second of the recording at a time, scrolling at real speed and
  // looping over the two seconds exported. Every point is a recorded value.
  const shownPoints = Math.round(data.waveform.eyes_closed.length / data.waveform.seconds);
  // Drawn as a ribbon, two vertices per sample, so it reads at any size.
  const livePositions = new Float32Array(shownPoints * 2 * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(livePositions, 3));
  const index = [];
  for (let k = 0; k < shownPoints - 1; k++) {
    const a = k * 2;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  geometry.setIndex(index);
  const live = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: COLORS.closed, side: THREE.DoubleSide }));
  live.position.y = 2.2;
  g.add(live);
  // Shown at a quarter of real speed, so the waves can be followed by eye.
  const PLAYBACK_SPEED = 0.25;
  const pointsPerSecond = shownPoints * PLAYBACK_SPEED;
  const thickness = 0.14;
  // A schematic simulation of why the scalp sees alpha (Buzsaki et al.,
  // 2012): scalp EEG sums the activity of many neurons. Firing in step,
  // their activity adds up; out of step, it cancels. Eyes closed puts the
  // simulated neurons in step and eyes open spreads them out, which is the
  // desynchronization Barry et al. (2007) describe. It runs slowed to one
  // cycle every two seconds: alpha's real 8 to 13 Hz would flash faster than the
  // three-per-second limit accessibility guidelines set.
  const SIM_HZ = 0.5;
  const SPREAD = { eyes_closed: 0.08, eyes_open: 1 };
  let spread = SPREAD[state.eyes];
  const sumPoints = 140;
  const sumPositions = new Float32Array(sumPoints * 2 * 3);
  const sumGeometry = new THREE.BufferGeometry();
  sumGeometry.setAttribute("position", new THREE.BufferAttribute(sumPositions, 3));
  const sumIndex = [];
  for (let k = 0; k < sumPoints - 1; k++) {
    const a = k * 2;
    sumIndex.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  sumGeometry.setIndex(sumIndex);
  const sum = new THREE.Mesh(sumGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }));
  sum.position.y = 6.2;
  g.add(sum);
  const pink = new THREE.Color(0xff6f91);
  const shade = new THREE.Color();
  let lastNow = null;
  animate(0, (t, now) => {
    const moving = Number.isFinite(t);
    // Eased by elapsed time, so the change settles in about a second
    // however fast the device draws.
    const dt = moving && lastNow !== null ? Math.min(now - lastNow, 0.1) : 0;
    lastNow = moving ? now : null;
    spread = moving ? THREE.MathUtils.lerp(spread, SPREAD[state.eyes], 1 - Math.exp(-dt * 4)) : SPREAD[state.eyes];
    const time = moving ? t : 0;
    for (let i = 0; i < count; i++) {
      const activity = 0.5 + 0.5 * Math.cos(2 * Math.PI * (SIM_HZ * time + offsets[i] * spread));
      cells.setColorAt(i, shade.copy(pink).multiplyScalar(0.3 + 0.7 * activity));
    }
    cells.instanceColor.needsUpdate = true;
    // The summed activity over the next three simulated cycles.
    for (let k = 0; k < sumPoints; k++) {
      const ahead = (k / (sumPoints - 1)) * 3;
      let total = 0;
      for (let i = 0; i < count; i++) total += Math.cos(2 * Math.PI * (SIM_HZ * time + ahead + offsets[i] * spread));
      const x = (k / (sumPoints - 1) - 0.5) * 18;
      const y = (total / count) * 1.6;
      sumPositions.set([x, y - 0.06, 0, x, y + 0.06, 0], k * 6);
    }
    sumGeometry.attributes.position.needsUpdate = true;

    const values = data.waveform[state.eyes];
    const start = Number.isFinite(t) ? Math.floor(t * pointsPerSecond) % values.length : 0;
    for (let k = 0; k < shownPoints; k++) {
      const v = values[(start + k) % values.length];
      const x = (k / (shownPoints - 1) - 0.5) * 18;
      const y = (v / waveMax) * 2.2;
      livePositions.set([x, y - thickness / 2, 0, x, y + thickness / 2, 0], k * 6);
    }
    geometry.attributes.position.needsUpdate = true;
    live.material.color.set(state.eyes === "eyes_closed" ? COLORS.closed : COLORS.open);
  });
})();

// 2. Sensors: scalp, electrode, wire, the recording device, and the
// antenna from the cited injection study.
const deviceGhosts = new THREE.Group();
(function buildSensors() {
  const g = stations[1];
  const scalp = new THREE.Mesh(new THREE.SphereGeometry(14, 48, 32), matte(0x2a3a3c));
  scalp.position.y = -14.6;
  g.add(scalp);
  const electrode = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 0.25, 32), new THREE.MeshStandardMaterial({ color: 0xc9d6d8, metalness: 0.9, roughness: 0.25 }));
  electrode.position.y = -0.5;
  g.add(electrode);
  const device = new THREE.Mesh(new THREE.BoxGeometry(2.6, 1.5, 1.6), matte(0x1d2b30));
  device.position.set(7.5, 1.3, -3);
  g.add(device);
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(2.1, 1.0), glow(0x6fe3d0, 0.5));
  screen.position.set(7.5, 1.35, -2.19);
  g.add(screen);
  const wireCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, -0.3, 0),
    new THREE.Vector3(1.5, 1.5, 0.5),
    new THREE.Vector3(4.5, 2.2, -1),
    new THREE.Vector3(6.4, 1.3, -2.6),
  ]);
  g.add(new THREE.Mesh(new THREE.TubeGeometry(wireCurve, 40, 0.06, 8), matte(0x9fb4b8)));
  // Each dot is a sample on its way to storage: acquisition, drawn.
  const samples = Array.from({ length: 14 }, () => {
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 10), glow(0x6fe3d0, 1.2));
    g.add(dot);
    return dot;
  });
  animate(1, (t) => {
    const phase = Number.isFinite(t) ? t * 0.35 : 0;
    samples.forEach((dot, k) => dot.position.copy(wireCurve.getPoint((phase + k / samples.length) % 1)));
  });
  // The antenna, about three meters away in the study, placed off to the
  // side with static arcs toward the wire.
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 6, 8), matte(0x5b2230));
  pole.position.set(-9, 2, -6);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.3, 16, 16), glow(COLORS.wrong, 1));
  tip.position.set(-9, 5.1, -6);
  g.add(pole, tip);
  for (let k = 1; k <= 4; k++) {
    const arc = new THREE.Mesh(new THREE.TorusGeometry(k * 1.6, 0.04, 6, 48, Math.PI / 2.4), glow(COLORS.wrong, 0.9));
    arc.position.set(-9, 5.1, -6);
    arc.rotation.set(0, Math.PI / 4.5, -Math.PI / 5);
    arc.material.transparent = true;
    arc.material.opacity = 1 - k * 0.18;
    g.add(arc);
  }
  deviceGhosts.position.set(7.5, 2.9, -3);
  g.add(deviceGhosts);
})();

// 3. Processing: the trace enters gates and leaves as pillars.
(function buildProcessing() {
  const g = stations[2];
  for (let k = 0; k < 7; k++) {
    const gate = new THREE.Mesh(new THREE.TorusGeometry(3.2, 0.07, 8, 48), glow(0x5ad1ff, 0.7));
    gate.rotation.y = Math.PI / 2;
    gate.position.set(-1.5 + k * 0.5, 2.2, 0);
    gate.material.transparent = true;
    gate.material.opacity = 0.35 + k * 0.08;
    g.add(gate);
  }
  const trace = traceLine(data.waveform.eyes_closed, 8, 1.2, COLORS.closed, waveMax);
  trace.position.set(-7, 2.2, 0);
  g.add(trace);
  // One divider per second of the drawn trace: these are the windows.
  for (let s = 0; s <= data.waveform.seconds; s++) {
    const x = -11 + (s / data.waveform.seconds) * 8;
    g.add(line([new THREE.Vector3(x, 0.6, 0), new THREE.Vector3(x, 3.8, 0)], 0xffffff, 0.5));
  }
  const pillars = pillarSet(data.baseline, { spacing: 0.32, height: 4, radius: 0.1, showWrong: false });
  pillars.children.filter((c) => c.geometry?.type === "PlaneGeometry" || c.type === "Line").forEach((c) => (c.visible = false));
  pillars.position.set(6.5, 0, 0);
  g.add(pillars);
  const columns = pillars.children.filter((c) => c.geometry?.type === "CylinderGeometry");
  columns.forEach((c) => (c.userData.height = c.geometry.parameters.height));
  animate(2, (t) => {
    columns.forEach((c, k) => {
      const grown = Number.isFinite(t) ? THREE.MathUtils.clamp((t - 0.3 - k * 0.05) / 0.5, 0.001, 1) : 1;
      c.scale.y = grown;
      c.position.y = (c.userData.height * grown) / 2;
    });
  });
})();

// 4. Inference: every window as a pillar, the threshold as a plane.
(function buildInference() {
  const g = stations[3];
  const grid = new THREE.GridHelper(24, 24, 0x3a3570, 0x1f1c40);
  g.add(grid);
  g.add(pillarSet(data.baseline));
})();

// 5. Decision: the junction, with the legitimate system's inference on
// one side and the requester's on the other. A safeguard rebuilds both
// from measured data, so its effect on each reader is seen side by side.
const systemLand = new THREE.Group();
const requesterLand = new THREE.Group();
const requesterGhosts = new THREE.Group();
const branchPath = new THREE.Group();
const fallenTags = new THREE.Group();
const purposeFence = new THREE.Group();
const denyBar = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.25, 0.25), glow(COLORS.wrong, 1));
const SYSTEM_AT = new THREE.Vector3(-8, 0, -6);
const REQUESTER_AT = new THREE.Vector3(10, 0, -6);
let landsBuiltAt = -Infinity;
let barDropStart = -Infinity;
(function buildDecision() {
  const g = stations[4];
  for (const x of [-2, 2]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.5, 5, 0.5), matte(0x3b2e12));
    post.position.set(x, 2.5, 0);
    g.add(post);
  }
  const lintel = new THREE.Mesh(new THREE.BoxGeometry(4.9, 0.4, 0.6), glow(0xffc46b, 0.6));
  lintel.position.y = 5.1;
  g.add(lintel);
  const building = new THREE.Mesh(new THREE.BoxGeometry(8, 9, 4), matte(0x2b2414));
  building.position.set(REQUESTER_AT.x, 4.5, REQUESTER_AT.z - 5);
  g.add(building);
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      const win = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8), glow(0xffc46b, 0.35));
      win.position.set(REQUESTER_AT.x - 2.8 + col * 1.4, 1.6 + row * 1.6, REQUESTER_AT.z - 2.99);
      g.add(win);
    }
  }
  const systemLabel = labelSprite("Legitimate system");
  systemLabel.position.copy(SYSTEM_AT).add(new THREE.Vector3(0, 6.8, 0));
  const requesterLabel = labelSprite("Requester", "#ffcf6b");
  requesterLabel.position.copy(REQUESTER_AT).add(new THREE.Vector3(0, 10, -3));
  systemLand.position.copy(SYSTEM_AT);
  requesterLand.position.copy(REQUESTER_AT);
  requesterGhosts.position.set(REQUESTER_AT.x - 5, 1.4, REQUESTER_AT.z + 2.5);
  g.add(systemLabel, requesterLabel, systemLand, requesterLand, requesterGhosts, branchPath, fallenTags, purposeFence, denyBar);

  // The gate to the requester, raised until a request is denied.
  denyBar.position.set(REQUESTER_AT.x - 7, 6, REQUESTER_AT.z + 4);
  denyBar.rotation.y = -0.4;

  const junction = new THREE.Vector3(0, 1.4, 2).sub(requesterGhosts.position);
  animate(4, (_t, now) => {
    const moving = Number.isFinite(now);
    // A new copy travels from the junction to the requester.
    const flight = moving ? THREE.MathUtils.clamp((now - copyFlightStart) / 1.4, 0, 1) : 1;
    for (const ghostMesh of requesterGhosts.children) {
      if (!ghostMesh.userData.rest) continue;
      if (ghostMesh.userData.flies) ghostMesh.position.lerpVectors(junction, ghostMesh.userData.rest, ease(flight));
      else ghostMesh.position.copy(ghostMesh.userData.rest);
    }
    // Both landscapes regrow when a safeguard changes them.
    for (const land of [systemLand, requesterLand]) {
      land.traverse((c) => {
        if (c.geometry?.type !== "CylinderGeometry" || c.userData.height === undefined) return;
        const grown = moving ? THREE.MathUtils.clamp((now - landsBuiltAt - 0.2 - c.userData.order * 0.03) / 0.5, 0.001, 1) : 1;
        c.scale.y = grown;
        c.position.y = (c.userData.height * grown) / 2;
      });
    }
    // Removed labels fall from the copy to the ground.
    for (const tag of fallenTags.children) {
      const fall = moving ? THREE.MathUtils.clamp((now - tag.userData.start) / 0.9, 0, 1) : 1;
      tag.position.y = THREE.MathUtils.lerp(tag.userData.fromY, 0.15, fall * fall);
      tag.rotation.z = fall * tag.userData.spin;
    }
    // A denial drops the gate.
    const drop = moving ? THREE.MathUtils.clamp((now - barDropStart) / 0.6, 0, 1) : 1;
    denyBar.position.y = THREE.MathUtils.lerp(6, 1.6, current().safeguard === "deny" ? ease(drop) : 0);
  });
})();

// 6. Action: a decision becomes a command on a screen.
(function buildAction() {
  const g = stations[5];
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(12, 7), matte(0x0d2416));
  screen.position.set(0, 3.5, -2);
  g.add(screen);
  const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(12, 7)), new THREE.LineBasicMaterial({ color: 0x7ef0a0 }));
  frame.position.copy(screen.position);
  g.add(frame);
  const target = new THREE.Mesh(new THREE.RingGeometry(0.6, 0.8, 32), glow(0x7ef0a0, 0.9));
  target.position.set(3, 4.5, -1.95);
  g.add(target);
  const cursor = new THREE.Mesh(new THREE.ConeGeometry(0.35, 0.9, 3), glow(0xffffff, 0.9));
  cursor.rotation.z = Math.PI * 0.8;
  cursor.position.set(-3, 2.5, -1.9);
  g.add(cursor);
  const cursorFrom = new THREE.Vector3(-3, 2.5, -1.9);
  const cursorTo = new THREE.Vector3(2.4, 4.2, -1.9);
  animate(5, (t) => {
    const moved = Number.isFinite(t) ? THREE.MathUtils.clamp((t - 0.4) / 1.6, 0, 1) : 1;
    cursor.position.lerpVectors(cursorFrom, cursorTo, ease(moved));
  });
  g.add(line([new THREE.Vector3(-3, 2.5, -1.9), new THREE.Vector3(2.4, 4.2, -1.9)], 0x7ef0a0, 0.6));
})();

// 7. Feedback: the loop back to the start.
(function buildFeedback() {
  const g = stations[6];
  const loop = new THREE.Mesh(new THREE.TorusGeometry(6, 0.12, 12, 96, Math.PI * 1.6), glow(0xd08cff, 0.8));
  loop.position.set(0, 4, -3);
  loop.rotation.z = -Math.PI * 0.3;
  g.add(loop);
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1.1, 16), glow(0xd08cff, 1));
  const angle = -Math.PI * 0.3 + Math.PI * 1.6;
  head.position.set(Math.cos(angle) * 6, 4 + Math.sin(angle) * 6, -3);
  head.rotation.z = angle;
  g.add(head);
  const pulse = new THREE.Mesh(new THREE.SphereGeometry(0.35, 16, 16), glow(0xffffff, 1.2));
  loop.add(pulse);
  animate(6, (t) => {
    const a = Number.isFinite(t) ? ((t * 0.35) % 1) * Math.PI * 1.6 : Math.PI * 0.8;
    pulse.position.set(Math.cos(a) * 6, Math.sin(a) * 6, 0);
  });
})();

// The pathway joining every station. The camera and the packet travel
// along this same curve, so moving between stages is a journey through
// the worlds in between rather than a cut from one to the next.
const pathCurve = new THREE.CatmullRomCurve3(STAGES.map((_, i) => stationPos(i).add(new THREE.Vector3(0, -0.4, 6))));
const stageU = (i) => i / (STAGES.length - 1);
scene.add(new THREE.Mesh(new THREE.TubeGeometry(pathCurve, 400, 0.12, 8), glow(COLORS.path, 0.5)));

// Drifting matter around the pathway, coloured by the nearest world. It
// does not move; travelling past it is what gives the sense of speed.
(function buildParticles() {
  const rand = seeded(11);
  const count = 3000;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const c = new THREE.Color();
  for (let k = 0; k < count; k++) {
    const u = rand();
    const p = pathCurve.getPoint(u);
    const angle = rand() * Math.PI * 2;
    const radius = 4 + rand() * 16;
    positions.set([p.x + Math.cos(angle) * radius, p.y + Math.sin(angle) * radius * 0.6 + 3, p.z + (rand() - 0.5) * 6], k * 3);
    c.set(WORLD[Math.round(u * (STAGES.length - 1))].light);
    colors.set([c.r, c.g, c.b], k * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.09, vertexColors: true, transparent: true, opacity: 0.75 })));
})();

// The learner: a packet that changes form at each stage and keeps a ring
// for every stage it has passed, so its history travels with it.
const packet = new THREE.Group();
const packetShapes = [
  new THREE.SphereGeometry(0.6, 32, 32),
  new THREE.IcosahedronGeometry(0.62, 1),
  new THREE.BoxGeometry(0.9, 0.9, 0.9),
  new THREE.OctahedronGeometry(0.72),
  new THREE.TetrahedronGeometry(0.8),
  new THREE.ConeGeometry(0.55, 1.1, 24),
  new THREE.SphereGeometry(0.6, 32, 32),
];
const packetCore = new THREE.Mesh(packetShapes[0], glow(0xffffff, 0.9));
const packetLight = new THREE.PointLight(0xffffff, 8, 12, 1.5);
const historyRings = new THREE.Group();
packet.add(packetCore, packetLight, historyRings);
scene.add(packet);

function syncHistoryRings() {
  historyRings.clear();
  [...state.visited].sort().forEach((i, k) => {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.95 + k * 0.16, 0.025, 6, 48), glow(WORLD[i].light, 1));
    ring.rotation.x = Math.PI / 2 + k * 0.35;
    ring.rotation.y = k * 0.5;
    historyRings.add(ring);
  });
}

// Identifying labels on a copy: three small tags.
function tagsFor(target) {
  for (let k = 0; k < 3; k++) {
    const tag = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.22, 0.05), glow(0xffc46b, 0.9));
    tag.position.set(-0.55 + k * 0.55, 0.95, 0);
    tag.userData.inspect = "Identifying labels: device serial number, recording time, session code.";
    target.add(tag);
  }
}

function dressCopy(mesh, c) {
  mesh.add(beacon());
  const mark = fingerprint();
  mark.position.set(0, -0.05, 0.62);
  mesh.add(mark);
  if (c.labels) tagsFor(mesh);
}

function syncCopies() {
  beacons.clear();
  deviceGhosts.clear();
  requesterGhosts.clear();
  for (const label of overviewLabels.splice(0)) label.removeFromParent();
  branchPath.clear();
  purposeFence.clear();
  copies().forEach((c) => {
    const g = ghost(0.8);
    g.userData.inspect = `${c.label}. Held by ${c.holder.toLowerCase()}. ${c.recall}.`;
    dressCopy(g, c);
    if (c.where === "Sensors") {
      g.position.set(deviceGhosts.children.length * 1.6, 0, 0);
      deviceGhosts.add(g);
    } else {
      g.position.set(requesterGhosts.children.length * 1.6, 0, 0);
      g.userData.rest = g.position.clone();
      g.userData.flies = c.pass === state.pass;
      requesterGhosts.add(g);
    }
  });
  for (const [group, text] of [[deviceGhosts, "Device copies"], [requesterGhosts, "Requester copies"]]) {
    if (!group.children.length) continue;
    const label = labelSprite(`${text}: ${group.children.length}`, "#b9f5ff");
    label.scale.set(44, 8.3, 1);
    label.position.set(0, 47, 0);
    label.visible = false;
    group.add(label);
    overviewLabels.push(label);
  }
  const record = current();
  denyBar.visible = record.safeguard === "deny";
  const outcome = safeguardOutcome();
  if (outcome?.requester) {
    branchPath.add(line([new THREE.Vector3(0, 0.6, 2), requesterGhosts.position.clone().setY(0.6), REQUESTER_AT.clone().setY(0.6)], COLORS.copy, 0.8));
  }
  if (record.safeguard === "purpose") {
    // The approved use, drawn as a boundary. The full copy sits inside it.
    const wall = new THREE.Mesh(
      new THREE.CylinderGeometry(7.5, 7.5, 3.2, 48, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x6fe3b0, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }),
    );
    wall.position.set(REQUESTER_AT.x - 2, 1.6, REQUESTER_AT.z + 1);
    wall.userData.inspect = "Approved use only. Enforcement depends on agreements, audits, and technical controls.";
    const rim = new THREE.Mesh(new THREE.TorusGeometry(7.5, 0.05, 6, 96), glow(0x6fe3b0, 1));
    rim.rotation.x = Math.PI / 2;
    rim.position.set(REQUESTER_AT.x - 2, 0.05, REQUESTER_AT.z + 1);
    purposeFence.add(wall, rim);
  }
}

// The two landscapes, rebuilt from measured data when the safeguard or
// its level changes, and the labels a metadata safeguard removes.
function syncLandscapes(changed) {
  systemLand.clear();
  requesterLand.clear();
  const record = current();
  const outcome = outcomeFor(record);
  const options = { spacing: 0.42, height: 5, radius: 0.14 };
  const order = (land) => {
    let k = 0;
    land.traverse((c) => {
      if (c.geometry?.type === "CylinderGeometry" && c.geometry.parameters.height > 0.06) {
        c.userData.height = c.geometry.parameters.height;
        c.userData.order = k++;
      }
    });
  };
  systemLand.add(landscape(outcome.legitimate, options));
  order(systemLand);
  if (record.safeguard && outcome.requester) {
    requesterLand.add(landscape(outcome.requester, options));
    order(requesterLand);
  }
  if (changed) landsBuiltAt = clock.elapsedTime;
  fallenTags.clear();
  if (record.safeguard === "metadata") {
    for (let k = 0; k < 3; k++) {
      const tag = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.22, 0.05), glow(0xffc46b, 0.5));
      tag.position.copy(requesterGhosts.position).add(new THREE.Vector3(-0.55 + k * 0.55, 0.95, 0));
      tag.userData = { start: clock.elapsedTime + 1.4, fromY: tag.position.y, spin: (k - 1) * 1.2, inspect: "A removed label, left behind." };
      fallenTags.add(tag);
    }
  }
}

// ------------------------------------------------------------------
// Camera, travel, and inspection
// ------------------------------------------------------------------

const view = { yaw: 0, pitch: 0 };
const trip = { from: 0, to: 0, t: 1, seconds: 2 };
const currentLook = new THREE.Vector3();
function ease(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function framing(i) {
  const station = stationPos(i);
  // The panel covers the right of the screen on wide layouts and the
  // bottom on narrow ones, so the subject is framed away from it.
  const look = narrow.matches ? station.clone().add(new THREE.Vector3(0, -5.5, 0)) : station.clone().add(new THREE.Vector3(5, 2, 0));
  const eye = station.clone().add(new THREE.Vector3(0, narrow.matches ? 7 : 5.5, narrow.matches ? 22 : 17));
  if (i === 4) {
    // Wide enough for both inference landscapes and the gate between them.
    return narrow.matches
      ? { eye: station.clone().add(new THREE.Vector3(1, 18, 36)), look: station.clone().add(new THREE.Vector3(1, -6, -6)) }
      : { eye: station.clone().add(new THREE.Vector3(3, 12, 25)), look: station.clone().add(new THREE.Vector3(7, 2, -6)) };
  }
  return { eye, look };
}

// Where the packet rests at a stage: just in front of the camera, low and
// to the left, so it is in view and never over the stage's subject.
function packetSpot(i) {
  const { eye, look } = framing(i);
  const forward = look.clone().sub(eye).normalize();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward);
  return eye
    .addScaledVector(forward, 8)
    .addScaledVector(right, narrow.matches ? -1.55 : -3.6)
    .addScaledVector(up, narrow.matches ? 1.9 : -1.6);
}

function moveCamera(i, instant) {
  trip.from = instant ? i : state.previous ?? i;
  trip.to = i;
  trip.seconds = Math.min(1.8 + 0.6 * Math.abs(trip.to - trip.from), 4.2);
  trip.t = instant || reducedMotion.matches || trip.from === trip.to ? 1 : 0;
  view.yaw = 0;
  view.pitch = 0;
}

// The view over the whole pathway that ends the journey, and how far
// into it the camera has risen.
// Looking back along the pathway from above its end, so every stage
// and every copy is in view, the nearest stages largest.
const OVERVIEW = {
  eye: new THREE.Vector3(-30, 85, -410),
  look: new THREE.Vector3(-34, 0, -175),
};
// Who holds the copies, named only in the overview.
const overviewLabels = [];
function overviewAmount(now) {
  if (state.stage !== STAGES.length - 1 || trip.t < 1) return 0;
  if (reducedMotion.matches) return 1;
  return ease(THREE.MathUtils.clamp((now - arrivedAt - 1.5) / 3.5, 0, 1));
}

// Camera, look target, and packet position for the current moment of a
// trip. At both ends it is the stage framing; in between, the camera
// follows the packet along the pathway.
function pose() {
  const e = ease(trip.t);
  const a = framing(trip.from);
  const b = framing(trip.to);
  const eye = a.eye.lerp(b.eye, e);
  const look = a.look.lerp(b.look, e);
  const rest = packetSpot(trip.from).lerp(packetSpot(trip.to), e);
  if (trip.t >= 1 || trip.from === trip.to) {
    const w = overviewAmount(clock.elapsedTime);
    return { eye: eye.lerp(OVERVIEW.eye, w), look: look.lerp(OVERVIEW.look, w), packetAt: rest, overview: w };
  }

  const u = THREE.MathUtils.lerp(stageU(trip.from), stageU(trip.to), e);
  const direction = Math.sign(trip.to - trip.from);
  const ride = pathCurve.getPoint(u).add(new THREE.Vector3(0, 1.6, 0));
  const ahead = pathCurve.getPoint(THREE.MathUtils.clamp(u + direction * 0.02, 0, 1)).add(new THREE.Vector3(0, 1.2, 0));
  const heading = ahead.clone().sub(ride).normalize();
  const chaseEye = ride.clone().addScaledVector(heading, -7).add(new THREE.Vector3(0, 2.4, 0));
  const w = Math.sin(Math.PI * trip.t);
  return {
    eye: eye.lerp(chaseEye, w),
    look: look.lerp(ahead, w),
    packetAt: rest.lerp(ride, w),
    overview: 0,
  };
}

// Inspection: tapping a pillar or a copy says what it is.
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
function inspectableAt(clientX, clientY) {
  pointer.set((clientX / window.innerWidth) * 2 - 1, -(clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  for (const hit of raycaster.intersectObjects(scene.children, true)) {
    if (hit.object.userData.inspect && hit.object.visible) return hit.object;
  }
  return null;
}
function showInspect(text, x, y) {
  const tip = $("inspect");
  if (!text) {
    tip.hidden = true;
    return;
  }
  tip.textContent = text;
  tip.hidden = false;
  tip.style.left = `${Math.min(x + 14, window.innerWidth - tip.offsetWidth - 8)}px`;
  tip.style.top = `${Math.max(y - tip.offsetHeight - 10, 8)}px`;
}

let dragging = null;
canvas.addEventListener("pointerdown", (e) => {
  dragging = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch, moved: false };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) {
    canvas.style.cursor = inspectableAt(e.clientX, e.clientY) ? "pointer" : "";
    return;
  }
  if (Math.hypot(e.clientX - dragging.x, e.clientY - dragging.y) > 6) dragging.moved = true;
  view.yaw = THREE.MathUtils.clamp(dragging.yaw - (e.clientX - dragging.x) * 0.005, -1.1, 1.1);
  view.pitch = THREE.MathUtils.clamp(dragging.pitch + (e.clientY - dragging.y) * 0.004, -0.35, 0.5);
});
canvas.addEventListener("pointerup", (e) => {
  if (dragging && !dragging.moved) {
    const hit = inspectableAt(e.clientX, e.clientY);
    showInspect(hit?.userData.inspect, e.clientX, e.clientY);
  }
  dragging = null;
});
canvas.addEventListener("pointercancel", () => (dragging = null));

const clock = new THREE.Clock();
const fogColor = new THREE.Color();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.05);
  if (trip.t < 1) trip.t = Math.min(1, trip.t + dt / trip.seconds);
  const { eye, look, packetAt, overview } = pose();
  // The fog thins as the camera rises, so the whole pathway shows.
  scene.fog.density = THREE.MathUtils.lerp(0.028, 0.0012, overview);
  for (const label of overviewLabels) label.visible = overview > 0.4;
  for (const b of beacons) b.scale.set(1 + 5 * overview, 1, 1 + 5 * overview);
  currentLook.copy(look);

  // Drag orbits the eye around what it is looking at.
  const offset = eye.clone().sub(look);
  offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), view.yaw);
  const side = new THREE.Vector3().crossVectors(offset, new THREE.Vector3(0, 1, 0)).normalize();
  offset.applyAxisAngle(side, view.pitch);
  camera.position.copy(look).add(offset);
  camera.lookAt(look);

  packet.position.copy(packetAt);

  const now = clock.elapsedTime;
  if (trip.t >= 1 && settledStage !== trip.to) {
    settledStage = trip.to;
    arrivedAt = now;
  }
  for (const a of animations) {
    if (a.stage !== state.stage) continue;
    if (reducedMotion.matches) a.update(Infinity, Infinity);
    else a.update(trip.t >= 1 ? now - arrivedAt : 0, now);
  }
  packet.scale.setScalar(narrow.matches ? 0.3 : 0.55);
  if (!reducedMotion.matches) historyRings.rotation.y += dt * 0.25;

  // The world's colour follows where the camera actually is on the path.
  const nearest = Math.round(THREE.MathUtils.lerp(trip.from, trip.to, ease(trip.t)));
  key.position.copy(stationPos(nearest)).add(new THREE.Vector3(0, 8, 6));
  fogColor.set(WORLD[nearest].fog);
  scene.fog.color.lerp(fogColor, reducedMotion.matches ? 1 : Math.min(1, dt * 2.5));
  scene.background.copy(scene.fog.color);

  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// ------------------------------------------------------------------
// Panel
// ------------------------------------------------------------------

// A recorded trace with labeled axes: microvolts up the side, seconds
// along the bottom. Every point is a recorded sample from data.waveform.
function traceSvg(values, { color, width = 1.4, height = 70, maxAbs = waveMax, label, cls = "" }) {
  const left = 38;
  const right = 8;
  const w = 360;
  const plotW = w - left - right;
  const plotH = height;
  const total = plotH + 18;
  const mid = plotH / 2;
  const y = (v) => (mid - (v / maxAbs) * (mid - 3)).toFixed(1);
  const n = values.length;
  const d = values.map((v, i) => `${i ? "L" : "M"}${(left + (i / (n - 1)) * plotW).toFixed(1)},${y(v)}`).join("");
  const top = Math.round(maxAbs);
  const secs = data.waveform.seconds;
  const xTicks = Array.from({ length: secs + 1 }, (_, s) => s)
    .map((s) => `<text x="${(left + (s / secs) * plotW).toFixed(1)}" y="${plotH + 13}" text-anchor="middle" class="ax">${s} s</text>`)
    .join("");
  return `<svg viewBox="0 0 ${w} ${total}" role="img" aria-label="${esc(label)}: recorded voltage at O1 in microvolts over ${secs} seconds" class="trace-svg">
    <line x1="${left}" y1="${mid}" x2="${w - right}" y2="${mid}" class="ax-line zero"/>
    <line x1="${left}" y1="0" x2="${left}" y2="${plotH}" class="ax-line"/>
    <line x1="${left}" y1="${plotH}" x2="${w - right}" y2="${plotH}" class="ax-line"/>
    <text x="${left - 4}" y="8" text-anchor="end" class="ax">+${top}</text>
    <text x="${left - 4}" y="${mid + 3}" text-anchor="end" class="ax">0 µV</text>
    <text x="${left - 4}" y="${plotH}" text-anchor="end" class="ax">−${top}</text>
    ${xTicks}
    <path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" class="${cls}"${cls ? ' pathLength="1"' : ""}/>
  </svg>`;
}

function waveformSvg(closed, open, maxAbs = waveMax) {
  const row = (values, color, label, key) => {
    const selected = state.eyes === key;
    return `<div class="wave${selected ? " selected" : ""}">${traceSvg(values, { color, width: selected ? 2 : 1, height: 46, maxAbs, label })}
      <p class="wave-label">${label}. Average alpha power ${ALPHA_MEAN[key].toFixed(1)} µV².</p></div>`;
  };
  return (
    row(closed, "var(--closed)", "Eyes closed", "eyes_closed") +
    row(open, "var(--open)", "Eyes open", "eyes_open") +
    `<p>Average alpha power is about ${(ALPHA_MEAN.eyes_closed / ALPHA_MEAN.eyes_open).toFixed(0)} times higher with the eyes closed in this recording.${refMarks(["barry"])}</p>`
  );
}

// ------------------------------------------------------------------
// Neuroanatomy figure: where the signal comes from, and what other
// sensors reach. Facts come from data.neuroanatomy (core/neuroanatomy.py);
// this code only places them on a drawing.
// ------------------------------------------------------------------

const ANAT = data.neuroanatomy;
const SYSTEM = Object.fromEntries(ANAT.systems.map((s) => [s.key, s]));
const MODALITY = Object.fromEntries(ANAT.modalities.map((m) => [m.key, m]));
const ELECTRODE = Object.fromEntries(ANAT.electrodes.map((e) => [e.label, e]));
const levelOf = (m, key) => m.coverage.find((c) => c.system === key);
const LEVEL_TEXT = { covered: "Signal from here", partial: "Partial or weak", none: "Out of range" };
// Colour shows what kind of signal a sensor gets; shading shows where from.
const MEASURE_COLOR = { electrical: "#7cc4ff", hemodynamic: "#c39bff", anatomy: "#b8c0d8", peripheral: "#6fe3b0" };
const MEASURE_LABEL = { electrical: "Electrical potentials", hemodynamic: "Blood oxygen", anatomy: "Anatomy only", peripheral: "Body response" };

// Side view, facing left. Shapes are a simplified teaching drawing.
const CEREBRUM = "M92,150 C66,132 62,92 84,66 C108,38 150,26 198,28 C248,30 286,52 296,92 C304,124 296,156 270,170 C252,178 232,172 214,170 C196,176 172,182 150,180 C124,178 104,168 92,150 Z";
const HEAD = "M165,14 C245,8 322,45 326,112 C330,165 300,200 272,214 L270,262 L120,262 L118,224 C95,224 72,216 64,202 C60,194 62,186 58,180 C52,178 48,172 52,166 C46,162 46,156 52,152 C44,148 38,142 36,136 L47,123 C47,111 45,100 49,88 C59,40 105,18 165,14 Z";
const SHAPES = {
  frontal: { d: "M0,0 L167,0 L167,28 L140,131 L98,140 L0,150 Z", label: [112, 92, "Frontal"] },
  sensorimotor: { d: "M167,0 L195,0 L193,32 L164,125 L140,131 L167,28 Z", label: [166, 80, "Motor / sensory", -72] },
  parietal: { d: "M195,0 L254,0 L252,44 L255,124 L210,122 L164,125 L193,32 Z", label: [222, 66, "Parietal"] },
  occipital: { d: "M254,0 L360,0 L360,270 L262,175 L255,124 L252,44 Z", label: [276, 104, "Occipital"] },
  temporal: { d: "M0,150 L98,140 L140,131 L164,125 L210,122 L255,124 L262,175 L0,270 Z", label: [128, 166, "Temporal"] },
};
const EEG_SITES = [[58, 72], [104, 32], [166, 16], [232, 20], [288, 50], [318, 90]];
const O1_AT = [323, 128];

function sideSvg(m, { electrodeOnly = false } = {}) {
  const sel = electrodeOnly ? null : state.anatomySystem;
  const color = MEASURE_COLOR[m.measurement];
  const fillFor = (key) => {
    if (electrodeOnly) return "none";
    const level = levelOf(m, key)?.level;
    return level === "covered" ? color : level === "partial" ? "url(#hatch)" : "none";
  };
  const op = (key) => (!electrodeOnly && levelOf(m, key)?.level === "covered" ? 0.42 : 1);
  const region = (key, inner) =>
    `<g class="region${sel === key ? " selected" : ""}" data-system="${key}" role="button" tabindex="0" aria-label="${esc(SYSTEM[key].name)}">${inner}<title>${esc(SYSTEM[key].name)}</title></g>`;
  const lobes = Object.entries(SHAPES)
    .map(([k, s]) => region(k, `<path d="${s.d}" fill="${fillFor(k)}" fill-opacity="${op(k)}" stroke="var(--line)" stroke-width="1"/>`))
    .join("");
  const lobeLabels = Object.entries(SHAPES)
    .map(([, s]) => {
      const [x, y, t, rot] = s.label;
      return `<text x="${x}" y="${y}" class="lbl${t.length > 10 ? " small" : ""}" text-anchor="middle"${rot ? ` transform="rotate(${rot} ${x} ${y})"` : ""}>${t}</text>`;
    })
    .join("");
  const deepStroke = (key) => (sel === key ? "var(--text)" : "var(--line-strong)");
  const deep = (key, shape) =>
    region(key, shape.replace("/>", ` fill="${fillFor(key)}" fill-opacity="${op(key)}" stroke="${deepStroke(key)}" stroke-dasharray="3 2" stroke-width="1.2"/>`));
  const face = !electrodeOnly && (m.key === "smri" || m.key === "fmri")
    ? `<path d="${HEAD}" clip-path="url(#face-clip)" fill="var(--does-not)" fill-opacity="0.22" stroke="var(--does-not)" stroke-width="3"/><text x="22" y="150" class="lbl warn" text-anchor="middle">Face</text>`
    : "";
  let sensors = "";
  if (electrodeOnly || m.key === "eeg") {
    sensors = EEG_SITES.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="4" class="electrode"/>`).join("") +
      `<circle cx="${O1_AT[0]}" cy="${O1_AT[1]}" r="6.5" class="electrode o1"/><text x="${O1_AT[0] + 10}" y="${O1_AT[1] + 4}" class="lbl strong">O1</text>`;
  } else if (m.key === "fnirs") {
    sensors = `<path d="M52,82 Q86,84 76,46" class="light"/><rect x="47" y="78" width="9" height="9" class="optode"/><rect x="71" y="38" width="9" height="9" class="optode"/><text x="20" y="64" class="lbl">Light</text>`;
  } else if (m.key === "peripheral") {
    sensors = `<text x="232" y="256" class="lbl strong">To heart and skin</text>`;
  }
  const autonomicOn = !electrodeOnly && levelOf(m, "autonomic")?.level !== "none";
  const youAre = `<g class="you"><circle cx="284" cy="128" r="5"/><circle cx="284" cy="128" r="5" class="pulse"/></g><text x="284" y="146" class="lbl strong" text-anchor="middle">You</text>`;
  const description = electrodeOnly
    ? "Side view of the head facing left. Electrode O1 sits on the scalp over the left occipital region, with a schematic glow showing that it picks up a blend of nearby sources."
    : `Side view of the head facing left. ${m.short_name} measures ${MEASURE_LABEL[m.measurement].toLowerCase()}. ${ANAT.systems.map((s) => `${LEVEL_TEXT[levelOf(m, s.key).level].toLowerCase()}: ${s.name}`).join("; ")}.`;
  return `<svg class="anat" viewBox="0 0 360 270" role="img" aria-label="${esc(description)}">
    <defs>
      <clipPath id="brain-clip"><path d="${CEREBRUM}"/></clipPath>
      <clipPath id="face-clip"><rect x="0" y="84" width="66" height="150"/></clipPath>
      <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="${color}" fill-opacity="0.08"/><line x1="0" y1="0" x2="0" y2="6" stroke="${color}" stroke-width="2" stroke-opacity="0.7"/></pattern>
      <radialGradient id="o1-glow"><stop offset="0" stop-color="#ff9f6b" stop-opacity="0.75"/><stop offset="0.55" stop-color="#ff9f6b" stop-opacity="0.25"/><stop offset="1" stop-color="#ff9f6b" stop-opacity="0"/></radialGradient>
    </defs>
    <path d="${HEAD}" class="head"/>
    ${face}
    ${region("autonomic", `<path d="M215,238 C214,248 216,256 214,268" class="nerve${autonomicOn ? " on" : ""}"/><path d="M200,236 h30 v34 h-30 z" fill="transparent"/>`)}
    ${region("brainstem", `<path d="M196,166 C200,190 204,214 206,240 L224,240 C222,214 220,190 222,168 Z" fill="${fillFor("brainstem")}" fill-opacity="${op("brainstem")}" stroke="var(--line)"/>`)}
    ${region("cerebellum", `<ellipse cx="250" cy="186" rx="35" ry="18" fill="${fillFor("cerebellum")}" fill-opacity="${op("cerebellum")}" stroke="var(--line)"/>`)}
    <path d="${CEREBRUM}" fill="var(--brain)"/>
    <g clip-path="url(#brain-clip)">${lobes}</g>
    ${electrodeOnly ? `<g clip-path="url(#brain-clip)"><circle cx="${O1_AT[0]}" cy="${O1_AT[1]}" r="78" fill="url(#o1-glow)"/></g>` : ""}
    <path d="${CEREBRUM}" fill="none" stroke="var(--line-strong)" stroke-width="1.4"/>
    ${deep("thalamus", `<ellipse cx="200" cy="114" rx="15" ry="9"/>`)}
    ${deep("hippocampus_amygdala", `<ellipse cx="178" cy="148" rx="20" ry="6" transform="rotate(-10 178 148)"/>`)}
    ${lobeLabels}
    <text x="200" y="104" class="lbl small" text-anchor="middle">Thalamus</text>
    <text x="178" y="140" class="lbl small" text-anchor="middle">Hippocampus</text>
    <text x="250" y="190" class="lbl small" text-anchor="middle">Cerebellum</text>
    <text x="232" y="230" class="lbl small">Brainstem</text>
    ${electrodeOnly ? "" : youAre}
    ${sensors}
  </svg>`;
}

// Top view of the 10-20 layout, nose up. Positions are the standard
// drawing: an outer ring at 36 degree steps and an inner cross.
const MAP_SITES = (() => {
  const ring = ["Fp2", "F8", "T8", "P8", "O2", "O1", "P7", "T7", "F7", "Fp1"];
  const sites = ring.map((label, i) => {
    const a = ((18 + i * 36) * Math.PI) / 180;
    return [label, Math.sin(a) * 0.8, -Math.cos(a) * 0.8];
  });
  return sites.concat([
    ["Fpz", 0, -0.8], ["Oz", 0, 0.8], ["Fz", 0, -0.4], ["Cz", 0, 0], ["Pz", 0, 0.4],
    ["C3", -0.4, 0], ["C4", 0.4, 0], ["F3", -0.34, -0.44], ["F4", 0.34, -0.44], ["P3", -0.34, 0.44], ["P4", 0.34, 0.44],
  ]);
})();

function topSvg() {
  const r = 100;
  const c = 120;
  const chosen = state.electrode;
  const dots = MAP_SITES.map(([label, x, y]) => {
    const cx = c + x * r;
    const cy = c + y * r;
    const cls = `site${label === ANAT.recording_electrode.label ? " o1" : ""}${label === chosen ? " chosen" : ""}`;
    return `<g class="${cls}" data-electrode="${label}" role="button" tabindex="0" aria-label="Electrode ${label}"><circle cx="${cx}" cy="${cy}" r="11"/><text x="${cx}" y="${cy + 3.5}" text-anchor="middle">${label}</text></g>`;
  }).join("");
  return `<svg class="anat top" viewBox="0 0 240 240" role="img" aria-label="Top view of the head with the nose at the top, showing the 10-20 electrode positions. O1 is at the back, left of the midline.">
    <path d="M${c - 12},${c - r - 2} L${c},${c - r - 16} L${c + 12},${c - r - 2}" class="head"/>
    <ellipse cx="${c - r - 4}" cy="${c}" rx="7" ry="16" class="head"/><ellipse cx="${c + r + 4}" cy="${c}" rx="7" ry="16" class="head"/>
    <circle cx="${c}" cy="${c}" r="${r}" class="head"/>
    <line x1="${c}" y1="${c - r}" x2="${c}" y2="${c + r}" class="midline"/>
    <text x="10" y="232" class="lbl small">Left</text><text x="230" y="232" class="lbl small" text-anchor="end">Right</text>
    <text x="${c}" y="12" class="lbl small" text-anchor="middle">Nose</text>
    ${dots}
  </svg>`;
}

function anatomyHtml(stage) {
  const a = stage.anatomy;
  if (!a) return "";
  if (a.view === "electrode") {
    const e = ELECTRODE[state.electrode] || ANAT.recording_electrode;
    const sys = SYSTEM[e.system];
    return `<div class="box anatomy"><h3>${esc(a.title)}</h3><p>${esc(a.text)}${refMarks(e.refs)}</p>
      ${topSvg()}
      <p class="electrode-read"><strong>${esc(e.label)}.</strong> ${esc(e.naming_note)} It sits over the ${esc(sys.name.toLowerCase())}.${e.label === ANAT.recording_electrode.label ? " This journey's recording comes from here." : ""}</p>
      ${sideSvg(MODALITY.eeg, { electrodeOnly: true })}
      <p class="caveat">Side view: O1 on the scalp over the left occipital region. ${esc(ANAT.electrode_caveat)}${refMarks(ANAT.electrode_caveat_refs)}</p></div>`;
  }
  const m = MODALITY[state.modality];
  const chips = ANAT.modalities
    .map((x) => `<button type="button" class="chip" data-modality="${x.key}" aria-pressed="${x.key === m.key}">${esc(x.short_name)}${x.in_journey ? " · you" : ""}</button>`)
    .join("");
  const sel = state.anatomySystem ? SYSTEM[state.anatomySystem] : null;
  const selReach = sel ? levelOf(m, sel.key) : null;
  const selected = sel
    ? `<div class="system-card"><strong>${esc(sel.name)}</strong> <span class="lvl ${selReach.level}">${esc(m.short_name)}: ${LEVEL_TEXT[selReach.level].toLowerCase()}</span>
        <p>${esc(sel.location)}. ${esc(sel.role)}${refMarks(sel.refs)}</p>${selReach.note ? `<p class="caveat">${esc(selReach.note)}</p>` : ""}</div>`
    : `<p class="caveat">Tap a region on the drawing to see what it does.</p>`;
  const rows = ANAT.systems
    .map((s) => {
      const r = levelOf(m, s.key);
      return `<li><button type="button" class="sys-row${state.anatomySystem === s.key ? " selected" : ""}" data-system="${s.key}"><span>${esc(s.name)}</span><span class="lvl ${r.level}">${LEVEL_TEXT[r.level]}</span></button></li>`;
    })
    .join("");
  return `<div class="box anatomy"><h3>${esc(a.title)}</h3><p>${esc(a.text)}</p>
    <div class="chips" role="group" aria-label="Sensor">${chips}</div>
    ${sideSvg(m)}
    <div class="legend" style="--m:${MEASURE_COLOR[m.measurement]}"><span><i style="background:var(--m);opacity:.6"></i>Signal from here</span><span><i class="hatch"></i>Partial or weak</span><span><i style="border:1px solid var(--line-strong)"></i>Out of range</span></div>
    <p class="measure-tag" style="--m:${MEASURE_COLOR[m.measurement]}"><i></i>Signal type: <strong>${MEASURE_LABEL[m.measurement]}</strong></p>
    ${selected}
    <dl class="modality-facts">
      <dt>Setup</dt><dd>${esc(m.configuration)}</dd>
      <dt>Measures</dt><dd>${esc(m.measures)}</dd>
      <dt>Link to neurons</dt><dd>${esc(m.neural_link)}</dd>
      <dt>Timing</dt><dd>${esc(m.timing)}</dd>
      <dt>Also captures</dt><dd>${esc(m.also_captures)} ${esc(m.privacy_note)}${refMarks(m.refs)}</dd>
    </dl>
    <details class="systems"><summary>All systems for ${esc(m.short_name)}</summary><ul>${rows}</ul></details>
    <p class="caveat">${esc(ANAT.caveat)}${refMarks(ANAT.caveat_refs)}</p></div>`;
}

function anatomyTextHtml(stage) {
  const a = stage.anatomy;
  if (!a) return "";
  if (a.view === "electrode") {
    const e = ANAT.recording_electrode;
    return `<h3>${esc(a.title)}</h3><p>${esc(a.text)}${refMarks(e.refs)}</p><p><strong>${esc(e.label)}.</strong> ${esc(e.naming_note)} It sits over the ${esc(SYSTEM[e.system].name.toLowerCase())}.</p><p class="caveat">${esc(ANAT.electrode_caveat)}${refMarks(ANAT.electrode_caveat_refs)}</p>`;
  }
  const head = ANAT.modalities.map((m) => `<th>${esc(m.short_name)}</th>`).join("");
  const body = ANAT.systems
    .map((s) => `<tr><th scope="row">${esc(s.name)}</th>${ANAT.modalities.map((m) => `<td>${LEVEL_TEXT[levelOf(m, s.key).level]}</td>`).join("")}</tr>`)
    .join("");
  const kinds = `<tr><th scope="row">Signal type</th>${ANAT.modalities.map((m) => `<td>${MEASURE_LABEL[m.measurement]}</td>`).join("")}</tr>`;
  const mods = ANAT.modalities
    .map((m) => `<li><strong>${esc(m.name)} (${esc(m.short_name)}).</strong> ${esc(m.configuration)} ${esc(m.measures)} ${esc(m.neural_link)} ${esc(m.timing)} Also captures: ${esc(m.also_captures)} ${esc(m.privacy_note)}${refMarks(m.refs)}</li>`)
    .join("");
  const systems = ANAT.systems.map((s) => `<li><strong>${esc(s.name)}.</strong> ${esc(s.location)}. ${esc(s.role)}${refMarks(s.refs)}</li>`).join("");
  return `<h3>${esc(a.title)}</h3><p>You start in the ${esc(SYSTEM.occipital.name.toLowerCase())}.</p>
    <h4>Brain and body systems</h4><ul>${systems}</ul>
    <h4>Where each sensor gets a signal</h4><div class="table-wrap"><table><thead><tr><th>System</th>${head}</tr></thead><tbody>${kinds}${body}</tbody></table></div>
    <ul>${mods}</ul><p class="caveat">${esc(ANAT.caveat)}${refMarks(ANAT.caveat_refs)}</p>`;
}

// ------------------------------------------------------------------
// Study room: how an eyes-closed / eyes-open recording is made, what
// each number means, and the resting / task and domain-general /
// domain-specific dichotomies. Words and numbers come from
// data.resting_state (core/resting_state.py); this code draws them.
// ------------------------------------------------------------------

const RS = data.resting_state;
const MEASURE = Object.fromEntries(RS.measures.map((m) => [m.key, m]));
const RANGE_LABEL = {
  "by design": "Fixed by design",
  calculated: "Calculated for this sample",
  direction: "Expected direction",
  "within person": "Compare within a person",
};
const STUDY_TABS = [
  ["room", "The Recording"],
  ["numbers", "What the Numbers Mean"],
  ["dichotomies", "Resting vs Task"],
];

// A seated participant, side view facing right. Eyes, the O1 glow and
// the alpha meter follow state.eyes; the trace is the recorded signal.
function participantSvg() {
  const closed = state.eyes === "eyes_closed";
  const glow = closed ? 0.85 : 0.25;
  const eye = closed
    ? `<path d="M181,84 q5,3.5 10,0" class="lid"/>`
    : `<ellipse cx="186" cy="84" rx="5.5" ry="3.6" class="eye-white"/><circle cx="188" cy="84" r="2.2" class="pupil"/>`;
  const capSites = [[150, 52], [168, 44], [186, 48], [200, 62], [136, 64], [128, 82]];
  return `<svg class="participant" viewBox="0 0 420 300" role="img" aria-label="A participant seated at rest with ${closed ? "eyes closed" : "eyes open"}, wearing an EEG cap with electrode O1 at the back of the head, and two skin conductance electrodes on the fingers of the left hand. Cables run to two amplifiers.">
    <defs><radialGradient id="o1g"><stop offset="0" stop-color="#ff9f6b" stop-opacity="${glow}"/><stop offset="1" stop-color="#ff9f6b" stop-opacity="0"/></radialGradient></defs>
    <rect x="0" y="268" width="420" height="32" class="floor"/>
    <path d="M92,150 L104,262 M206,196 L214,262 M92,196 H220" class="chair"/>
    <path d="M86,112 Q80,150 92,200" class="chair back"/>
    <path d="M120,198 C150,200 196,196 222,200 L232,262 L246,262" class="body"/>
    <path d="M132,118 C110,140 108,178 120,200 L200,200 C206,170 196,138 176,118 Z" class="body torso"/>
    <path d="M160,124 C176,150 190,170 230,176" class="body arm"/>
    <ellipse cx="236" cy="177" rx="10" ry="6" class="skin"/>
    <rect x="242" y="171" width="7" height="4" rx="1.5" class="scl"/><rect x="242" y="178" width="7" height="4" rx="1.5" class="scl"/>
    <path d="M249,173 C290,170 300,214 330,226 M249,180 C286,184 296,220 330,232" class="wire"/>
    <rect x="330" y="214" width="66" height="34" rx="5" class="box dashed"/><text x="363" y="235" class="lbl" text-anchor="middle">Skin cond.</text>
    <text x="363" y="262" class="lbl small" text-anchor="middle">published finding</text>
    <rect x="150" y="104" width="22" height="18" class="skin"/>
    <circle cx="164" cy="80" r="34" class="skin head"/>
    <path d="M196,80 L204,92 L196,95 M188,104 q4,2 8,0 M180,76 q6,-3 12,0" class="feature"/>
    <ellipse cx="160" cy="86" rx="5" ry="8" class="skin"/>
    ${eye}
    <path d="M130,78 C130,36 196,30 200,70" class="cap"/>
    ${capSites.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="3.2" class="site"/>`).join("")}
    <circle cx="131" cy="96" r="22" fill="url(#o1g)"/>
    <circle cx="131" cy="96" r="4.5" class="site o1"/><text x="104" y="118" class="lbl strong">O1</text>
    <path d="M130,80 C90,90 60,160 54,214" class="wire"/>
    <rect x="16" y="214" width="76" height="34" rx="5" class="box"/><text x="54" y="229" class="lbl" text-anchor="middle">EEG amplifier</text><text x="54" y="241" class="lbl small" text-anchor="middle">500 samples/s</text>
    <text x="54" y="262" class="lbl small" text-anchor="middle">recorded here</text>
    <text x="232" y="34" class="lbl strong">${closed ? "Eyes closed" : "Eyes open"}, at rest</text>
    <text x="232" y="50" class="lbl small">No task. Sitting still.</text>
  </svg>`;
}

function liveTraceHtml() {
  const values = data.waveform[state.eyes];
  const r = RS.reactivity;
  const mean = state.eyes === "eyes_closed" ? r.mean_closed : r.mean_open;
  const pctOfMax = (mean / Math.max(r.mean_closed, r.mean_open)) * 100;
  return `<div class="live">
    <p class="live-label"><span class="src recorded">${esc(RS.source_labels.recorded)}</span> EEG at O1</p>
    <div class="trace">${traceSvg(values, { color: state.eyes === "eyes_closed" ? "var(--closed)" : "var(--open)", label: state.eyes === "eyes_closed" ? "Eyes closed" : "Eyes open", cls: `trace-path ${state.eyes}` })}</div>
    <p class="live-label">Mean alpha power: <strong>${mean.toFixed(1)} µV²</strong></p>
    <div class="meter-bar"><i class="${state.eyes}" style="width:${pctOfMax.toFixed(0)}%"></i></div>
    <p class="live-label"><span class="src published">${esc(RS.source_labels.published)}</span> Skin conductance: <strong>${state.eyes === "eyes_closed" ? "lower" : "higher"}</strong> ${state.eyes === "eyes_closed" ? "↓" : "↑"}</p>
    <p class="caveat">Barry et al. (2007) measured this in 28 students. This dataset has no skin conductance channel, so no value is shown for this recording.${refMarks(["barry"])}</p>
  </div>`;
}

function studyCard(s, title) {
  return `<div class="box"><h3>${esc(title)}</h3>
    <p><strong>Who:</strong> ${esc(s.participants)}</p>
    <p><strong>Conditions:</strong> ${s.conditions.map(esc).join("; ")}.</p>
    <p><strong>Recorded:</strong> ${s.recordings.map(esc).join(" ")}</p>
    <p>${esc(s.protocol)}</p>
    <ul>${s.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>
    <p>${esc(s.interpretation)}${refMarks(s.refs)}</p></div>`;
}

function roomHtml() {
  const steps = [
    ["Seated at rest", "Cap on, electrodes checked, no task."],
    ["Eyes closed", "Alpha strongest. Skin conductance lower (published finding)."],
    ["Eyes open", "Alpha drops. Skin conductance rises (published finding)."],
    ["Compare", "Same person, two conditions, second by second."],
  ];
  return `<div class="choices levels" role="group" aria-label="Condition">
      <button type="button" class="choice" data-eyes="eyes_closed" aria-pressed="${state.eyes === "eyes_closed"}">Eyes closed</button>
      <button type="button" class="choice" data-eyes="eyes_open" aria-pressed="${state.eyes === "eyes_open"}">Eyes open</button>
    </div>
    <div class="room">${participantSvg()}${liveTraceHtml()}</div>
    <ol class="protocol">${steps.map(([t, d], i) => `<li class="${(state.eyes === "eyes_closed" && i === 1) || (state.eyes === "eyes_open" && i === 2) ? "now" : ""}"><span class="rail-num">${i + 1}</span><strong>${t}</strong><span>${d}</span></li>`).join("")}</ol>
    <p class="caveat">Barry and De Blasio (2017) recorded 3 minutes of each, eyes closed first. This journey uses a 15-second excerpt of each condition from one channel.${refMarks(["barry_2017", "sanchez_gama"])}</p>
    ${studyCard(RS.study, "The Study: Barry et al. (2007)")}
    ${studyCard(RS.ageing, "Does It Hold With Age? Barry and De Blasio (2017)")}`;
}

// Range graphics: each draws only what core/resting_state.py supports.
function bandsSvg() {
  const max = 30;
  const x = (hz) => 10 + (hz / max) * 340;
  const colors = { Delta: "#5a6bd8", Theta: "#9a8cff", Alpha: "#ff9f6b", Beta: "#6fe3d0" };
  const bars = RS.bands
    .map((b) => `<rect x="${x(b.low_hz)}" y="18" width="${x(b.high_hz) - x(b.low_hz)}" height="22" fill="${colors[b.name]}" fill-opacity="${b.name === "Alpha" ? 0.9 : 0.35}"/><text x="${(x(b.low_hz) + x(b.high_hz)) / 2}" y="33" text-anchor="middle" class="lbl ${b.name === "Alpha" ? "dark" : ""}">${b.name}</text>`)
    .join("");
  const ticks = [0, 4, 8, 13, 20, 30].map((t) => `<text x="${x(t)}" y="54" text-anchor="middle" class="lbl small">${t}</text>`).join("");
  return `<svg viewBox="0 0 360 62" class="range" role="img" aria-label="Conventional EEG bands from 0 to 30 Hz: delta, theta, alpha from 8 to 13 Hz, and beta.">${bars}${ticks}<text x="350" y="12" text-anchor="end" class="lbl small">Hz</text></svg>`;
}

function reactivitySvg() {
  const r = RS.reactivity;
  const lo = Math.log2(0.25);
  const hi = Math.log2(32);
  const x = (v) => 10 + ((Math.log2(v) - lo) / (hi - lo)) * 340;
  const ticks = [0.5, 1, 2, 4, 8, 16, 32].map((t) => `<text x="${x(t)}" y="58" text-anchor="middle" class="lbl small">${t}×</text>`).join("");
  return `<svg viewBox="0 0 360 66" class="range" role="img" aria-label="Ratio scale. 1 means no difference. Above 1 is the expected direction. This recording: ${r.ratio.toFixed(1)} times.">
    <rect x="${x(1)}" y="20" width="${x(32) - x(1)}" height="18" class="expected"/>
    <line x1="10" y1="29" x2="350" y2="29" class="axis"/>
    <line x1="${x(1)}" y1="14" x2="${x(1)}" y2="44" class="axis strong"/>
    <text x="${x(1) + 4}" y="12" class="lbl small">1 = no difference</text>
    <text x="${x(16)}" y="12" class="lbl small" text-anchor="middle">expected side</text>
    <circle cx="${x(r.ratio)}" cy="29" r="6" class="mark"/><text x="${x(r.ratio)}" y="50" text-anchor="middle" class="lbl strong">${r.ratio.toFixed(1)}×</text>
    ${ticks}</svg>`;
}

function accuracySvg() {
  const c = RS.chance;
  const x = (p) => 10 + p * 340;
  return `<svg viewBox="0 0 360 70" class="range" role="img" aria-label="Accuracy from 0 to 100 percent. Chance is 50 percent. With ${c.n_trials} windows, ${c.threshold_correct} correct (${Math.round(c.threshold_accuracy * 100)} percent) beats chance. This recording: ${Math.round(c.accuracy * 100)} percent.">
    <rect x="${x(c.threshold_accuracy)}" y="22" width="${x(1) - x(c.threshold_accuracy)}" height="16" class="expected"/>
    <line x1="10" y1="30" x2="350" y2="30" class="axis"/>
    <line x1="${x(c.chance_accuracy)}" y1="16" x2="${x(c.chance_accuracy)}" y2="44" class="axis strong"/>
    <text x="${x(c.chance_accuracy)}" y="12" text-anchor="middle" class="lbl small">chance ${Math.round(c.chance_accuracy * 100)}%</text>
    <line x1="${x(c.threshold_accuracy)}" y1="16" x2="${x(c.threshold_accuracy)}" y2="44" class="axis dashed"/>
    <text x="${x(c.threshold_accuracy)}" y="56" text-anchor="middle" class="lbl small">beats chance from ${c.threshold_correct}/${c.n_trials}</text>
    <circle cx="${x(c.accuracy)}" cy="30" r="6" class="mark"/><text x="${x(c.accuracy)}" y="12" text-anchor="middle" class="lbl strong">${c.n_correct}/${c.n_trials}</text>
    ${[0, 0.25, 0.5, 0.75, 1].map((t) => `<text x="${x(t)}" y="68" text-anchor="middle" class="lbl small">${t * 100}%</text>`).join("")}</svg>`;
}

function alphaStripSvg() {
  const b = data.baseline;
  const max = Math.max(...b.alpha_power_eyes_closed, ...b.alpha_power_eyes_open);
  const x = (v) => 70 + (v / max) * 280;
  const row = (vals, y, cls, label) =>
    `<text x="0" y="${y + 4}" class="lbl">${label}</text>${vals.map((v) => `<circle cx="${x(v).toFixed(1)}" cy="${y}" r="4" class="dot ${cls}"/>`).join("")}`;
  return `<svg viewBox="0 0 360 74" class="range" role="img" aria-label="Alpha power per one-second window. Eyes-closed windows sit mostly to the right of eyes-open windows.">
    ${row(b.alpha_power_eyes_closed, 16, "eyes_closed", "Closed")}
    ${row(b.alpha_power_eyes_open, 40, "eyes_open", "Open")}
    <line x1="${x(b.threshold)}" y1="4" x2="${x(b.threshold)}" y2="50" class="axis dashed"/>
    <text x="${x(b.threshold) + 3}" y="60" class="lbl small">threshold ${b.threshold.toFixed(1)} µV² = midpoint of the two means</text>
    <text x="350" y="72" text-anchor="end" class="lbl small">µV² per second →</text></svg>`;
}

function sclSvg() {
  return `<svg viewBox="0 0 360 60" class="range" role="img" aria-label="Skin conductance rises from eyes closed to eyes open. Direction only, no values.">
    <text x="20" y="40" class="lbl">Eyes closed</text><path d="M100,40 C160,40 200,18 260,16" class="arrow"/><polygon points="260,10 272,16 260,22" class="arrowhead"/>
    <text x="280" y="20" class="lbl">Eyes open</text><text x="180" y="54" text-anchor="middle" class="lbl small">direction only: absolute levels vary between people</text></svg>`;
}

const MEASURE_GRAPHIC = {
  frequency: bandsSvg,
  alpha_power: alphaStripSvg,
  reactivity: reactivitySvg,
  accuracy: accuracySvg,
  skin_conductance: sclSvg,
};

function numbersHtml() {
  const cards = RS.measures
    .map((m) => {
      const graphic = MEASURE_GRAPHIC[m.key] ? MEASURE_GRAPHIC[m.key]() : "";
      const note = m.key === "alpha_power" || m.key === "accuracy" ? `<p class="caveat">${esc(RS.threshold.text)}</p>` : "";
      return `<details class="box measure"${m.key === "reactivity" ? " open" : ""}><summary><h3>${esc(m.name)}</h3><span class="unit">${esc(m.unit)}</span></summary>
        <p><span class="src ${m.source}">${esc(RS.source_labels[m.source])}</span></p>
        ${graphic}${note}
        <dl class="modality-facts">
          <dt>What it is</dt><dd>${esc(m.what_it_is)}</dd>
          <dt>How it is computed</dt><dd>${esc(m.how_computed)}</dd>
          <dt>How to read it</dt><dd>${esc(m.how_to_read)}</dd>
          <dt>${esc(RANGE_LABEL[m.range_kind])}</dt><dd>${esc(m.expected)}</dd>
          <dt>What shifts it</dt><dd>${esc(m.what_shifts_it)}${refMarks(m.refs)}</dd>
        </dl></details>`;
    })
    .join("");
  return `<p>Each number in this journey, in plain terms, with the range you can expect and how that range is known.</p>
    ${cards}
    <p class="caveat">${esc(RS.band_caveat)}${refMarks(RS.band_refs)}</p>`;
}

function dichotomiesHtml() {
  const axis = state.studyAxis;
  const [a, b] = RS.axis_poles[axis];
  const rows = RS.dichotomies
    .filter((d) => d.axis === axis)
    .map((d) => `<div class="dich-row"><h4>${esc(d.modality)}</h4><div class="dich-pole"><span class="tag">${esc(a)}</span><p>${esc(d.first)}</p></div><div class="dich-pole"><span class="tag alt">${esc(b)}</span><p>${esc(d.second)}${refMarks(d.refs)}</p></div></div>`)
    .join("");
  const tabs = Object.entries(RS.axis_poles)
    .map(([k, [p, q]]) => `<button type="button" role="tab" data-axis="${k}" aria-selected="${k === axis}">${esc(p)} vs ${esc(q)}</button>`)
    .join("");
  return `<p>Two questions to ask of any measurement, whatever the sensor. Barry et al. (2007) answers both for EEG: the study is resting, and it found one domain-general change (alpha) and several domain-specific ones.${refMarks(["barry"])}</p>
    <div class="persp-tabs" role="tablist" aria-label="Dichotomy">${tabs}</div>
    ${rows}`;
}

function renderStudy() {
  $("study-tabs").innerHTML = STUDY_TABS.map(([k, label]) => `<button type="button" role="tab" data-study-tab="${k}" aria-selected="${state.studyTab === k}">${label}</button>`).join("");
  const body = { room: roomHtml, numbers: numbersHtml, dichotomies: dichotomiesHtml }[state.studyTab]();
  $("study-body").innerHTML = body;
}

function openStudy(tab) {
  if (tab) state.studyTab = tab;
  renderStudy();
  const dialog = $("study");
  if (!dialog.open) dialog.showModal();
}

function studyTextHtml() {
  const measures = RS.measures
    .map((m) => `<li><strong>${esc(m.name)} (${esc(m.unit)}).</strong> <em>${esc(RS.source_labels[m.source])}.</em> ${esc(m.what_it_is)} ${esc(m.how_computed)} ${esc(m.how_to_read)} <em>${esc(RANGE_LABEL[m.range_kind])}:</em> ${esc(m.expected)} What shifts it: ${esc(m.what_shifts_it)}${refMarks(m.refs)}</li>`)
    .join("");
  const r = RS.reactivity;
  const c = RS.chance;
  const dich = RS.dichotomies
    .map((d) => `<tr><td>${esc(d.modality)}</td><td>${esc(RS.axis_poles[d.axis].join(" vs "))}</td><td>${esc(d.first)}</td><td>${esc(d.second)}${refMarks(d.refs)}</td></tr>`)
    .join("");
  return `<section><h2>The Study Room</h2>
    ${studyCard(RS.study, "The Study: Barry et al. (2007)")}
    ${studyCard(RS.ageing, "Does It Hold With Age? Barry and De Blasio (2017)")}
    <h3>This Recording's Numbers</h3>
    <p>Mean alpha power ${r.mean_closed.toFixed(1)} µV² with the eyes closed and ${r.mean_open.toFixed(1)} µV² with them open, a ratio of ${r.ratio.toFixed(1)}. Accuracy ${c.n_correct} of ${c.n_trials}; chance is ${Math.round(c.chance_accuracy * 100)}%, and ${c.threshold_correct} of ${c.n_trials} is needed to beat it.${refMarks(["combrisson"])}</p><p>${esc(RS.threshold.text)}</p>
    <h3>What the Numbers Mean</h3><ul>${measures}</ul>
    <h3>Resting vs Task, Domain-General vs Domain-Specific</h3>
    <div class="table-wrap"><table><thead><tr><th>Modality</th><th>Axis</th><th>First pole</th><th>Second pole</th></tr></thead><tbody>${dich}</tbody></table></div></section>`;
}

function studyEntryHtml(stage) {
  if (stage.interaction === "waveform") {
    return `<div class="box study-entry"><h3>Step Into the Study</h3>
      <p>See how an eyes-closed and eyes-open recording is made, what each number means, and how resting studies compare with task studies.</p>
      <div class="study-buttons">
        <button type="button" class="primary" data-open-study="room">The recording</button>
        <button type="button" class="ghost" data-open-study="numbers">What the numbers mean</button>
        <button type="button" class="ghost" data-open-study="dichotomies">Resting vs task</button>
      </div></div>`;
  }
  if (stage.interaction === "infer") {
    return `<p><button type="button" class="linklike" data-open-study="numbers">What does ${pct(data.baseline.accuracy)} mean, and what counts as beating chance?</button></p>`;
  }
  return "";
}

function perspectivesHtml(stage) {
  const tabs = Object.entries(content.perspectives)
    .map(([k, p]) => `<button type="button" role="tab" data-persp="${k}" aria-selected="${state.perspective === k}">${esc(p.label)}</button>`)
    .join("");
  const p = content.perspectives[state.perspective];
  return `<div class="perspectives"><div class="persp-tabs" role="tablist" aria-label="Perspective">${tabs}</div>
    <p class="persp-q">${esc(p.question)}</p><p>${esc(stage.perspectives[state.perspective])}</p></div>`;
}

const ICON = {
  eye: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>',
  tag: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12V4h8l10 10-8 8z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="7.5" cy="8" r="1.6" fill="currentColor"/></svg>',
  print: '<svg viewBox="0 0 24 24" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 15c0-5 2.5-8 6-8s6 3 6 8"/><path d="M9 17c0-4 1-7 3-7s3 3 3 7"/><path d="M12 13v7"/><path d="M4 11c1-5 4-8 8-8s7 3 8 8"/></g></svg>',
  lock: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="11" width="14" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
  away: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20 20 4M9 4h11v11" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
};

function copyCard(c) {
  const readable = c.accuracy === null ? 0 : c.accuracy;
  const reads = c.accuracy === null ? "Eyes state unreadable" : `Eyes state readable at ${pct(c.accuracy)}, chance is 50%`;
  const labels = c.labels ? "Identifying labels attached" : "Identifying labels removed";
  const away = c.where === "Sensors" ? ICON.lock : ICON.away;
  return `<li class="copy-card">
    <div class="copy-name">${esc(c.label)}${state.pass > 1 ? ` <span class="caveat">pass ${c.pass}</span>` : ""}</div>
    <div class="copy-holder">${esc(c.holder)}</div>
    <div class="copy-icons">
      <span class="icon" title="${reads}" aria-label="${reads}">${ICON.eye}<span class="meter"><i style="width:${readable * 100}%"></i><b></b></span></span>
      <span class="icon${c.labels ? "" : " dim"}" title="${labels}" aria-label="${labels}">${ICON.tag}</span>
      <span class="icon on" title="Identifiable from the signal itself" aria-label="Identifiable from the signal itself">${ICON.print}</span>
      <span class="icon" title="${esc(c.recall)}" aria-label="${esc(c.recall)}">${away}</span>
    </div></li>`;
}

function copiesHtml() {
  const list = copies();
  if (!list.length) return `<div class="box copies"><h3>Your Exposure</h3><p class="caveat">You exist only inside the person.</p></div>`;
  return `<div class="box copies"><h3>Your Exposure: ${list.length} ${list.length === 1 ? "Copy" : "Copies"}</h3>
    <ol class="copy-cards">${list.map(copyCard).join("")}</ol>
    <p class="icon-key">${ICON.eye} readable ${ICON.tag} labels ${ICON.print} identifiable ${ICON.lock} deletable by holder ${ICON.away} stays with receiver${refMarks(["bagley"])}</p></div>`;
}

function exposureHud() {
  const list = copies();
  const holders = [...new Set(list.map((c) => c.holder.toLowerCase()))];
  $("hud").innerHTML = list.length
    ? `<strong>${list.length}</strong> ${list.length === 1 ? "copy" : "copies"} of you · held by ${esc(holders.join(" and "))}`
    : "You exist only inside the person";
}

function interactionHtml(stage) {
  const base = data.baseline;
  switch (stage.interaction) {
    case "waveform":
      return `<div class="choices levels" role="group" aria-label="Recorded condition">
          <button type="button" class="choice" data-eyes="eyes_closed" aria-pressed="${state.eyes === "eyes_closed"}">Eyes closed</button>
          <button type="button" class="choice" data-eyes="eyes_open" aria-pressed="${state.eyes === "eyes_open"}">Eyes open</button>
        </div>
        <p>${esc(stage.mechanism)}${refMarks(stage.mechanism_refs)}</p>
        <div class="legend"><span><i style="background:#fff"></i>Simulated sum of the neurons, slowed about 20 times</span><span><i style="background:var(--${state.eyes === "eyes_closed" ? "closed" : "open"})"></i>Recorded signal, at quarter speed</span></div>
        ${waveformSvg(data.waveform.eyes_closed, data.waveform.eyes_open)}
        <p class="caveat">${esc(data.waveform.note)} ${esc(data.source.subject)}, channel ${esc(data.source.channel)}, ${esc(data.source.dataset)}.${refMarks(["sanchez_gama"])}</p>`;
    case "threat":
      return `<div class="box threat"><span class="tag">${esc(stage.threat.kind)}</span><h3>${esc(stage.threat.title)}</h3><p>${esc(stage.threat.text)}${refMarks(stage.threat.refs)}</p><p class="caveat">${esc(stage.threat.definition)}</p></div>
        ${aboutDataHtml()}`;
    case "features": {
      const n = base.alpha_power_eyes_closed.length + base.alpha_power_eyes_open.length;
      return `<div class="metric-row">
          <div class="metric"><span>Windows</span><strong>${n}</strong></div>
          <div class="metric"><span>Window length</span><strong>${data.window_seconds} s</strong></div>
          <div class="metric"><span>Band</span><strong>${data.alpha_band_hz[0]} to ${data.alpha_band_hz[1]} Hz</strong></div>
        </div>
        <p class="caveat">On the left, ${data.waveform.seconds} seconds of the recorded trace, cut at each second. On the right, all ${n} windows as pillars. Heights are scaled to the tallest pillar.</p>`;
    }
    case "infer": {
      const wrong = misclassified(base);
      const errors = wrong.closedAsOpen + wrong.openAsClosed;
      return `<div class="legend"><span><i style="background:var(--closed)"></i>Labelled eyes closed</span><span><i style="background:var(--open)"></i>Labelled eyes open</span><span><i style="background:var(--wrong)"></i>Wrong label</span></div>
        <div class="metric-row">
          <div class="metric"><span>Seconds labelled correctly</span><strong>${wrong.total - errors} of ${wrong.total}</strong></div>
          <div class="metric"><span>Accuracy</span><strong>${pct(base.accuracy)}</strong></div>
        </div>
        <p>${wrong.closedAsOpen} eyes-closed seconds fell below the threshold and were labelled eyes open${wrong.openAsClosed ? `, and ${wrong.openAsClosed} eyes-open seconds were labelled eyes closed` : ""}. Back row: eyes closed. Front row: eyes open. The flat plane is the threshold. Tap a pillar to inspect it.</p>
        <p>${esc(stage.reading_note)}</p>
        <div class="box"><h3>${esc(stage.second_inference.title)}</h3><p>${esc(stage.second_inference.text)}${refMarks(stage.second_inference.refs)}</p></div>
        <p class="caveat">${esc(stage.caveat)}</p>`;
    }
    case "safeguard":
      return safeguardHtml(stage);
    default:
      return "";
  }
}

function safeguardHtml(stage) {
  const record = current();
  const choices = Object.entries(stage.safeguards)
    .map(([k, s]) => `<button type="button" class="choice" data-safeguard="${k}" aria-pressed="${record.safeguard === k}">${esc(s.label)}</button>`)
    .join("");
  let levels = "";
  let outcome = "";
  if (record.safeguard) {
    const s = stage.safeguards[record.safeguard];
    if (s.levels) {
      levels = `<div class="choices levels" role="group" aria-label="Level">${s.levels
        .map((l) => `<button type="button" class="choice" data-level="${l.value}" aria-pressed="${record.level === l.value}">${esc(l.label)}</button>`)
        .join("")}</div>`;
    }
    const o = outcomeFor(record);
    const reqAcc = o.requester ? pct(o.requester.accuracy) : "copy withheld";
    const failure = o.requester && o.requester.accuracy === null ? `<p class="caveat">${esc(o.requester.failure)}</p>` : "";
    const metadata =
      record.safeguard === "metadata"
        ? `<p class="caveat">Removed: ${Object.keys(data.metadata.illustrative_fields).filter((k) => !(k in data.metadata.after_removal)).join(", ")}. Kept: ${Object.keys(data.metadata.after_removal).join(", ")}. ${esc(data.metadata.note)}</p>`
        : "";
    outcome = `<p class="acts-on">Acts on: <strong>${esc(s.acts_on)}</strong></p>
      <div class="metric-row">
        <div class="metric"><span>Requester can read eyes state</span><strong>${reqAcc}</strong></div>
        <div class="metric"><span>Legitimate system reads eyes state</span><strong>${pct(o.legitimate.accuracy)}</strong></div>
      </div>
      <p class="caveat">Before any safeguard, both were ${pct(data.baseline.accuracy)}. With equal numbers of each condition, 50% is chance.</p>
      ${failure}
      <details class="box why"><summary><h3>What It Protects and Leaves Exposed</h3></summary>
        <p><strong>Protects.</strong> ${esc(s.protects)}</p>
        <p><strong>Leaves exposed.</strong> ${esc(s.does_not)}${record.safeguard === "metadata" ? refMarks(["bagley", "rothstein"]) : ""}</p>
        ${metadata}
        <p>${esc(stage.safeguards_note)}</p>
        <p>${esc(stage.future_inference.text)}${refMarks(stage.future_inference.refs)}</p>
      </details>`;
  }
  return `<div class="box threat"><span class="tag">Unauthorized access attempt</span><p>${esc(stage.request)}</p></div>
    <p class="caveat">Left: the legitimate system's inference. Right: the requester's. Each safeguard rebuilds both from measured data.</p>
    <div class="choices" role="group" aria-label="Safeguard">${choices}</div>${levels}${outcome}`;
}

function aboutDataHtml() {
  const a = content.about_data;
  return `<details class="box about"><summary><h3>${esc(a.title)}</h3></summary>
    <p>${esc(a.source)}${refMarks(["sanchez_gama"])}</p><p>${esc(a.excerpt)}</p>
    <p><strong>Research questions.</strong> ${esc(a.questions)}</p>
    <p><strong>Data shape.</strong> ${esc(a.shape)}${refMarks(["gramfort", "delorme"])}</p></details>`;
}

function loopSummaryHtml() {
  if (STAGES[state.stage].interaction !== "loop") return "";
  const sg = current().safeguard ? DECISION.safeguards[current().safeguard].label : "None chosen";
  const w = misclassified(data.baseline);
  const list = copies();
  const holders = [...new Set(list.map((c) => c.holder.toLowerCase()))];
  return `<div class="box ending"><h3>End of the Journey</h3><p>${esc(content.ending)}</p>
    <ul>
      <li>Labels: ${w.total - w.closedAsOpen - w.openAsClosed} of ${w.total} seconds correct.</li>
      <li>Safeguard at the junction: ${esc(sg)}.</li>
      <li>Copies of you: ${list.length}, held by ${esc(holders.join(" and "))}.</li>
    </ul>
    <p class="takeaway">${esc(content.takeaway)}</p>
    <button type="button" class="ghost" id="new-pass">Travel again as a new signal</button></div>`;
}

function renderPanel() {
  const i = state.stage;
  const stage = STAGES[i];
  const next = STAGES[(i + 1) % STAGES.length];
  const handoff =
    i < STAGES.length - 1
      ? `Passing on to ${i + 2}. ${esc(next.name)}: <strong>${esc(stage.handoff)}</strong>`
      : `Returning to 1. ${esc(STAGES[0].name)}: <strong>${esc(stage.handoff)}</strong>`;
  $("panel").innerHTML = `
    <p class="stage-kicker">Stage ${i + 1} of ${STAGES.length}</p>
    <h2>${esc(stage.name)}</h2>
    <p class="environment">${esc(stage.environment)}</p>
    <div class="you-are"><b>You are</b>${esc(stage.you_are)}</div>
    <p>${esc(stage.lesson)}${refMarks(stage.interaction === "threat" || stage.interaction === "infer" ? [] : stage.refs)}</p>
    <div class="box changed"><h3>What Changed</h3><p>${esc(stage.what_changed)}</p></div>
    ${interactionHtml(stage)}
    ${studyEntryHtml(stage)}
    ${anatomyHtml(stage)}
    ${loopSummaryHtml()}
    ${perspectivesHtml(stage)}
    ${copiesHtml()}
    <p class="handoff">${handoff}</p>
    <div class="nav">
      <button type="button" class="ghost" id="prev" ${i === 0 ? "disabled" : ""}>Back</button>
      <button type="button" class="primary" id="next">${i === STAGES.length - 1 ? "Finish" : "Next"}</button>
    </div>`;
}

const RAIL_GROUPS = [
  { label: "Measured", stages: [0, 1, 2] },
  { label: "Interpreted", stages: [3] },
  { label: "Acted Upon", stages: [4, 5, 6] },
];

function renderRail() {
  const step = (i) => {
    const s = STAGES[i];
    return `<div class="rail-step"><button type="button" class="rail-btn ${state.visited.has(i) ? "visited" : ""}" data-stage="${i}" ${i === state.stage ? 'aria-current="step"' : ""} title="${i + 1}. ${esc(s.name)}, passes on ${esc(s.handoff)}"><span class="rail-num">${i + 1}</span>${esc(s.name)}</button>${i < STAGES.length - 1 ? '<span class="rail-link" aria-hidden="true"></span>' : ""}</div>`;
  };
  $("rail").innerHTML = RAIL_GROUPS.map(
    (g) => `<div class="rail-group${g.stages.includes(state.stage) ? " active" : ""}"><span class="rail-group-label">${g.label}</span><div class="rail-group-steps">${g.stages.map(step).join("")}</div></div>`,
  ).join("");
}

function goTo(i) {
  if (i < 0 || i >= STAGES.length) return;
  state.previous = state.stage;
  state.stage = i;
  showInspect(null);
  state.visited.add(i);
  if (i >= 1) state.passes[state.pass].device = true;
  if (renderer) {
    packetCore.geometry = packetShapes[i];
    packetLight.color.set(WORLD[i].light);
    key.color.set(WORLD[i].light);
    syncHistoryRings();
    syncCopies();
    syncLandscapes(false);
    moveCamera(i);
  }
  renderRail();
  renderPanel();
  exposureHud();
  $("panel").scrollTop = 0;
  if (state.textView) renderTextView();
}

function setSafeguard(k) {
  const record = current();
  if (!record.safeguard || record.safeguard === "deny") copyFlightStart = clock.elapsedTime;
  record.safeguard = k;
  const levels = DECISION.safeguards[k].levels;
  record.level = levels ? levels[0].value : null;
  if (k === "deny") barDropStart = clock.elapsedTime;
  if (renderer) {
    syncCopies();
    syncLandscapes(true);
  }
  renderPanel();
  exposureHud();
}

function setLevel(value) {
  current().level = value;
  if (renderer) {
    syncCopies();
    syncLandscapes(true);
  }
  renderPanel();
  exposureHud();
}

function newPass() {
  state.pass += 1;
  state.passes[state.pass] = { device: false, safeguard: null, level: null };
  state.visited = new Set([0]);
  goTo(0);
}

// ------------------------------------------------------------------
// Text view: the same content and numbers, without the 3D scene.
// ------------------------------------------------------------------

function renderTextView() {
  const noiseRows = data.noise.filter((n) => n.sd_uv % 50 === 0).map((n) => `<tr><td>${n.sd_uv}</td><td>${pct(n.accuracy)}</td></tr>`).join("");
  const downRows = data.downsample.map((d) => `<tr><td>${d.factor}</td><td>${d.effective_rate_hz}</td><td>${pct(d.accuracy)}</td></tr>`).join("");
  const stages = STAGES.map((s, i) => {
    let extra = "";
    if (s.threat) extra += `<p><strong>${esc(s.threat.title)}.</strong> ${esc(s.threat.text)}${refMarks(s.threat.refs)} ${esc(s.threat.definition)}</p>`;
    if (s.interaction === "infer") {
      const w = misclassified(data.baseline);
      extra += `<p>Measured: ${w.total - w.closedAsOpen - w.openAsClosed} of ${w.total} one-second windows labelled correctly (${pct(data.baseline.accuracy)}).</p>`;
      extra += `<p>${esc(s.reading_note)}</p>`;
      extra += `<p><strong>${esc(s.second_inference.title)}.</strong> ${esc(s.second_inference.text)}${refMarks(s.second_inference.refs)}</p><p class="caveat">${esc(s.caveat)}</p>`;
    }
    if (s.safeguards) {
      extra += `<p><strong>Request:</strong> ${esc(s.request)}</p>`;
      extra += `<p><strong>${esc(s.future_inference.title)}.</strong> ${esc(s.future_inference.text)}${refMarks(s.future_inference.refs)}</p>`;
      extra += `<p>${esc(s.safeguards_note)}</p>`;
      extra += Object.values(s.safeguards).map((g) => `<h3>${esc(g.label)}</h3><p><strong>Acts on:</strong> ${esc(g.acts_on)}.</p><p><strong>Protects:</strong> ${esc(g.protects)}</p><p><strong>Leaves exposed:</strong> ${esc(g.does_not)}</p>`).join("");
      extra += `<h3>Measured Effect of Noise</h3><table><thead><tr><th>Noise SD (µV)</th><th>Eyes-state accuracy</th></tr></thead><tbody>${noiseRows}</tbody></table>`;
      extra += `<h3>Measured Effect of a Lower Sampling Rate</h3><table><thead><tr><th>One sample in</th><th>Rate (Hz)</th><th>Eyes-state accuracy</th></tr></thead><tbody>${downRows}</tbody></table>`;
    }
    const persp = Object.entries(content.perspectives).map(([k, p]) => `<li><strong>${esc(p.label)}.</strong> ${esc(s.perspectives[k])}</li>`).join("");
    return `<section><p class="stage-kicker">Stage ${i + 1} of ${STAGES.length}</p><h2>${esc(s.name)}</h2>
      <p><strong>You are:</strong> ${esc(s.you_are)}</p><p>${esc(s.lesson)}${refMarks(s.refs)}</p>
      <p><strong>What changed:</strong> ${esc(s.what_changed)}</p>${s.mechanism ? `<p>${esc(s.mechanism)}${refMarks(s.mechanism_refs)}</p>` : ""}${extra}${anatomyTextHtml(s)}${s.threat ? aboutDataHtml() : ""}
      <ul>${persp}</ul><p class="handoff">Passes on: <strong>${esc(s.handoff)}</strong></p></section>`;
  }).join("");
  $("text-view").innerHTML = `<h1>${esc(content.title)}</h1><p>${esc(content.subtitle)}</p><p class="caveat">${esc(content.audience)}</p>
    <p>${esc(content.beginning)}</p>
    ${stages}${studyTextHtml()}<section><h2>End of the Journey</h2><p>${esc(content.ending)}</p><p class="takeaway">${esc(content.takeaway)}</p></section>`;
}

function setTextView(on) {
  state.textView = on;
  $("toggle-text").setAttribute("aria-pressed", String(on));
  $("toggle-text").textContent = on ? "3D view" : "Text view";
  $("text-view").hidden = !on;
  $("panel").hidden = on;
  $("hud").hidden = on;
  showInspect(null);
  if (on) renderTextView();
}

// ------------------------------------------------------------------
// Evidence panel
// ------------------------------------------------------------------

function renderRefs() {
  $("refs-source").textContent = `Every number in this experience is computed from ${data.source.dataset} (${data.source.licence}) by ${data.generated_by}.`;
  $("refs-list").innerHTML = refOrder
    .map((k) => `<li id="ref-${k}">${esc(REFS[k].text)}${REFS[k].url ? ` <a href="${REFS[k].url}" target="_blank" rel="noopener">${esc(REFS[k].url)}</a>` : ""}</li>`)
    .join("");
}

function openRefs(keyToShow) {
  const dialog = $("refs");
  if (!dialog.open) dialog.showModal();
  dialog.querySelectorAll("li").forEach((li) => li.classList.remove("flash"));
  if (keyToShow) {
    const item = $(`ref-${keyToShow}`);
    item.classList.add("flash");
    item.scrollIntoView({ block: "center" });
  }
}

// ------------------------------------------------------------------
// Events
// ------------------------------------------------------------------

// Regions and electrodes on the anatomy drawings, and the system list.
function studyPick(target) {
  const el = target.closest("[data-open-study], [data-study-tab], [data-axis]");
  if (!el) return false;
  if (el.dataset.openStudy) openStudy(el.dataset.openStudy);
  if (el.dataset.studyTab) {
    state.studyTab = el.dataset.studyTab;
    renderStudy();
  }
  if (el.dataset.axis) {
    state.studyAxis = el.dataset.axis;
    renderStudy();
  }
  return true;
}

function anatomyPick(target) {
  const el = target.closest("[data-system], [data-electrode], [data-modality]");
  if (!el) return false;
  if (el.dataset.modality) state.modality = el.dataset.modality;
  if (el.dataset.system) state.anatomySystem = el.dataset.system;
  if (el.dataset.electrode) state.electrode = el.dataset.electrode;
  const scroll = $("panel").scrollTop;
  renderPanel();
  $("panel").scrollTop = scroll;
  return true;
}

document.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && e.target.matches?.("g[data-system], g[data-electrode]")) {
    e.preventDefault();
    const selector = e.target.dataset.system ? `g[data-system="${e.target.dataset.system}"]` : `g[data-electrode="${e.target.dataset.electrode}"]`;
    anatomyPick(e.target);
    document.querySelector(`#panel ${selector}`)?.focus();
  }
});

document.addEventListener("click", (e) => {
  if (studyPick(e.target)) return;
  if (anatomyPick(e.target)) return;
  const t = e.target.closest("button");
  if (!t) return;
  if (t.dataset.ref) return openRefs(t.dataset.ref);
  if (t.dataset.stage) return goTo(Number(t.dataset.stage));
  if (t.dataset.persp) {
    state.perspective = t.dataset.persp;
    return renderPanel();
  }
  if (t.dataset.safeguard) return setSafeguard(t.dataset.safeguard);
  if (t.dataset.level) return setLevel(Number(t.dataset.level));
  if (t.dataset.eyes) {
    state.eyes = t.dataset.eyes;
    if ($("study").open) renderStudy();
    return renderPanel();
  }
  switch (t.id) {
    case "next":
      if (state.stage === STAGES.length - 1) return $("new-pass")?.focus();
      return goTo(state.stage + 1);
    case "prev":
      return goTo(state.stage - 1);
    case "new-pass":
      return newPass();
    case "begin":
      $("intro").hidden = true;
      return $("next").focus({ preventScroll: true });
    case "begin-text":
      $("intro").hidden = true;
      return setTextView(true);
    case "toggle-text":
      return setTextView(!state.textView);
    case "open-refs":
      return openRefs();
    case "close-refs":
      return $("refs").close();
    case "close-study":
      return $("study").close();
    case "open-study-intro":
      return openStudy("room");
  }
});

document.addEventListener("keydown", (e) => {
  if ($("refs").open || $("study").open || !$("intro").hidden || e.target.matches("input")) return;
  if (e.key === "ArrowRight") goTo(state.stage + 1);
  if (e.key === "ArrowLeft") goTo(state.stage - 1);
  if (/^[1-7]$/.test(e.key)) goTo(Number(e.key) - 1);
});

// ------------------------------------------------------------------
// Start
// ------------------------------------------------------------------

$("intro-sub").textContent = content.subtitle;
$("intro-audience").textContent = content.audience;
$("intro-beginning").textContent = content.beginning;
renderRefs();

if (renderer) {
  window.addEventListener("resize", resize);
  resize();
  goTo(0);
  moveCamera(0, true);
  requestAnimationFrame(frame);
} else {
  // No WebGL: the text view is the experience.
  $("toggle-text").hidden = true;
  canvas.hidden = true;
  goTo(0);
  setTextView(true);
}
