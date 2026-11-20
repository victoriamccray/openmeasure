// Become the Signal: EEG, from signal to inference.
//
// Two files drive everything here. content.json holds the words, with a
// citation key on every claim taken from a source. data.json holds the
// numbers, every one computed by modules/signal_pipeline/core from a real
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
  pass: 1,
  // One entry per pass: whether the device recorded, and the safeguard
  // chosen at the junction with its level. Earlier passes are never
  // cleared, so their copies stay on the map.
  passes: { 1: { device: false, safeguard: null, level: null } },
  textView: false,
};

// ------------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const esc = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (value) => (value === null || value === undefined ? "fails" : `${Math.round(value * 100)}%`);

// References are numbered by first appearance along the journey, so the
// numbers read in order as the learner moves forward.
const refOrder = [];
for (const stage of STAGES) {
  const keys = [...(stage.refs || []), ...(stage.threat?.refs || []), ...(stage.second_inference?.refs || [])];
  for (const key of keys) if (!refOrder.includes(key)) refOrder.push(key);
}
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
        recall: "Only by whoever runs the device",
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
        recall: "No. It has left the system",
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

// A trace drawn from real samples, centred on the origin.
function traceLine(values, width, height, color, maxAbs) {
  const n = values.length;
  const points = values.map((v, i) => new THREE.Vector3((i / (n - 1) - 0.5) * width, (v / maxAbs) * height, 0));
  return line(points, color);
}

// A row of pillars from real alpha powers. Heights are scaled to the
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

function ghost(scale = 1) {
  const mesh = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.55 * scale, 1),
    new THREE.MeshBasicMaterial({ color: COLORS.copy, wireframe: true, transparent: true, opacity: 0.75 }),
  );
  return mesh;
}

const stations = STAGES.map((_, i) => {
  const group = new THREE.Group();
  group.position.copy(stationPos(i));
  scene.add(group);
  return group;
});

const waveMax = Math.max(
  ...data.waveform.eyes_closed.map(Math.abs),
  ...data.waveform.eyes_open.map(Math.abs),
);

function seeded(seed) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) / 2147483647);
}

// 1. Signals: neural tissue, with the real trace running through it.
(function buildSignals() {
  const g = stations[0];
  const rand = seeded(7);
  const count = 160;
  const cells = new THREE.InstancedMesh(new THREE.SphereGeometry(0.22, 12, 12), glow(0xff6f91, 0.5), count);
  const m = new THREE.Matrix4();
  const positions = [];
  for (let i = 0; i < count; i++) {
    const p = new THREE.Vector3((rand() - 0.5) * 34, (rand() - 0.2) * 14, (rand() - 0.5) * 26 - 6);
    positions.push(p);
    const s = 0.5 + rand() * 1.4;
    m.compose(p, new THREE.Quaternion(), new THREE.Vector3(s, s, s));
    cells.setMatrixAt(i, m);
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
  const closed = traceLine(data.waveform.eyes_closed, 18, 1.6, COLORS.closed, waveMax);
  closed.position.y = 3.4;
  const open = traceLine(data.waveform.eyes_open, 18, 1.6, COLORS.open, waveMax);
  open.position.y = 0.6;
  g.add(closed, open);
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
})();

// 4. Inference: every window as a pillar, the threshold as a plane.
(function buildInference() {
  const g = stations[3];
  const grid = new THREE.GridHelper(24, 24, 0x3a3570, 0x1f1c40);
  g.add(grid);
  g.add(pillarSet(data.baseline));
})();

// 5. Decision: the junction, the system road, and the requester.
const requesterGroup = new THREE.Group();
const requesterGhosts = new THREE.Group();
const branchPath = new THREE.Group();
const denyBar = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.25, 0.25), glow(COLORS.wrong, 1));
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
  const building = new THREE.Mesh(new THREE.BoxGeometry(5, 9, 5), matte(0x2b2414));
  building.position.set(13, 4.5, -8);
  g.add(building);
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 3; col++) {
      const win = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8), glow(0xffc46b, 0.35));
      win.position.set(11.8 + col * 1.2, 1.6 + row * 1.6, -5.49);
      g.add(win);
    }
  }
  requesterGroup.position.set(13, 9.2, -8);
  requesterGroup.scale.setScalar(0.7);
  requesterGhosts.position.set(9.5, 1.4, -4.8);
  denyBar.position.set(9, 1.6, -4.4);
  denyBar.rotation.y = -0.55;
  g.add(requesterGroup, requesterGhosts, branchPath, denyBar);
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

function syncCopies() {
  deviceGhosts.clear();
  requesterGhosts.clear();
  branchPath.clear();
  requesterGroup.clear();
  copies().forEach((c) => {
    const g = ghost(0.8);
    g.userData.inspect = `${c.label}. Held by: ${c.holder.toLowerCase()}. Can the person recall it? ${c.recall}.`;
    if (c.where === "Sensors") {
      g.position.set(deviceGhosts.children.length * 1.1, 0, 0);
      deviceGhosts.add(g);
    } else {
      g.position.set(requesterGhosts.children.length * 1.1, 0, 0);
      requesterGhosts.add(g);
    }
  });
  const outcome = safeguardOutcome();
  denyBar.visible = current().safeguard === "deny";
  if (outcome?.requester) {
    branchPath.add(line([new THREE.Vector3(0, 0.6, 2), new THREE.Vector3(9.5, 0.6, -4.8), new THREE.Vector3(13, 0.6, -6)], COLORS.copy, 0.8));
  }
  if (outcome?.requester && outcome.requester.accuracy !== null) {
    requesterGroup.add(pillarSet(outcome.requester, { spacing: 0.5, height: 5 }));
  }
}

// ------------------------------------------------------------------
// Camera, travel, and inspection
// ------------------------------------------------------------------

const view = { yaw: 0, pitch: 0 };
const trip = { from: 0, to: 0, t: 1, seconds: 2 };
const currentLook = new THREE.Vector3();
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

function framing(i) {
  const station = stationPos(i);
  // The panel covers the right of the screen on wide layouts and the
  // bottom on narrow ones, so the subject is framed away from it.
  const look = narrow.matches ? station.clone().add(new THREE.Vector3(0, -5.5, 0)) : station.clone().add(new THREE.Vector3(5, 2, 0));
  const eye = station.clone().add(new THREE.Vector3(0, narrow.matches ? 7 : 5.5, narrow.matches ? 22 : 17));
  if (i === 4) look.add(new THREE.Vector3(3, 1, -2));
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
  if (trip.t >= 1 || trip.from === trip.to) return { eye, look, packetAt: rest };

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
  const { eye, look, packetAt } = pose();
  currentLook.copy(look);

  // Drag orbits the eye around what it is looking at.
  const offset = eye.clone().sub(look);
  offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), view.yaw);
  const side = new THREE.Vector3().crossVectors(offset, new THREE.Vector3(0, 1, 0)).normalize();
  offset.applyAxisAngle(side, view.pitch);
  camera.position.copy(look).add(offset);
  camera.lookAt(look);

  packet.position.copy(packetAt);
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

function waveformSvg(closed, open, maxAbs = waveMax) {
  const w = 360;
  const h = 46;
  const path = (values) => {
    const n = values.length;
    return values.map((v, i) => `${i === 0 ? "M" : "L"}${((i / (n - 1)) * w).toFixed(1)},${(h / 2 - (v / maxAbs) * (h / 2 - 2)).toFixed(1)}`).join("");
  };
  const row = (values, color, label) =>
    `<div class="wave"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}"><path d="${path(values)}" fill="none" stroke="${color}" stroke-width="1.2"/></svg><p class="wave-label">${label}</p></div>`;
  return row(closed, "var(--closed)", "Eyes closed: alpha is strong") + row(open, "var(--open)", "Eyes open: alpha is weak");
}

function perspectivesHtml(stage) {
  const tabs = Object.entries(content.perspectives)
    .map(([k, p]) => `<button type="button" role="tab" data-persp="${k}" aria-selected="${state.perspective === k}">${esc(p.label)}</button>`)
    .join("");
  const p = content.perspectives[state.perspective];
  return `<div class="perspectives"><div class="persp-tabs" role="tablist" aria-label="Perspective">${tabs}</div>
    <p class="persp-q">${esc(p.question)}</p><p>${esc(stage.perspectives[state.perspective])}</p></div>`;
}

function copiesHtml() {
  const list = copies();
  if (!list.length) return `<div class="box copies"><h3>Your Exposure</h3><p class="caveat">No copies yet. You exist only inside the person.</p></div>`;
  const items = list
    .map(
      (c) => `<li><strong>${esc(c.label)}</strong>${state.pass > 1 ? ` (pass ${c.pass})` : ""}<br>
        Held by ${esc(c.holder.toLowerCase())}. Eyes state readable: ${pct(c.accuracy)}.
        Identifying labels attached: ${c.labels ? "yes" : "no"}. Identifiable from the signal itself: possible.${refMarks(["bagley"])}
        Recall by the person: ${esc(c.recall)}.</li>`,
    )
    .join("");
  return `<div class="box copies"><h3>Your Exposure: ${list.length} ${list.length === 1 ? "Copy" : "Copies"}</h3><ol>${items}</ol></div>`;
}

function exposureHud() {
  const list = copies();
  const holders = [...new Set(list.map((c) => c.holder.toLowerCase()))];
  $("hud").innerHTML = list.length
    ? `<strong>${list.length}</strong> ${list.length === 1 ? "copy" : "copies"} of you · held by ${esc(holders.join(" and "))}`
    : "No copies of you yet";
}

function interactionHtml(stage) {
  const base = data.baseline;
  switch (stage.interaction) {
    case "waveform":
      return `${waveformSvg(data.waveform.eyes_closed, data.waveform.eyes_open)}
        <p class="caveat">${esc(data.waveform.note)} ${esc(data.source.subject)}, channel ${esc(data.source.channel)}, ${esc(data.source.dataset)}.${refMarks(["sanchez_gama"])}</p>`;
    case "threat":
      return `<div class="box threat"><span class="tag">${esc(stage.threat.kind)}</span><h3>${esc(stage.threat.title)}</h3><p>${esc(stage.threat.text)}${refMarks(stage.threat.refs)}</p></div>`;
    case "features": {
      const n = base.alpha_power_eyes_closed.length + base.alpha_power_eyes_open.length;
      return `<div class="metric-row">
          <div class="metric"><span>Windows</span><strong>${n}</strong></div>
          <div class="metric"><span>Window length</span><strong>${data.window_seconds} s</strong></div>
          <div class="metric"><span>Band</span><strong>${data.alpha_band_hz[0]} to ${data.alpha_band_hz[1]} Hz</strong></div>
        </div>
        <p class="caveat">On the left, ${data.waveform.seconds} seconds of the real trace, cut at each second. On the right, all ${n} windows as pillars. Heights are scaled to the tallest pillar.</p>`;
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
    const reqAcc = o.requester ? pct(o.requester.accuracy) : "no copy";
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
      <div class="outcome"><div class="box protects"><h3>What It Protects</h3><p>${esc(s.protects)}</p></div>
      <div class="box does-not"><h3>What It Does Not</h3><p>${esc(s.does_not)}${record.safeguard === "metadata" ? refMarks(["bagley", "rothstein"]) : ""}</p></div></div>
      ${metadata}`;
  }
  return `<div class="box threat"><span class="tag">Unauthorized access attempt</span><p>${esc(stage.request)}</p></div>
    <p class="caveat">${esc(stage.safeguards_note)}</p>
    <div class="choices" role="group" aria-label="Safeguard">${choices}</div>${levels}${outcome}`;
}

function loopSummaryHtml() {
  if (STAGES[state.stage].interaction !== "loop") return "";
  const sg = current().safeguard ? DECISION.safeguards[current().safeguard].label : "No safeguard chosen";
  return `<div class="box"><h3>Your Pass</h3><p>Safeguard: ${esc(sg)}.</p>
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
    ${loopSummaryHtml()}
    ${perspectivesHtml(stage)}
    ${copiesHtml()}
    <p class="handoff">${handoff}</p>
    <div class="nav">
      <button type="button" class="ghost" id="prev" ${i === 0 ? "disabled" : ""}>Back</button>
      <button type="button" class="primary" id="next">${i === STAGES.length - 1 ? "Finish" : "Next"}</button>
    </div>`;
}

function renderRail() {
  $("rail").innerHTML = STAGES.map(
    (s, i) => `<div class="rail-step"><button type="button" class="rail-btn ${state.visited.has(i) ? "visited" : ""}" data-stage="${i}" ${i === state.stage ? 'aria-current="step"' : ""} title="${i + 1}. ${esc(s.name)}, passes on ${esc(s.handoff)}"><span class="rail-num">${i + 1}</span>${esc(s.name)}</button>${i < STAGES.length - 1 ? '<span class="rail-link" aria-hidden="true"></span>' : ""}</div>`,
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
  record.safeguard = k;
  const levels = DECISION.safeguards[k].levels;
  record.level = levels ? levels[0].value : null;
  if (renderer) syncCopies();
  renderPanel();
  exposureHud();
}

function setLevel(value) {
  current().level = value;
  if (renderer) syncCopies();
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
    if (s.threat) extra += `<p><strong>${esc(s.threat.title)}.</strong> ${esc(s.threat.text)}${refMarks(s.threat.refs)}</p>`;
    if (s.interaction === "infer") {
      const w = misclassified(data.baseline);
      extra += `<p>Measured: ${w.total - w.closedAsOpen - w.openAsClosed} of ${w.total} one-second windows labelled correctly (${pct(data.baseline.accuracy)}).</p>`;
      extra += `<p>${esc(s.reading_note)}</p>`;
      extra += `<p><strong>${esc(s.second_inference.title)}.</strong> ${esc(s.second_inference.text)}${refMarks(s.second_inference.refs)}</p><p class="caveat">${esc(s.caveat)}</p>`;
    }
    if (s.safeguards) {
      extra += `<p><strong>Request:</strong> ${esc(s.request)}</p>`;
      extra += `<p>${esc(s.safeguards_note)}</p>`;
      extra += Object.values(s.safeguards).map((g) => `<h3>${esc(g.label)}</h3><p><strong>Acts on:</strong> ${esc(g.acts_on)}.</p><p><strong>Protects:</strong> ${esc(g.protects)}</p><p><strong>Does not:</strong> ${esc(g.does_not)}</p>`).join("");
      extra += `<h3>Measured Effect of Noise</h3><table><thead><tr><th>Noise SD (µV)</th><th>Eyes-state accuracy</th></tr></thead><tbody>${noiseRows}</tbody></table>`;
      extra += `<h3>Measured Effect of a Lower Sampling Rate</h3><table><thead><tr><th>One sample in</th><th>Rate (Hz)</th><th>Eyes-state accuracy</th></tr></thead><tbody>${downRows}</tbody></table>`;
    }
    const persp = Object.entries(content.perspectives).map(([k, p]) => `<li><strong>${esc(p.label)}.</strong> ${esc(s.perspectives[k])}</li>`).join("");
    return `<section><p class="stage-kicker">Stage ${i + 1} of ${STAGES.length}</p><h2>${esc(s.name)}</h2>
      <p><strong>You are:</strong> ${esc(s.you_are)}</p><p>${esc(s.lesson)}${refMarks(s.refs)}</p>
      <p><strong>What changed:</strong> ${esc(s.what_changed)}</p>${extra}
      <ul>${persp}</ul><p class="handoff">Passes on: <strong>${esc(s.handoff)}</strong></p></section>`;
  }).join("");
  $("text-view").innerHTML = `<h1>${esc(content.title)}</h1><p>${esc(content.subtitle)}</p><p class="caveat">${esc(content.audience)}</p>
    ${stages}<section><h2>Takeaway</h2><p>${esc(content.takeaway)}</p></section>`;
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
    .map((k) => `<li id="ref-${k}">${esc(content.references[k].text)} <a href="${content.references[k].url}" target="_blank" rel="noopener">${esc(content.references[k].url)}</a></li>`)
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

document.addEventListener("click", (e) => {
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
  }
});

document.addEventListener("keydown", (e) => {
  if ($("refs").open || !$("intro").hidden || e.target.matches("input")) return;
  if (e.key === "ArrowRight") goTo(state.stage + 1);
  if (e.key === "ArrowLeft") goTo(state.stage - 1);
  if (/^[1-7]$/.test(e.key)) goTo(Number(e.key) - 1);
});

// ------------------------------------------------------------------
// Start
// ------------------------------------------------------------------

$("intro-sub").textContent = content.subtitle;
$("intro-audience").textContent = content.audience;
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
