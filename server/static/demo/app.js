"use strict";
// Demo UI (review use only). Calls /demo/api/* on this same server; nothing is downloaded.

const REASONS = {
  EXPIRED: "the one-time challenge had expired (60 s) or was never issued",
  REPLAY: "this exact evidence was already used once",
  MODE_MISMATCH: "the request tried to switch the crypto mode (downgrade attempt)",
  TAMPER: "the encrypted package was changed on the way",
  TX_MISMATCH: "the evidence belongs to a different transaction or challenge",
  HASH_MISMATCH: "the video is not the one the device signed",
  SIG_INVALID: "the device signature is not valid for the enrolled device",
  NO_MATCH: "the face does not match the enrolled person",
  PAD_FAIL: "looks like a photo, screen or mask held to the camera",
  MORPH_FLAG: "the enrolled photo looks like a blend of two people",
  DEEPFAKE_FAIL: "the face looks generated or swapped by AI",
  SCRIPT_FAIL: "the requested actions (blink / turn) were not seen in order",
  INJECTION_FAIL: "the video came from a virtual camera, not a real webcam",
};
const CHECK_OF = { EXPIRED: 1, REPLAY: 1, MODE_MISMATCH: 2, TAMPER: 3, TX_MISMATCH: 4, HASH_MISMATCH: 5, SIG_INVALID: 6 };
const MODULES = [
  ["match", "Face matcher", "accept if ≥"], ["m1_pad", "M1 presentation attack", "reject if >"],
  ["m2_morph", "M2 morph", "reject if >"], ["m3_deepfake", "M3 deepfake", "reject if >"],
  ["m4_replay", "M4 live actions", "reject if >"], ["m5_injection", "M5 injection", "reject if >"],
];
const STATUS_MODULES = [["face_detector", "Face detector"], ["matcher", "Matcher"], ["m1_pad", "M1 photo/screen"],
  ["m2_morph", "M2 morph"], ["m3_deepfake", "M3 deepfake"], ["m4_replay", "M4 live actions"], ["m5_injection", "M5 injection"]];

const STEPS = [
  { id: "enrol", title: "1. Enrol", webcam: true,
    what: "Registers you: one reference photo and a device key for each crypto mode.",
    how: "A webcam window opens; press SPACE. Your device keys are created on this laptop and the server's keys are pinned." },
  { id: "verify", title: "2. Verify me (genuine)", webcam: true, mode: true,
    what: "A normal login or payment by the real person: it should be ACCEPTED.",
    how: "Do the two actions shown in the webcam window. The evidence is signed, encrypted and checked; every module scores it." },
  { id: "photo", title: "3. Photo attack", webcam: true, mode: true,
    what: "Someone holds your photo on a phone to the camera: it should be REJECTED.",
    how: "Hold a phone photo of your face up to the webcam during the countdown and recording. M1 spots the screen and M4 sees no blinks or turns." },
  { id: "replay", title: "4. Replay attacks",
    what: "An attacker re-uses or alters captured evidence: all three attempts are REJECTED.",
    how: "Re-sends your last accepted package unchanged, then flips one bit of it, then re-binds it to a 50,000 INR payment." },
  { id: "morph", title: "5. Morph detection",
    what: "A morphed enrolment photo (two people blended) is caught; a genuine pair is not.",
    how: "M2 compares the enrolled photo with a live photo of the same person (FRLL dataset). Illustration only: this pair was in M2's training data." },
  { id: "deepfake", title: "6. Deepfake detection",
    what: "AI-generated faces are flagged; real faces pass.",
    how: "M3 (GenD) scores 3 real and 3 fake faces from the OpenForensics test set. At the strict setting some fakes can still pass." },
  { id: "injection", title: "7. Injection (OBS)", webcam: true, obs: true,
    what: "A fake video fed through a virtual camera is caught.",
    how: "Verifies through OBS Virtual Camera instead of the real webcam; M5 recognises the virtual camera." },
  { id: "verdict", title: "8. Signed verdict check (our novelty)",
    what: "The AI's decision is itself signed evidence: anyone can re-check it, and any edit is detected.",
    how: "Verifies the signature of your last accepted verdict offline, then edits one score in a copy of the database and verifies again." },
  { id: "crypto", title: "9. Crypto overhead",
    what: "What post-quantum cryptography costs compared with classical cryptography.",
    how: "Median time per operation and extra bytes per request, from the benchmark file eval/results/crypto_bench.csv." },
];

let status = null;
let busy = false;

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, html = "") => { const e = document.createElement(tag); Object.assign(e, attrs); if (html) e.innerHTML = html; return e; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (x, d = 3) => (x === null || x === undefined || x === "" || isNaN(Number(x)) ? "–" : Number(x).toFixed(d));
const ms = (us) => (us === undefined ? "" : `<span class="t">${(us / 1000).toFixed(us < 10000 ? 2 : 0)} ms</span>`);
const mode = () => document.querySelector('input[name="mode"]:checked').value;

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  let body = null;
  try { body = await res.json(); } catch (e) { console.error("non-JSON reply from", path, e); }
  if (!res.ok && !body) throw new Error(`HTTP ${res.status}`);
  return body;
}

// ---- status ------------------------------------------------------------------------------
async function loadStatus() {
  try {
    status = await api("/demo/api/status");
  } catch (e) {
    console.error(e);
    $("#status-body").innerHTML = `<div class="error">The server is not answering. Start it with start_demo.bat.</div>`;
    return;
  }
  const t = status.thresholds;
  const lights = STATUS_MODULES.map(([k, label]) => {
    const v = status.model_versions[k]; const stub = v === "stub";
    return `<div class="light"><span class="dot ${stub ? "stub" : "real"}"></span><b>${label}</b>
      <span class="tag">${stub ? "STUB" : "real"}</span><span class="muted small">${esc(v)}</span></div>`;
  }).join("");
  $("#status-body").innerHTML = `<div class="lights">${lights}</div>
    <div>Crypto modes: <b>${status.allowed_modes.join(" and ")}</b> (default ${status.default_mode}).
    Thresholds <span class="tag pre">preliminary</span> <span class="muted small">${esc(t.version)} — calibrated on validation data only</span></div>`;
  renderCards();
}

// ---- cards -------------------------------------------------------------------------------
function renderCards() {
  const root = $("#cards");
  if (root.childElementCount) { updateDisabled(); return; }
  for (const s of STEPS) {
    const card = el("div", { className: "card", id: `card-${s.id}` });
    card.innerHTML = `<h3>${s.title}</h3>
      <p class="what"><b>What this shows:</b> ${s.what}</p>
      <p class="how"><b>How it works:</b> ${s.how}</p>
      <div class="actions"><button class="run">Run</button><span class="spinner"></span><span class="note"></span></div>
      <div class="result"></div>`;
    $("button.run", card).addEventListener("click", () => runStep(s));
    root.appendChild(card);
  }
  updateDisabled();
}

function obsMissing(s) { return s.obs && !(status && status.obs_camera); }

function updateDisabled() {
  for (const s of STEPS) {
    const card = $(`#card-${s.id}`);
    const off = obsMissing(s);
    card.classList.toggle("disabled", off);
    $("button.run", card).disabled = busy || off;
    if (off && !$(".result", card).innerHTML) $(".result", card).innerHTML = `<span class="badge skip">OBS not installed</span>`;
  }
  document.querySelectorAll("button.retry, button.reset").forEach((b) => (b.disabled = busy));
}

async function runStep(s) {
  if (busy) return;
  busy = true; updateDisabled();
  const card = $(`#card-${s.id}`);
  card.classList.add("busy");
  $(".note", card).textContent = s.webcam ? "Look at the webcam window on this laptop…" : "Running…";
  $(".result", card).innerHTML = "";
  let body = null;
  try {
    body = await api(`/demo/api/run/${s.id}`, { method: "POST", body: JSON.stringify({ mode: mode() }) });
  } catch (e) {
    console.error(`step ${s.id} failed`, e);
    body = { ok: false, message: "The server did not answer." };
  }
  card.classList.remove("busy");
  $(".note", card).textContent = "";
  busy = false;
  showResult(s, card, body || { ok: false, message: "No answer." });
  updateDisabled();
  loadStatus();
}

function showResult(s, card, body) {
  const out = $(".result", card);
  if (body.skipped) { out.innerHTML = `<span class="badge skip">Skipped</span><p>${esc(body.message)}</p>`; return; }
  if (!body.ok) {
    console.error(`step ${s.id}:`, body);
    out.innerHTML = `<div class="error">${esc(body.message || "This step failed.")}</div>
      <div class="actions"><button class="retry">Retry</button><button class="reset">Reset demo</button></div>`;
    $("button.retry", out).addEventListener("click", () => runStep(s));
    $("button.reset", out).addEventListener("click", resetDemo);
    return;
  }
  const render = { enrol: showEnrol, verify: showVerdict, photo: showVerdict, injection: showVerdict, replay: showReplay,
    morph: showMorph, deepfake: showDeepfake, verdict: showTamper, crypto: showCrypto }[s.id];
  out.innerHTML = render(body);
}

function showEnrol(b) { return `<span class="badge ok">ENROLLED</span><p>${esc(b.message)}</p>`; }

function badge(decision) {
  const ok = decision === "accepted";
  return `<span class="badge ${ok ? "ok" : "bad"}">${ok ? "ACCEPTED" : "REJECTED"}</span>`;
}
function reasonLine(code) {
  if (!code) return `<p class="reason">Every check passed.</p>`;
  return `<p class="reason">Reason <code>${esc(code)}</code>: ${esc(REASONS[code] || "")}.</p>`;
}

function scoreTable(scores, thresholds) {
  const rows = MODULES.map(([k, label, rule]) => {
    const v = scores[k]; const thr = Number(thresholds[k]);
    if (v === null || v === undefined) return `<tr><td>${label}</td><td class="num">–</td><td class="num">${num(thr, 4)}</td><td class="muted small">not run</td></tr>`;
    const val = Number(v);
    const bad = k === "match" ? val < thr : val > thr;
    const pos = (x) => Math.max(0, Math.min(1, k === "match" ? (x + 1) / 2 : x)) * 100; // cosine is -1..1
    return `<tr><td>${label}</td><td class="num">${num(val)}</td><td class="num">${num(thr, 4)} <span class="muted small">${rule}</span></td>
      <td><div class="bar"><div class="fill ${bad ? "bad" : ""}" style="width:${pos(val)}%"></div><div class="thr" style="left:${pos(thr)}%"></div></div></td></tr>`;
  }).join("");
  return `<table><tr><th>module</th><th>score</th><th>threshold</th><th>score vs threshold (black line)</th></tr>${rows}</table>`;
}

function trace(rec, req, mode) {
  const t = (req && req.timing) || {};
  const failed = CHECK_OF[rec.reason_code] || 0;
  const pq = mode === "PQ";
  const st = (n) => (failed === 0 || n < failed ? "pass" : n === failed ? "fail" : "skip");
  const modulesRan = failed === 0;
  const items = [
    ["info", "Challenge issued: one-time nonce, transaction hash and two actions", ""],
    ["info", `Evidence summary signed on the device (${pq ? "ML-DSA-65" : "Ed25519"})`, ""],
    ["info", `Encrypted with ${pq ? "ML-KEM-768" : "X25519"} + AES-256-GCM and sent${req ? ` (${(req.bytes_in / 1024).toFixed(0)} KB)` : ""}`, ""],
    [st(1), "Check 1: challenge exists, not expired, not used before (now marked used)", ms(t.check1_challenge)],
    [st(2), "Check 2: crypto mode matches the challenge (no downgrade)", ""],
    [st(3), "Check 3: decrypted, encryption tag valid (not altered)", ms(t.check3_unseal)],
    [st(4), "Check 4: nonce and transaction hash match what was issued", ms(t.check4_binding)],
    [st(5), "Check 5: video hash matches what the device signed", ms(t.check5_hash)],
    [st(6), "Check 6: device signature valid for the enrolled key", ms(t.check6_signature)],
    [modulesRan ? "pass" : "skip", "Face found and matched; M1–M5 run in parallel", modulesRan ? ms((t.decode || 0) + (t.faces || 0) + (t.embed || 0) + (t.detectors_parallel || 0)) : ""],
    [modulesRan ? (rec.decision === "accepted" ? "pass" : "fail") : "skip", `Decision: ${rec.decision}`, ms(t.decision)],
    ["pass", `Verdict signed by the server (${pq ? "ML-DSA-65" : "Ed25519"}) and stored`, ms(t.verdict_sign)],
  ];
  const total = t.total ? ` <span class="t">(server total ${(t.total / 1000).toFixed(0)} ms)</span>` : "";
  return `<h4>What happened${total}</h4><ul class="trace">${items.map(([c, txt, tm]) => `<li class="${c}">${txt} ${tm}</li>`).join("")}</ul>`;
}

function showVerdict(b) {
  const v = b.verdict; const rec = v.record;
  return `${badge(rec.decision)} <span class="muted small">${esc(v.mode)} · signature ${v.signature_valid ? "valid" : "INVALID"}</span>
    ${reasonLine(rec.reason_code)}${scoreTable(rec.scores, v.thresholds)}${trace(rec, b.request, v.mode)}`;
}

function showReplay(b) {
  const rows = b.attacks.map((a) => `<tr><td>${esc(a.attack)}</td><td>${badge(a.client_sees)}</td>
    <td><code>${esc(a.server_reason)}</code><br><span class="muted small">${esc(REASONS[a.server_reason] || "")}</span></td></tr>`).join("");
  return `<table><tr><th>attack</th><th>client sees</th><th>server's reason (never sent to the client)</th></tr>${rows}</table>`;
}

function showMorph(b) {
  const rows = b.pairs.map((p) => `<tr><td>${esc(p.pair)}</td><td class="num">${num(p.score, 4)}</td>
    <td>${p.score === null ? "no face" : badge(p.rejected ? "rejected" : "accepted")}${p.rejected ? " <code>MORPH_FLAG</code>" : ""}</td></tr>`).join("");
  return `<p class="muted small">${esc(b.model)} · reject if score &gt; ${num(b.threshold, 4)} (BPCER 0.5%)</p>
    <table><tr><th>pair</th><th>score</th><th>decision</th></tr>${rows}</table>`;
}

function showDeepfake(b) {
  const rows = b.faces.map((f) => `<tr><td>${f.truth === "fake" ? "<b>fake</b>" : "real"}</td><td class="num">${num(f.p_fake, 4)}</td>
    <td>${badge(f.rejected ? "rejected" : "accepted")}${f.rejected ? " <code>DEEPFAKE_FAIL</code>" : ""}</td><td class="muted small">${esc(f.file)}</td></tr>`).join("");
  return `<p class="muted small">${esc(b.model)} on ${esc(b.dataset)} · reject if P(fake) &gt; ${num(b.threshold, 4)}</p>
    <table><tr><th>truth</th><th>P(fake)</th><th>decision</th><th>file</th></tr>${rows}</table>`;
}

function showTamper(b) {
  const o = b.original; const rec = o.record;
  return `<table>
    <tr><th>record</th><th>offline signature check</th></tr>
    <tr><td>Your last accepted verdict (${esc(o.mode)}, ${esc(o.challenge_id.slice(0, 8))}…) as stored</td>
      <td><span class="badge ${o.signature_valid ? "ok" : "bad"}">${o.signature_valid ? "VALID" : "INVALID"}</span></td></tr>
    <tr><td>Copy of the database with the M3 deepfake score edited ${num(b.old_value)} → ${num(b.new_value)}</td>
      <td><span class="badge ${b.edited_valid ? "ok" : "bad"}">${b.edited_valid ? "VALID" : "INVALID"}</span>${b.edited_valid ? "" : " edit detected"}</td></tr>
  </table><p class="muted small">The verdict binds the challenge, transaction, evidence hash, every score, model versions and the decision.
  Original database untouched; the edited copy is ${esc(b.copy)}.</p>${scoreTable(rec.scores, o.thresholds)}`;
}

function showCrypto(b) {
  if (b.pending) return `<span class="badge skip">pending</span><p>The benchmark file eval/results/crypto_bench.csv does not exist yet (it runs on an idle machine).</p>`;
  const rows = b.ops.map((o) => `<tr><td>${esc(o.op)}</td><td class="num">${num(o.classical_us, 1)} µs</td><td class="num">${num(o.pq_us, 1)} µs</td></tr>`).join("");
  const s = b.sizes;
  return `<table><tr><th>operation (median of ${esc(b.runs)} runs)</th><th>X25519 / Ed25519</th><th>ML-KEM-768 / ML-DSA-65</th></tr>${rows}</table>
    <p>Bytes per request: KEM ciphertext ${esc(s.kem_ciphertext[0])} → ${esc(s.kem_ciphertext[1])}, signature ${esc(s.signature[0])} → ${esc(s.signature[1])};
    post-quantum adds <b>${Number(s.extra_per_request).toLocaleString()} bytes</b>.</p>
    <p class="muted small">${esc(b.system)}, cryptography ${esc(b.cryptography)}, payload ${esc(b.payload_kb)} KB.</p>`;
}

async function resetDemo() {
  if (busy) return;
  busy = true; updateDisabled();
  try {
    const b = await api("/demo/api/reset", { method: "POST" });
    document.querySelectorAll(".card .result").forEach((r) => (r.innerHTML = ""));
    alert(b && b.message ? b.message : "Demo reset.");
  } catch (e) { console.error(e); alert("Reset failed; restart with stop_demo.bat and start_demo.bat."); }
  busy = false; updateDisabled(); loadStatus();
}

// ---- results tab -------------------------------------------------------------------------
const RESULT_VIEWS = [
  ["matcher", "Face matcher on FRLL (threshold at FMR 0.1%)", (r) => r.view === "all", ["split", "threshold", "fmr", "fnmr", "n_genuine", "n_impostor"]],
  ["m2", "M2 morph detection, pooled held-out (per morph method)", (r) => r.rows === "pooled_held_out", null],
  ["m3_zero_shot", "M3 deepfake, zero-shot GenD (thresholds from Val)", () => true, ["split", "target_bpcer", "threshold", "bpcer", "apcer_max", "eer", "auc", "n_bona_fide"]],
  ["m3_compare", "M3 on Test-Dev: zero-shot vs LayerNorm-tuned", () => true, ["model", "thresholds_from", "target_bpcer", "bpcer", "apcer_max", "eer", "auc"]],
  ["replay", "Replay suite (protocol attacks)", () => true, ["mode", "case", "runs", "expected_code", "correct_code_rate"]],
  ["genuine", "Genuine webcam captures (all of them, including rejections)", () => true, ["request", "phase", "mode", "decision", "reason_code", "match", "m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection"]],
];

async function loadResults() {
  let data;
  try { data = await api("/demo/api/results"); } catch (e) { console.error(e); $("#results-body").innerHTML = `<div class="error">Could not load results.</div>`; return; }
  $("#results-body").innerHTML = RESULT_VIEWS.map(([key, title, keep, cols]) => {
    const t = data[key];
    if (!t) return `<h4>${title}</h4><p><span class="badge skip">pending</span></p>`;
    const rows = t.rows.filter(keep);
    const columns = cols || t.columns.filter((c) => c === "target_bpcer" || c === "bpcer" || c.startsWith("apcer[") || c === "eer" || c === "auc" || c === "n_bona_fide");
    const fmt = (c, v) => (v !== "" && !isNaN(Number(v)) && String(v).includes(".") ? num(v, 4) : esc(v));
    return `<h4>${title} <span class="muted small">${esc(t.file)}</span></h4><table><tr>${columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>
      ${rows.map((r) => `<tr>${columns.map((c) => `<td class="num">${fmt(c, r[c])}</td>`).join("")}</tr>`).join("")}</table>`;
  }).join("");
}

// ---- tabs and start --------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $("#tab-demo").hidden = b.dataset.tab !== "demo";
  $("#tab-results").hidden = b.dataset.tab !== "results";
  if (b.dataset.tab === "results") loadResults();
}));
loadStatus();
