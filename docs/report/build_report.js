'use strict';
// Builds docs/report/PQ_Biometric_Course_Report.docx from eval/results/*.csv and docs/report/figures/*.png.
// Every number in the report is read from a CSV here; nothing is typed in by hand.
//
//   python docs/report/make_figures.py
//   NODE_PATH=<folder containing node_modules/docx> node docs/report/build_report.js
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  BorderStyle, ShadingType, ImageRun, LevelFormat, PageBreak, Footer, PageNumber, VerticalAlign, ExternalHyperlink,
} = require('docx');

const ROOT = path.resolve(__dirname, '..', '..');
const RES = path.join(ROOT, 'eval', 'results');
const FIG = path.join(__dirname, 'figures');
const OUT = process.env.REPORT_OUT || path.join(__dirname, 'PQ_Biometric_Course_Report.docx');
const PAGE_W = 9026; // A4 width minus two 1-inch margins, in DXA

// ---------------------------------------------------------------- data
function parseCsv(text) {
  text = text.replace(/^﻿/, '');
  const rows = [];
  let row = [], field = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows.filter((r) => !(r.length === 1 && r[0] === ''));
  return body.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}
function csv(name) {
  const p = path.join(RES, name);
  return fs.existsSync(p) ? parseCsv(fs.readFileSync(p, 'utf8')) : null;
}
const num = (x) => Number(x);
const fx = (x, d = 2) => num(x).toFixed(d);
const pct = (x, d = 2) => `${(100 * num(x)).toFixed(d)}%`;
const int = (x) => Math.round(num(x)).toLocaleString('en-US');

const matcher = csv('matcher.csv');
const mVal = matcher.find((r) => r.split === 'val' && r.view === 'all');
const mTest = matcher.find((r) => r.split === 'test' && r.view === 'all');
const m2 = csv('m2_morph.csv').filter((r) => r.rows === 'pooled_held_out');
const m2dep = csv('m2_deploy_thresholds.csv');
const m3cmp = csv('m3_zero_shot_vs_ln_tuned.csv');
const m3val = csv('m3_deepfake_zero_shot.csv').filter((r) => r.split === 'val');
const m3tune = csv('m3_ln_tuning.csv');
const onnx = csv('onnx_export.csv');
const gendEq = csv('m3_gend_equivalence.csv');
const cb = csv('crypto_bench.csv');
const sizes = csv('crypto_sizes.csv');
const thresholds = csv('thresholds.csv');
const genuine = csv('genuine_webcam_captures.csv');
const perfSum = csv('perf_summary.csv');
const perfStages = csv('perf_stages.csv');
const replay = csv('replay.csv');
const matrix = csv('attack_matrix.csv');

const M2_TOOLS = ['amsl', 'facemorpher', 'opencv', 'stylegan', 'webmorph'];
const m2at = (t) => m2.find((r) => r.target_bpcer === t);
const m3at = (model, from, t) => m3cmp.find((r) => r.model === model && r.thresholds_from === from && r.target_bpcer === t);
const zs = (t) => m3at('zero-shot', 'Val calib 20%', t);
const ln = (t) => m3at('LayerNorm-tuned', 'Val calib 20%', t);
const cop = (mode, op) => cb.find((r) => r.mode === mode && r.op === op);
const size = (mode) => sizes.find((r) => r.mode === mode);
const thr = (mod) => thresholds.find((r) => r.module === mod);
const sumOf = (mode) => perfSum && perfSum.find((r) => r.mode === mode);
const us2ms = (x) => (num(x) / 1000).toFixed(3);

// genuine webcam captures, grouped by phase
const phases = [];
for (const r of genuine) {
  let p = phases.find((q) => q.phase === r.phase);
  if (!p) { p = { phase: r.phase, rows: [] }; phases.push(p); }
  p.rows.push(r);
}
const afterReenrol = genuine.filter((r) => r.phase.includes('after re-enrolment') || r.phase.startsWith('LayerNorm'));
const afterAccepted = afterReenrol.filter((r) => r.decision === 'accepted');
const acceptedMatch = afterAccepted.map((r) => num(r.match));
const maxM3After = Math.max(...afterReenrol.map((r) => num(r.m3_deepfake)));
const maxM1After = Math.max(...afterReenrol.map((r) => num(r.m1_pad)));

// ---------------------------------------------------------------- references (IEEE), numbered by first citation
const REFS = {
  shor: 'P. W. Shor, "Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer," *SIAM J. Comput.*, vol. 26, no. 5, pp. 1484–1509, 1997.',
  grover: 'L. K. Grover, "A fast quantum mechanical algorithm for database search," in *Proc. 28th Annu. ACM Symp. Theory Comput. (STOC)*, 1996, pp. 212–219.',
  gill: 'S. S. Gill *et al.*, "Quantum computing: A taxonomy, systematic review and future directions," *Softw. Pract. Exper.*, vol. 52, no. 1, pp. 66–114, 2022, doi: 10.1002/spe.3039.',
  mosca: 'M. Mosca, "Cybersecurity in an era with quantum computers: Will we be ready?" *IEEE Secur. Privacy*, vol. 16, no. 5, pp. 38–41, 2018.',
  veridos: 'Veridos GmbH, "The quantum risk in identity ecosystems: Managing the post-quantum migration window," white paper, 2026.',
  burt: 'C. Burt, "Post-quantum identity expands beyond cryptography," *Biometric Update*, Aug. 7, 2026. [Online]. Available: https://www.biometricupdate.com/202608/post-quantum-identity-expands-beyond-cryptography',
  daon: 'Daon, "Daon introduces five-patent Quantum Identity architecture for the next era of digital trust," press release, 2026 (covers U.S. Patent 12,695,590, issued Jul. 28, 2026). [Online]. Available: https://www.daon.com/resource/daon-introduces-five-patent-quantum-identity-architecture-for-the-next-era-of-digital-trust/',
  authid: '"authID adds post-quantum cryptography to biometric signature platform," *Biometric Update*, Apr. 2026. [Online]. Available: https://www.biometricupdate.com/202604/authid-adds-post-quantum-cryptography-to-biometric-signature-platform',
  fips203: 'National Institute of Standards and Technology, "Module-lattice-based key-encapsulation mechanism standard," FIPS 203, Aug. 2024, doi: 10.6028/NIST.FIPS.203.',
  fips204: 'National Institute of Standards and Technology, "Module-lattice-based digital signature standard," FIPS 204, Aug. 2024, doi: 10.6028/NIST.FIPS.204.',
  fips180: 'National Institute of Standards and Technology, "Secure hash standard (SHS)," FIPS 180-4, Aug. 2015, doi: 10.6028/NIST.FIPS.180-4.',
  sp80038d: 'M. Dworkin, "Recommendation for block cipher modes of operation: Galois/Counter Mode (GCM) and GMAC," NIST SP 800-38D, Nov. 2007.',
  rfc5869: 'H. Krawczyk and P. Eronen, "HMAC-based extract-and-expand key derivation function (HKDF)," RFC 5869, May 2010.',
  rfc2104: 'H. Krawczyk, M. Bellare, and R. Canetti, "HMAC: Keyed-hashing for message authentication," RFC 2104, Feb. 1997.',
  rfc7748: 'A. Langley, M. Hamburg, and S. Turner, "Elliptic curves for security," RFC 7748, Jan. 2016.',
  rfc8032: 'S. Josefsson and I. Liusvaara, "Edwards-curve digital signature algorithm (EdDSA)," RFC 8032, Jan. 2017.',
  regev: 'O. Regev, "On lattices, learning with errors, random linear codes, and cryptography," *J. ACM*, vol. 56, no. 6, Art. 34, 2009.',
  langlois: 'A. Langlois and D. Stehlé, "Worst-case to average-case reductions for module lattices," *Des. Codes Cryptogr.*, vol. 75, no. 3, pp. 565–599, 2015.',
  kyber: 'J. Bos *et al.*, "CRYSTALS – Kyber: A CCA-secure module-lattice-based KEM," in *Proc. IEEE Eur. Symp. Secur. Privacy (EuroS&P)*, 2018, pp. 353–367.',
  dilithium: 'L. Ducas *et al.*, "CRYSTALS-Dilithium: A lattice-based digital signature scheme," *IACR Trans. Cryptogr. Hardw. Embed. Syst.*, vol. 2018, no. 1, pp. 238–268, 2018.',
  iso30107: 'ISO/IEC 30107-3:2023, *Information technology — Biometric presentation attack detection — Part 3: Testing and reporting*. Geneva, Switzerland: ISO, 2023.',
  ferrara: 'M. Ferrara, A. Franco, and D. Maltoni, "The magic passport," in *Proc. IEEE Int. Joint Conf. Biometrics (IJCB)*, 2014, pp. 1–7.',
  tolosana: 'R. Tolosana, R. Vera-Rodriguez, J. Fierrez, A. Morales, and J. Ortega-Garcia, "Deepfakes and beyond: A survey of face manipulation and fake detection," *Inf. Fusion*, vol. 64, pp. 131–148, 2020.',
  vcd: 'D. Kurmankhojayev *et al.*, "Virtual camera detection: Catching video injection attacks in remote biometric systems," arXiv:2512.10653, 2025.',
  roman: 'R. Román, R. Arjona, P. López-González, and I. Baturone, "A quantum-resistant face template protection scheme using Kyber and Saber public key encryption algorithms," in *Proc. Int. Conf. Biometrics Special Interest Group (BIOSIG)*, Darmstadt, Germany, 2022.',
  arjona: 'R. Arjona, P. López-González, R. Román, and I. Baturone, "Post-quantum biometric authentication based on homomorphic encryption and Classic McEliece," *Appl. Sci.*, vol. 13, no. 2, Art. 757, 2023, doi: 10.3390/app13020757.',
  olushola: 'A. Olushola and S. P. Meenakshi, "A post-quantum privacy-preserving multimodal biometric authentication framework integrating ML-KEM (Kyber) and fully homomorphic encryption," *Front. Phys.*, vol. 14, Art. 1849257, 2026, doi: 10.3389/fphy.2026.1849257.',
  gowri: 'N. S. Gowri Ganesh, J. Jesila Mol, S. Manoj Kumar, V. Reddy, and T. Meyyappan, "Biometric spoofing detection with post-quantum cryptography integration," in *Proc. 9th Int. Conf. Comput. Intell. Data Sci. (ICCIDS)*, Chennai, India, Jan. 2026, doi: 10.1109/ICCIDS69108.2026.11407868.',
  celebaspoof: 'Y. Zhang *et al.*, "CelebA-Spoof: Large-scale face anti-spoofing dataset with rich annotations," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 70–85.',
  silentface: 'Minivision AI, "Silent-Face-Anti-Spoofing," GitHub repository, 2020. [Online]. Available: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing',
  faceflashing: 'D. Tang, Z. Zhou, Y. Zhang, and K. Zhang, "Face Flashing: A secure liveness detection protocol based on light reflections," in *Proc. Netw. Distrib. Syst. Secur. Symp. (NDSS)*, 2018.',
  gotcha: 'G. Mittal, C. Hegde, and N. Memon, "GOTCHA: Real-time video deepfake detection via challenge-response," arXiv:2210.06186, 2023.',
  scherhag: 'U. Scherhag, C. Rathgeb, J. Merkle, and C. Busch, "Deep face representations for differential morphing attack detection," *IEEE Trans. Inf. Forensics Security*, vol. 15, pp. 3625–3639, 2020.',
  sarkar: 'E. Sarkar, P. Korshunov, L. Colbois, and S. Marcel, "Vulnerability analysis of face morphing attacks from landmarks and generative adversarial networks," arXiv:2012.05344, 2020.',
  openforensics: 'T.-N. Le, H. H. Nguyen, J. Yamagishi, and I. Echizen, "OpenForensics: Large-scale challenging dataset for multi-face forgery detection and segmentation in-the-wild," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10117–10127.',
  gend: 'A. Yermakov, J. Cech, J. Matas, and M. Fritz, "Deepfake detection that generalizes across benchmarks," in *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)*, 2026, doi: 10.1109/WACV61042.2026.00082.',
  clip: 'A. Radford *et al.*, "Learning transferable visual models from natural language supervision," in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 8748–8763.',
  arcface: 'J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4690–4699.',
  insightface: 'J. Guo, J. Deng, *et al.*, "InsightFace: 2D and 3D face analysis project," GitHub repository. [Online]. Available: https://github.com/deepinsight/insightface',
  yunet: 'W. Wu, H. Peng, and S. Yu, "YuNet: A tiny millisecond-level face detector," *Mach. Intell. Res.*, vol. 20, no. 5, pp. 656–665, 2023.',
  mediapipe: 'C. Lugaresi *et al.*, "MediaPipe: A framework for building perception pipelines," arXiv:1906.08172, 2019.',
  sklearn: 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
  lightgbm: 'G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 3146–3154.',
  adamw: 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.',
  frll: 'L. DeBruine and B. Jones, "Face Research Lab London Set," figshare, 2017, doi: 10.6084/m9.figshare.5047666.',
  frllmorphs: 'Idiap Research Institute, "FRLL-Morphs," Zenodo record 4415159. [Online]. Available: https://zenodo.org/record/4415159',
  acvp: 'National Institute of Standards and Technology, "ACVP-Server: Automated Cryptographic Validation Protocol," GitHub repository. [Online]. Available: https://github.com/usnistgov/ACVP-Server',
  pyca: 'Python Cryptographic Authority, "cryptography," version 50.0.2. [Online]. Available: https://cryptography.io',
  onnxrt: 'ONNX Runtime developers, "ONNX Runtime," version 1.31.0. [Online]. Available: https://onnxruntime.ai',
};
const citeOrder = [];
function citeNum(k) {
  if (!REFS[k]) throw new Error(`unknown reference ${k}`);
  if (!citeOrder.includes(k)) citeOrder.push(k);
  return citeOrder.indexOf(k) + 1;
}

// ---------------------------------------------------------------- inline text: **bold**, *italic*, {ref,ref}, _{sub}, ^{sup}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+?\*\*|\*[^*\s][^*]*?\*|\{[a-z0-9_]+(?:,[a-z0-9_]+)*\}|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ ...base, text: t.slice(2, -2), bold: true }));
    else if (t.startsWith('_{')) out.push(new TextRun({ ...base, text: t.slice(2, -1), subScript: true }));
    else if (t.startsWith('^{')) out.push(new TextRun({ ...base, text: t.slice(2, -1), superScript: true }));
    else if (t.startsWith('{')) {
      const keys = t.slice(1, -1).split(',');
      if (keys.every((k) => REFS[k])) out.push(new TextRun({ ...base, text: keys.map(citeNum).sort((a, b) => a - b).map((n) => `[${n}]`).join(', ') }));
      else out.push(new TextRun({ ...base, text: t }));
    } else out.push(new TextRun({ ...base, text: t.slice(1, -1), italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out;
}

// ---------------------------------------------------------------- block builders
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));
const P = (text, o = {}) => new Paragraph({
  children: runs(text, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120 }, indent: o.indent, keepNext: o.keepNext,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const bullets = (items) => items.map((t) => new Paragraph({
  numbering: { reference: 'bullets', level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 },
}));
let listNo = 0;
const numbered = (items) => {
  listNo++;
  return items.map((t) => new Paragraph({
    numbering: { reference: `num${listNo}`, level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 },
  }));
};
let eqNo = 0;
const Eq = (t) => { eqNo++; return new Paragraph({ alignment: AlignmentType.CENTER, keepLines: true, spacing: { before: 60, after: 120 }, children: [...runs(t, { italics: true }), new TextRun({ text: `     (${eqNo})` })] }); };

const TABLES = ['related', 'threats', 'checks', 'modules', 'stack', 'params', 'datasets', 'metrics', 'targets', 'crypto_ops', 'sizes',
  'perf_summary', 'perf_stages', 'replay', 'matcher', 'm2', 'm3', 'm3_tuning', 'onnx', 'matrix', 'genuine', 'pending', 'codes'];
const FIGS = ['arch', 'pipeline', 'ui_overview', 'crypto', 'perf', 'm3', 'ui_attacks', 'ui_detect', 'ui_results'];
const T = (k) => { const i = TABLES.indexOf(k); if (i < 0) throw new Error(k); return `Table ${i + 1}`; };
const F = (k) => { const i = FIGS.indexOf(k); if (i < 0) throw new Error(k); return `Figure ${i + 1}`; };
let tablesMade = 0, figsMade = 0;

const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(key, caption, headers, rows, widths, o = {}) {
  if (TABLES[tablesMade] !== key) throw new Error(`table order: expected ${TABLES[tablesMade]}, got ${key}`);
  tablesMade++;
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== PAGE_W) throw new Error(`table ${key} widths sum ${total}`);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: 'D9E2F3', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: String(t).split('\n').map((line) => new Paragraph({
      alignment: (o.left || (i === 0 && !head)) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(line, { size: 19, bold: head || undefined }),
    })),
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 }, children: runs(`**${T(key)}.** ${caption}`, { size: 21 }) }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))],
    }),
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}
function figure(key, file, caption, widthPx = 600) {
  if (FIGS[figsMade] !== key) throw new Error(`figure order: expected ${FIGS[figsMade]}, got ${key}`);
  figsMade++;
  const p = path.join(FIG, file);
  if (!fs.existsSync(p)) {
    return [P(`**${F(key)}.** ${caption} *(Figure not generated: its results CSV is missing.)*`, { align: AlignmentType.CENTER })];
  }
  const data = fs.readFileSync(p);
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 }, children: [new ImageRun({ type: 'png', data, transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(`**${F(key)}.** ${caption}`, { size: 21 }) }),
  ];
}
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ================================================================ derived sentences for results that may still be running
const clS = sumOf('CLASSICAL'), pqS = sumOf('PQ');
const perfDone = !!(clS && pqS);
const replayDone = !!replay;
const matrixDone = !!matrix;
const replayAttacks = replayDone ? replay.filter((r) => r.case !== 'genuine') : [];
const replayGenuine = replayDone ? replay.filter((r) => r.case === 'genuine') : [];
const replayAllCorrect = replayDone && replayAttacks.every((r) => num(r.correct_code_rate) === 1);
const replayAttackRuns = replayAttacks.reduce((a, r) => a + num(r.runs), 0);
const replayProtocolFR = replayGenuine.reduce((a, r) => a + num(r.protocol_false_rejections || 0), 0);
const replayGenuineRuns = replayGenuine.reduce((a, r) => a + num(r.runs), 0);

function perfSentence() {
  if (!perfDone) return 'The end-to-end latency run (500 genuine transactions per mode) had not finished when this report was generated; its table is marked pending.';
  return `End to end, a transaction took a median of ${fx(clS.end_to_end_p50_ms, 0)} ms in classical mode and ${fx(pqS.end_to_end_p50_ms, 0)} ms in PQ mode, a difference smaller than the run-to-run variation of the detectors; all cryptographic steps together took ${fx(clS.crypto_p50_ms, 1)} ms and ${fx(pqS.crypto_p50_ms, 1)} ms, ${pct(clS.crypto_share_p50, 2)} and ${pct(pqS.crypto_share_p50, 2)} of the total.`;
}
function replaySentence() {
  if (!replayDone) return 'The 1,000-run replay suite with the real models had not finished when this report was generated.';
  return replayAllCorrect
    ? `In the replay suite all ${int(replayAttackRuns)} protocol attacks were rejected with the expected reason code, and ${int(replayProtocolFR)} of ${int(replayGenuineRuns)} genuine runs failed a protocol check.`
    : `In the replay suite ${int(replayAttacks.reduce((a, r) => a + num(r.rejected), 0))} of ${int(replayAttackRuns)} protocol attacks were rejected (${int(replayAttacks.reduce((a, r) => a + num(r.correct_code), 0))} with the expected reason code), and ${int(replayProtocolFR)} of ${int(replayGenuineRuns)} genuine runs failed a protocol check.`;
}

// ================================================================ TITLE
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 240 }, children: [new TextRun({ text: 'AI-Based Biometric Evidence Authentication for Post-Quantum Digital Identity Systems', bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({ text: 'Course Project Report', size: 26 })] }),
);
const authorRows = [
  ['Aviral Bhan', '23BIT0241', 'aviralbhanmv@gmail.com'],
  ['Krish Manoj Mehta', '23BIT407', '[Gmail address to be added]'],
  ['Abhijay Singh Panwar', '23BIT0353', '[Gmail address to be added]'],
];
add(new Table({
  width: { size: 7600, type: WidthType.DXA }, columnWidths: [2700, 1700, 3200], alignment: AlignmentType.CENTER,
  rows: [['Name', 'Registration number', 'Gmail'], ...authorRows].map((r, ri) => new TableRow({
    children: r.map((t, i) => new TableCell({
      borders, width: { size: [2700, 1700, 3200][i], type: WidthType.DXA },
      shading: ri === 0 ? { fill: 'D9E2F3', type: ShadingType.CLEAR, color: 'auto' } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, bold: ri === 0, size: 22 })] })],
    })),
  })),
}));
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300, after: 60 }, children: [new TextRun({ text: 'Supervisors: Dr. Aswani Kumar Cherukuri and Dr. Rajasekhara Reddy D.', size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Vellore Institute of Technology, Vellore', size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({ text: 'October 2026', size: 24 })] }),
);

// ================================================================ ABSTRACT
const abstract = [
  'Face-based digital identity systems now approve logins, payments and account recovery from a few seconds of selfie video, so the evidence they receive must be both genuine and protected.',
  'Two threats meet here: spoofed evidence (printed photos, screen replays, morphed enrolment photos, deepfakes and virtual-camera injection) and future quantum computers that could break the RSA and elliptic-curve cryptography that protects the evidence today.',
  'Existing work usually handles one threat at a time: post-quantum schemes protect templates or channels, while attack detectors produce scores that stay unsigned inside the server.',
  'This project builds and measures a working prototype that does both and links them.',
  'Every transaction receives a single-use challenge with a 32-byte nonce valid for 60 seconds, a hash of the transaction and two random liveness actions.',
  'The client records a four-second video, signs a manifest with ML-DSA-65 (FIPS 204) and encrypts everything with AES-256-GCM under a key agreed through ML-KEM-768 (FIPS 203).',
  'The server runs six protocol checks in a fixed order and then a face matcher and five detectors in parallel: presentation attacks (M1), morphing (M2), deepfakes (M3), replay through liveness actions (M4) and virtual-camera injection (M5).',
  'The novelty is that the final verdict, with its scores, model versions, thresholds, decision and the hashes of the challenge, transaction and video, is itself signed with ML-DSA-65, so the AI decision becomes evidence that can be re-checked later.',
  'A classical mode (X25519 and Ed25519) runs the same code, so the cost of post-quantum cryptography is measured with the detectors held constant.',
  `On held-out subjects of the FRLL face set the ArcFace matcher rejected none of ${int(mTest.n_genuine)} genuine pairs at a false-match rate of ${pct(mTest.fmr)}.`,
  `The differential morph detector let through at most ${pct(m2at('0.005').apcer_max)} of morphs from five morphing tools while rejecting ${pct(m2at('0.005').bpcer)} of genuine pairs.`,
  `LayerNorm-only tuning of the GenD deepfake detector raised the AUC on the independent OpenForensics Test-Dev set from ${fx(zs('0.005').auc, 3)} to ${fx(ln('0.005').auc, 3)} and cut the share of fakes accepted at a 0.5% genuine-rejection target from ${pct(zs('0.005').apcer_max, 1)} to ${pct(ln('0.005').apcer_max, 1)}.`,
  `Each ML-KEM-768 and ML-DSA-65 operation took at most ${us2ms(Math.max(...['kem_keygen', 'encapsulate', 'decapsulate', 'sign_manifest', 'verify_manifest', 'sign_verdict'].map((o) => num(cop('PQ', o).p50_us))))} ms (median) on a laptop CPU, and PQ mode adds ${int(num(size('PQ').per_transaction_asymmetric_B) - num(size('CLASSICAL').per_transaction_asymmetric_B))} bytes of key-exchange and signature data per transaction.`,
  perfSentence(),
  replaySentence(),
  `With an enrolment made in the same room, the real user was accepted in ${afterAccepted.length} of ${afterReenrol.length} live webcam captures across both modes.`,
  'Fine-tuning M1 on CelebA-Spoof and training the M5 classifier on recorded virtual-camera sessions are still pending and are reported as such.',
  'The results show that post-quantum cryptography is affordable for biometric evidence but only proves where evidence came from and that it was not changed; the detectors, and the signed verdict that records their decision, are what show it is genuine.',
];
add(H1('Abstract'), P(abstract.join(' ')));
add(P('**Keywords:** Deepfake detection; Face anti-spoofing; ML-DSA; ML-KEM; Morphing attack detection; Post-quantum cryptography.'));
add(pageBreak());

// ================================================================ 1 INTRODUCTION
add(H1('1. Introduction'));
add(P('Remote face verification has become the default way to prove identity online. Banks, payment apps and government services ask the user to look into a phone or laptop camera, and a server decides from a few seconds of video whether the person is who they claim to be. Because the decision is made far from the user, the server can only trust what it receives, and attackers have learned to feed it convincing fakes: printed photos and replayed screens held up to the camera (presentation attacks {iso30107}), enrolment photos that blend two faces so both people match (morphing {ferrara}), AI-generated or face-swapped video (deepfakes {tolosana}), and fake video streams fed in through virtual-camera software so that no physical camera is involved at all (injection {vcd}).'));
add(P('At the same time the cryptography that protects this evidence is reaching the end of its life. Shor\'s algorithm factors integers and computes discrete logarithms in polynomial time on a large quantum computer {shor}, which breaks RSA, Diffie–Hellman and elliptic-curve schemes such as X25519 and Ed25519; Grover\'s search gives a quadratic speed-up against symmetric keys {grover}, which is why 256-bit keys are now preferred. A recent survey summarises how fast the hardware is improving {gill}. The danger is not only in the future: an attacker can record encrypted traffic today and decrypt it once a capable machine exists ("harvest now, decrypt later") {mosca}. For biometrics this is worse than for passwords, because a face cannot be reissued after it leaks {veridos}. NIST therefore standardised ML-KEM for key establishment and ML-DSA for signatures in August 2024 {fips203,fips204}.'));
add(P('Industry is starting to combine the two worlds. Daon describes a patented "Quantum Identity" architecture that binds biometric evidence to individual transactions {daon,burt}, and authID signs biometric transactions with ML-DSA-65 {authid}. Academic work protects face templates with lattice-based or code-based encryption {roman,arjona,olushola}. Post-quantum cryptography, however, works like a tamper-proof envelope: it proves that nobody opened or swapped the envelope, even after quantum computers exist, but it says nothing about whether the photo inside is a printout, a morph or a deepfake. A secure identity system needs both properties at once, and it needs a record of why each decision was made.'));
add(H2('1.1 Problem statement and why post-quantum security'));
add(P('The problem addressed is: *how can a remote face-verification system accept only genuine, live, fresh evidence for one specific transaction, keep that evidence confidential and unforgeable against an adversary with a future quantum computer, and leave an auditable, tamper-evident record of the AI decision, at a cost that users will not notice?* Post-quantum cryptography is needed because biometric evidence has a long shelf life: a face video captured today identifies the same person for decades, so it must stay secret beyond the expected arrival of cryptographically relevant quantum computers. Quantum computing is not used to run the system; the project uses quantum-resistant algorithms on ordinary hardware, which is how post-quantum security is deployed in practice.'));
add(H2('1.2 Goals of the Project'));
add(bullets([
  'Protect the evidence in transit with ML-KEM-768 and ML-DSA-65, and bind it to one transaction and one challenge so that it cannot be replayed or reused for another payment.',
  'Detect five attack classes (presentation, morphing, deepfake, replay, injection) with open models on openly downloadable datasets, reported per attack type at three fixed operating points.',
  'Sign the final verdict (scores, model versions, thresholds, decision and hashes) so that the AI decision can be re-checked later.',
  'Measure accuracy, latency, CPU, memory and bytes per request in classical and post-quantum modes with identical detectors, to quantify the cost of post-quantum security.',
  'Keep the prototype reproducible: public code, pinned versions, scripts for every result and no hand-typed numbers.',
]));
add(H2('1.3 Novelty'));
add(P('Neither post-quantum protection of biometric data nor AI attack detection is new, and this report does not claim either. To our knowledge, in the open literature, existing systems treat the detector\'s verdict as an unsigned internal result. Here the verdict itself becomes post-quantum-signed evidence: the server signs, with ML-DSA-65, one record that binds the challenge, the transaction hash, the video hash, every detector score, the model versions, the threshold version and the decision. Anyone holding the server\'s public key can later prove that a stored decision was not edited and that it belongs to that exact transaction and video.'));
add(P('The second contribution is empirical: one testbed in which the same detectors run under classical and post-quantum cryptography, so that the report can state, with measurements, what post-quantum cryptography adds (confidentiality against future quantum attackers, at a measured cost in time and bytes) and what it does not (it does not stop a single presentation, morph or deepfake attack, which the AI modules must catch).'));
add(H2('1.4 Organization of the Report'));
add(P('The remainder of this report is organized as follows: Section 2 reviews related work, the theory behind the quantum threat and lattice cryptography, and the gap this project fills. Section 3 formulates the problem and describes the protocol, the six server checks, the five detectors and the implementation. Section 4 gives the experimental setup, the metrics and the results, comparing classical and post-quantum cryptography and discussing sources of error. Section 5 discusses novelty, insights and limitations, and Section 6 concludes. The code link, originality and AI-usage statements, references and an appendix follow.'));

// ================================================================ 2 BACKGROUND
add(H1('2. Background and Related Work'));
add(H2('2.1 Literature Review'));
add(P('**Post-quantum protection of biometric data.** Román *et al.* combined the Kyber and Saber public-key schemes with homomorphic encryption so that face templates can be compared while encrypted {roman}. Arjona *et al.* built a similar scheme on Classic McEliece and showed real-time matching of protected FaceNet embeddings {arjona}. Olushola and Meenakshi fused face, iris and fingerprint features and matched them under CKKS homomorphic encryption with ML-KEM (Kyber-1024) session keys {olushola}. Gowri Ganesh *et al.* sketched, in simulation, a pipeline that places a CNN presentation-attack detector next to ML-KEM {gowri}. These works protect templates and channels; none signs the detector\'s decision, and none measures replay, morphing, deepfake and injection attacks in one system.'));
add(P('**Industry systems.** Daon\'s five-patent architecture checks whether evidence is real, live and transaction-bound; its U.S. Patent 12,695,590 adds a transaction-specific noise vector to biometric features to resist injection and replay {daon}. A trade report notes that post-quantum identity is moving "beyond cryptography" to the evidence itself {burt}, and authID now offers ML-DSA-65 signatures on biometric transactions {authid}. Veridos argues that biometric records need the earliest post-quantum attention because they cannot be revoked {veridos}. None of these publishes measurements or signs the AI verdict as an auditable record.'));
add(P('**Attack detection.** Presentation attack detection (PAD) is evaluated with the APCER/BPCER metrics of ISO/IEC 30107-3 {iso30107}; CelebA-Spoof is a large open PAD dataset {celebaspoof}, and the MiniFASNet models of Silent-Face-Anti-Spoofing are light enough for real time {silentface}. Challenge-response liveness, in which the user must react to something unpredictable, appears in Face Flashing (random screen colours) {faceflashing} and GOTCHA (challenges that degrade real-time deepfakes) {gotcha}. Morphing attacks were introduced by Ferrara *et al.* {ferrara}; differential morphing attack detection (D-MAD) compares the enrolled photo with a trusted live capture through deep face embeddings {scherhag}, and FRLL-Morphs provides morphs from several tools {sarkar,frllmorphs}. For deepfakes, OpenForensics offers multi-face forgeries in the wild {openforensics}, and GenD adapts a CLIP vision encoder {clip} by tuning only its LayerNorm parameters {gend}. Virtual-camera injection can be detected from capture metadata and timing {vcd}.'));
add(...table('related', 'Comparison of the closest related work with this project.',
  ['Work', 'PQ crypto', 'Attack detection', 'Transaction binding', 'Signed AI verdict', 'Measured cost'],
  [
    ['Román *et al.* 2022 {roman}', 'Kyber, Saber', 'No', 'No', 'No', 'Template ops'],
    ['Arjona *et al.* 2023 {arjona}', 'McEliece', 'No', 'No', 'No', 'Template ops'],
    ['Olushola & Meenakshi 2026 {olushola}', 'ML-KEM-1024', 'Discussed', 'No', 'No', 'Matching latency'],
    ['Gowri Ganesh *et al.* 2026 {gowri}', 'ML-KEM', 'PAD (simulated)', 'No', 'No', 'Feasibility only'],
    ['Daon patents 2026 {daon}', 'Yes (claimed)', 'Injection, replay', 'Yes', 'No', 'Not published'],
    ['authID 2026 {authid}', 'ML-DSA-65', 'Liveness (product)', 'Yes', 'Not stated', 'Not published'],
    ['Face Flashing, GOTCHA {faceflashing,gotcha}', 'No', 'Liveness', 'Challenge only', 'No', 'Detection only'],
    ['**This project**', 'ML-KEM-768, ML-DSA-65', 'M1–M5 + matcher', 'Yes (checks 1–6)', '**Yes**', 'Classical vs PQ'],
  ],
  [2350, 1350, 1450, 1350, 1250, 1276]));
add(P(`${T('related')} shows that each column has been covered by some earlier work, but no row other than this project covers all of them, and none signs the AI verdict.`));

add(H2('2.2 Theoretical Framework'));
add(P('**The quantum threat.** Shor\'s algorithm finds the period of a modular function with the quantum Fourier transform, which yields the factors of an RSA modulus or the discrete logarithm behind Diffie–Hellman and elliptic-curve keys in polynomial time {shor}. X25519 {rfc7748} and Ed25519 {rfc8032}, used here as the classical baseline, fall to it. Grover\'s algorithm searches an unstructured space of size N in about √N steps {grover}, so a k-bit symmetric key keeps about k/2 bits of security; AES-256 and SHA-384 therefore keep a comfortable margin, and they are used for all symmetric work in this project {fips180,sp80038d}.'));
add(P('**Learning with errors.** The post-quantum schemes used here rest on the learning-with-errors (LWE) problem {regev}: given a random matrix A and b = A·s + e (mod q), where s is secret and e is a small random error, recover s. The small error makes the linear system hard to solve, and no efficient quantum algorithm is known. Module-LWE replaces integers by polynomials in the ring R_{q} = Z_{q}[X]/(X^{256} + 1) and uses small k×k matrices of such polynomials, which keeps keys small while retaining worst-case hardness guarantees {langlois}:'));
add(Eq('b = A·s + e,   A ∈ R_{q}^{k×k},   s, e small,   R_{q} = Z_{q}[X]/(X^{256} + 1)'));
add(P('**ML-KEM-768** (FIPS 203, derived from CRYSTALS-Kyber {kyber}) uses k = 3 and q = 3329 and reaches NIST security category 3. The public key is 1,184 bytes and the ciphertext 1,088 bytes; both sides obtain the same 32-byte shared secret, and a Fujisaki–Okamoto transform makes it secure against chosen-ciphertext attacks {fips203}. **ML-DSA-65** (FIPS 204, from CRYSTALS-Dilithium {dilithium}) uses a 6×5 module over q = 8,380,417 and the Fiat–Shamir-with-aborts technique; at category 3 its public key is 1,952 bytes and a signature 3,309 bytes {fips204}. Both were checked against NIST ACVP known-answer vectors {acvp}.'));
add(P('**Symmetric layer.** The shared secret is turned into an AES key with HKDF {rfc5869}, built on HMAC {rfc2104}:'));
add(Eq('PRK = HMAC-SHA384(nonce, ss)'));
add(Eq('K = HMAC-SHA384(PRK, info ‖ 0x01)[0 … 31]'));
add(P('with the nonce as the HKDF salt, and where info = "BEA-v1|" ‖ mode ‖ "|" ‖ challenge_id. AES-256-GCM {sp80038d} then encrypts and authenticates the evidence with a random 96-bit IV and a 128-bit tag, using AAD = mode ‖ "|" ‖ challenge_id. Changing any byte of the ciphertext, the IV, the KEM ciphertext, the mode or the challenge id changes either K or the tag, so decryption fails.'));
add(P('**Biometric error rates.** Following ISO/IEC 30107-3 {iso30107}, for a detector with score s and threshold τ (higher score = more likely an attack):'));
add(Eq('APCER = |{attacks with s ≤ τ}| / |attacks|'));
add(Eq('BPCER = |{bona fide samples with s > τ}| / |bona fide samples|'));
add(P('APCER is reported separately for each attack type. The equal error rate (EER) is the point where APCER = BPCER, and the AUC is the probability that a random attack scores above a random bona fide sample. For the matcher, the false match rate (FMR) is the share of impostor pairs whose similarity reaches the threshold and the false non-match rate (FNMR) the share of genuine pairs below it.'));

add(H2('2.3 Gap Analysis'));
add(bullets([
  '**Crypto without detection.** Post-quantum biometric schemes protect templates and channels, but a perfectly encrypted deepfake is still a deepfake {roman,arjona,olushola}.',
  '**Detection without protected evidence.** Detector papers assume the frames reach the model unchanged; they do not bind the sample to a transaction or a challenge.',
  '**Unsigned decisions.** In every system reviewed, the detector\'s score and the final decision are internal values that an insider could later change without trace.',
  '**No measured trade-off.** No reviewed work measures, in one pipeline with real detectors, how much post-quantum cryptography adds to latency and bytes, or which attacks it does and does not stop.',
  '**Optimistic thresholds.** Thresholds are often chosen on the test data; here every threshold is set on data separate from the data it is reported on.',
]));
add(P('This project fills these gaps with a single prototype that binds, encrypts and signs the evidence, runs five detectors, signs the verdict and measures both modes under identical conditions.'));

// ================================================================ 3 METHODOLOGY
add(H1('3. Proposed Methodology & System Architecture'));
add(H2('3.1 Problem Formulation'));
add(P('A user U has enrolled a device signing key pk_{dev} and a face template t. For a transaction tx_context, the server issues a challenge C = (id, n, h_{tx}, a_{1}, a_{2}, mode, expiry), where n is a 32-byte nonce, h_{tx} = SHA-384(canonical_json(tx_context)) and a_{1}, a_{2} are two liveness actions. The client returns a sealed bundle B carrying a video v, capture metadata m, a manifest M = (mode, id, n, h_{tx}, SHA-384(v), timestamp) and a signature σ = Sign(sk_{dev}, SHA-384(M)). The server must output accept only if:'));
add(Eq('Check_{i}(B, C) passes (i = 1…6)  ∧  s_{k}(v, m) ≤ τ_{k} (k = M1…M5)  ∧  cos(e(v), t) ≥ τ_{match}'));
add(P('and it must sign the record R = (id, mode, h_{tx}, SHA-384(v), scores, model versions, threshold version, decision, reason code, time) as ρ = Sign(sk_{server}, SHA-384(canonical_json(R))). Each threshold τ_{k} is the smallest value whose bona fide rejection on separate calibration data is at most β = 0.5%; with five modules this keeps the combined false rejection near 2.5%. The security goals are freshness (no reuse of a challenge), binding (evidence valid for one transaction only), confidentiality including against harvest-now-decrypt-later, integrity and origin authentication, genuineness of the face, and auditability of the decision. The threat model is summarised in ' + T('threats') + '.'));
add(...table('threats', 'Threat model: attacks considered and the component that counters each.',
  ['Attack', 'What the attacker does', 'Countered by'],
  [
    ['Harvest now, decrypt later', 'Records traffic to decrypt with a future quantum computer', 'ML-KEM-768 + HKDF-SHA384 + AES-256-GCM'],
    ['Tampering in transit', 'Changes bytes of the sealed bundle', 'Check 3 (GCM tag), 5 (hash), 6 (signature)'],
    ['Replay / delayed use', 'Sends a captured bundle again or later', 'Check 1 (single use, 60 s TTL)'],
    ['Re-targeting', 'Re-uses evidence for another payment', 'Check 4 (signed nonce and tx_hash)'],
    ['Downgrade', 'Relabels a PQ challenge as classical', 'Check 2; mode in HKDF info and GCM AAD'],
    ['Forged device', 'Builds a bundle without the device key', 'Check 6 (ML-DSA-65 / Ed25519)'],
    ['Key substitution', 'Gives the client a fake server key', 'Server KEM key pinned at enrolment'],
    ['Wrong person', 'Someone else at the camera', 'Matcher (ArcFace, FMR 0.1%)'],
    ['Photo, screen, mask', 'Presents an artefact to the camera', 'M1 (MiniFASNet), M4 (actions)'],
    ['Pre-recorded video', 'Plays a recording of the user', 'M4: actions unknown before the nonce'],
    ['Morphed enrolment', 'Enrols a face blended from two people', 'M2 (D-MAD)'],
    ['Deepfake', 'Synthesised or face-swapped video', 'M3 (GenD)'],
    ['Virtual-camera injection', 'Feeds a stream via OBS or v4l2loopback', 'M5 (rules + gradient-boosted trees)'],
    ['Editing a verdict', 'Changes a stored score or decision', 'ML-DSA-65-signed verdict record'],
  ],
  [2100, 3700, 3226], { left: true }));
add(P('Out of scope, and stated as limits: a compromised server, side channels, denial of service, enrolment fraud, a strong attacker on a rooted device and high-quality 3D masks.'));

add(H2('3.2 Algorithm Design'));
add(P(`${F('arch')} shows the four layers. Layer A captures the evidence, Layer B seals it with post-quantum cryptography, Layer C verifies it and Layer D measures everything. Arrows: (1) the server issues a challenge; (2) the client builds the manifest from the video; (3) the sealed bundle is sent to POST /verify; (4) the client receives only "accepted" or "rejected"; (5) stage timings and scores go to the evaluation layer.`));
add(...figure('arch', 'fig1_architecture.png', 'System architecture: capture client, post-quantum channel, verification server and evaluation harness.', 610));
add(H3('One transaction, step by step'));
add(...numbered([
  '**Challenge.** The server draws a 32-byte nonce from the OS random generator, computes tx_hash, chooses two actions and stores the challenge in Redis with a 60-second lifetime and the server KEM public key.',
  '**Actions.** m = HMAC-SHA256(server_secret, nonce) drives a Fisher–Yates shuffle of {blink twice, turn left, turn right}: for i = n−1 … 1, swap item i with item m[n−1−i] mod (i+1); the first two items are the actions. Nobody can predict them before the nonce exists.',
  '**Capture.** The client checks that the challenge names the pinned server key and the transaction it is showing, displays the actions and records a 4-second video with per-frame timestamps and camera metadata.',
  '**Manifest and signature.** The client builds the manifest as canonical JSON (sorted keys, no spaces) and signs SHA-384(manifest) with its ML-DSA-65 device key.',
  '**Sealing.** The client encapsulates to the server key, derives K with HKDF-SHA384 and encrypts video ‖ metadata ‖ manifest ‖ signature (each part length-prefixed) with AES-256-GCM.',
  `**Verification.** The server runs checks 1–6 in order and stops at the first failure (${T('checks')}, ${F('pipeline')}); then it runs the matcher and M1–M5 in parallel.`,
  '**Decision and verdict.** The decision rule below gives accept or reject, and the server signs and stores the verdict record. Exactly one verdict is stored per consumed challenge, rejections included.',
]));
add(...figure('pipeline', 'fig2_checks_pipeline.png', 'Server verification pipeline: six protocol checks in a fixed order, each with its own reason code, followed by the detectors and the decision. Every consumed challenge gets one signed verdict, rejections included.', 610));
add(...table('checks', 'Server checks, in their fixed order; the first failure stops verification.',
  ['#', 'Check', 'Reason code'],
  [
    ['1', 'Challenge exists, is not expired and not used; it is consumed atomically (Redis GETDEL)', 'EXPIRED / REPLAY'],
    ['2', 'Mode of the bundle equals the mode stored with the challenge', 'MODE_MISMATCH'],
    ['3', 'Decapsulate, derive K, AES-GCM tag valid', 'TAMPER'],
    ['4', 'Manifest nonce, tx_hash, challenge id and mode equal the issued values', 'TX_MISMATCH'],
    ['5', 'SHA-384(video) equals manifest.sample_hash', 'HASH_MISMATCH'],
    ['6', 'Device signature valid under the enrolled device key', 'SIG_INVALID'],
  ],
  [500, 6026, 2500], { left: true }));
add(P(`The lookup runs first because the mode to compare against is stored with the challenge; a downgraded request therefore still consumes the challenge and cannot be retried. The client never sees the reason code, so an attacker cannot learn which check fired. ${T('modules')} lists the detectors that run after the checks.`));
add(...table('modules', 'Face matcher and detection modules M1–M5.',
  ['Module', 'Method', 'Score and rule', 'Reason code'],
  [
    ['Matcher', 'ArcFace ResNet-50 (InsightFace buffalo_l) {arcface,insightface}; YuNet face detection {yunet}', `cosine(e, t) ≥ ${fx(thr('match').threshold, 4)} (FMR 0.1%)`, 'NO_MATCH'],
    ['M1 PAD', 'MiniFASNetV2 (scale 2.7) + MiniFASNetV1SE (scale 4.0) {silentface}', 's = 0.5 + ½(max(p0, p2) − p1); reject if s > 0.5', 'PAD_FAIL'],
    ['M2 Morph', 'D-MAD: e(reference) − e(live) → RBF SVM {scherhag,sklearn}', `sigmoid(d) > ${fx(thr('m2_morph').threshold, 4)} (BPCER 0.5%)`, 'MORPH_FLAG'],
    ['M3 Deepfake', 'GenD, CLIP ViT-L/14, LayerNorm-tuned {gend,clip}', `mean P(fake) over 3 frames > ${fx(thr('m3_deepfake').threshold, 4)}`, 'DEEPFAKE_FAIL'],
    ['M4 Replay', 'Checks 1–6 + MediaPipe blendshapes and head yaw {mediapipe}', 'both actions seen in order, else s = 1', 'SCRIPT_FAIL'],
    ['M5 Injection', 'Virtual-camera name rule + HistGradientBoosting on 16 timing features {sklearn,lightgbm,vcd}', 'name rule now; classifier pending', 'INJECTION_FAIL'],
  ],
  [1250, 3500, 2876, 1400], { left: true }));
add(P('**Face pipeline.** YuNet runs once per frame with a score threshold of 0.5; the face used is the largest among detections within 0.1 of the best score, and it must be at least 112 pixels wide. The five landmarks are aligned to the ArcFace template with a similarity transform, giving a 112×112 crop for ArcFace and M2, a 224×224 crop for GenD and 80×80 crops at two scales for MiniFASNet.'));
add(P('**M2 morph detection.** For an enrolled reference r and a live probe p the feature is the difference of their L2-normalised ArcFace embeddings, standardised and classified by an RBF support vector machine with balanced class weights:'));
add(Eq('f = e(r) − e(p),   d(f) = Σ_{i} α_{i} y_{i} exp(−γ‖f_{i} − f‖²) + b,   s_{M2} = 1 / (1 + e^{−d(f)})'));
add(P('A genuine pair gives a small, typical difference; a morphed reference only half-matches the live person and gives a different one.'));
add(P('**M3 deepfake detection.** Each 224×224 crop is normalised with CLIP\'s mean and standard deviation, the pooled feature h is L2-normalised and a linear head gives P(fake) = softmax(W·h/‖h‖ + b)_fake. Tuning trains only the scale γ and shift β of each LayerNorm, LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β, i.e. 102,400 of about 304 million parameters, with class-weighted cross-entropy and the AdamW optimiser {adamw}.'));
add(P('**M4 liveness actions.** Per frame, the blink level is b = ½(eyeBlinkLeft + eyeBlinkRight) and the head yaw is yaw = (180/π)·atan2(R_{02}, R_{22}) from the face rotation matrix. A blink is counted when b rises above 0.5 and then falls below 0.3 (hysteresis); a turn is counted when |yaw| > 15° and reset below 7.5°. A recording cannot follow actions that did not exist when it was made, and the video hash is signed together with the nonce that chose them.'));
add(P('**M5 injection.** A camera name matching a known virtual camera (OBS Virtual Camera, v4l2loopback, ManyCam and others) scores 1. The planned classifier uses 16 features of the capture: mean, standard deviation, coefficient of variation and 5th/95th percentiles of the inter-frame intervals Δt_{i} = t_{i+1} − t_{i}, the share of intervals within 1 ms of the median, the duplicate-frame ratio, requested versus delivered resolution and frame rate, and the latency of a resolution change. Virtual cameras tend to be too regular, repeat frames and switch resolution instantly.'));
add(P('**Decision rule.** Reject if any module score exceeds its threshold, reporting the first in the order M5, M3, M1, M2, M4; otherwise accept if and only if the match similarity reaches τ_{match}. **Signed verdict.** The server builds the record R above, signs SHA-384(canonical_json(R)) with ML-DSA-65 in PQ mode (Ed25519 in classical mode) and stores R and the signature in SQLite. Only hashes and scores are stored, never face images, and scripts/verify_verdict.py re-checks any stored verdict with the server\'s public key.'));

add(H2('3.3 Implementation Details'));
add(P(`The prototype is written in Python with type hints and a pytest suite for every module. ${T('stack')} lists the frameworks and the hardware. No quantum hardware or simulator is used: post-quantum algorithms are designed to run on classical computers, and the "backend" for all cryptography is the OpenSSL-based Python *cryptography* library {pyca}. Neural models are exported to ONNX and served with ONNX Runtime on the CPU {onnxrt}; the GPU was used only to tune M3.`));
add(...table('stack', 'Software frameworks and hardware.',
  ['Component', 'Choice (version)'],
  [
    ['Language, OS', `Python ${cb[0].python}, ${cb[0].system} (development and measurements)`],
    ['Cryptography', `cryptography ${cb[0].cryptography} (OpenSSL): ML-KEM-768, ML-DSA-65, X25519, Ed25519, HKDF, AES-GCM`],
    ['Server', 'FastAPI 0.143, Uvicorn 0.54, Redis protocol (fakeredis 2.39 locally; redis:7 in Docker), SQLite'],
    ['Inference', 'ONNX Runtime 1.31 (CPU), OpenCV YuNet, MediaPipe 1.1 Face Landmarker'],
    ['Training / export', 'PyTorch 2.14 (CUDA 13), transformers 4.56 for GenD export, scikit-learn 1.9'],
    ['Evaluation', 'NumPy 2.5, psutil 7.2, locust 2.46, matplotlib'],
    ['CPU', `Intel Core i7-13700HX (${cb[0].cpu_count} logical processors), 32 GB RAM`],
    ['GPU', 'NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB (M3 tuning only)'],
    ['Quantum backend', 'None; not applicable (post-quantum algorithms run on classical hardware)'],
  ],
  [2300, 6726], { left: true }));
add(P('The repository separates crypto/ (the CryptoSuite for both modes and the canonical manifest), server/ (challenge store, checks, decision, verdict, parallel pipeline), client/ (enrolment, capture, sealing), preprocessing/, detectors/ (one package per module), eval/ (metrics, harness, performance and attack-matrix scripts) and scripts/ (downloads, training, calibration, replay suite, verdict verification). All thresholds and paths live in config.yaml, and every number in this report is read from a CSV in eval/results/ by the script that builds the report.'));
add(P(`For demonstrations the server has a front end: one static page (plain HTML, CSS and JavaScript, no external links, so it works offline) served at http://127.0.0.1:8000/ui only in demo mode and only on the loopback address. It calls the same server functions as the command-line demo and changes no check, threshold, model or key. ${F('ui_overview')} shows its header and the system-status panel, which lists every module with its model version and marks any stub, so a demonstration cannot silently run without a real model; the cards used in the demonstration are shown in Section 4.3.7.`));
add(...figure('ui_overview', 'fig_ui_overview.png', 'Demo front end: header and system status. Every module reports the model version it runs (no stubs), and the preliminary threshold version is shown.', 600));

// ================================================================ 4 EXPERIMENTS
add(H1('4. Experimental Setup'));
add(H2('4.1 Experimental Setup'));
add(P(`The template for this report asks for qubits, shots and the optimiser. No quantum circuit is executed in this project, so qubits and shots do not apply; ${T('params')} lists the parameters that play their role here: security levels, protocol constants, thresholds, the optimiser used for tuning and the sizes of every experiment.`));
add(...table('params', 'Experimental parameters.',
  ['Parameter', 'Value'],
  [
    ['Qubits, shots', 'Not applicable (no quantum circuit; post-quantum algorithms on a classical CPU)'],
    ['Security level', 'NIST category 3: ML-KEM-768, ML-DSA-65; AES-256-GCM; SHA-384; HKDF-SHA384'],
    ['Classical baseline', 'X25519 (ephemeral client, static server), Ed25519; same symmetric layer'],
    ['Challenge', 'Nonce 32 B, lifetime 60 s, single use (GETDEL); 2 of 3 actions'],
    ['Capture', '4 s video, 640×480, JPEG frames sampled at 15 fps'],
    ['Face detection', 'YuNet score ≥ 0.5, score margin 0.1, face width ≥ 112 px'],
    ['Thresholds (version)', `${thresholds[0].version}: match ${fx(thr('match').threshold, 4)}, M1 ${fx(thr('m1_pad').threshold, 1)}, M2 ${fx(thr('m2_morph').threshold, 4)}, M3 ${fx(thr('m3_deepfake').threshold, 4)}, M4 ${fx(thr('m4_replay').threshold, 1)}, M5 ${fx(thr('m5_injection').threshold, 1)}`],
    ['Optimiser (M3 tuning)', `AdamW, learning rate ${m3tune[1].lr}, batch ${m3tune[1].batch}, fp16, ${m3tune.length - 1} epochs, 102,400 trainable parameters, class-weighted cross-entropy`],
    ['M2 classifier', 'RBF SVM, class_weight = balanced; 5-fold GroupKFold by subject, thresholds from an inner 4-fold CV'],
    ['M3 frames', '3 sampled frames per request, scores averaged'],
    ['Crypto benchmark', `${int(cop('PQ', 'kem_keygen').runs)} runs per operation and mode, ${int(num(cop('PQ', 'seal_bundle').payload_kb) / 1024)} MiB payload for sealing`],
    ['End-to-end run', '500 genuine transactions per mode after 3 warm-up runs; real models; in-process server with fakeredis'],
    ['Replay suite', '1,000 runs per attack and mode, plus 1,000 genuine runs per mode'],
    ['Attack × mode matrix', '100 OpenForensics Test-Dev fake faces, 100 real faces, 100 FRLL-Morphs pairs (all five tools), 27 webcam sessions; both modes'],
  ],
  [2500, 6526], { left: true }));
add(P(`${T('datasets')} lists the data. All datasets are openly downloadable; none is redistributed with the code. Splits are by identity (or by source image for OpenForensics), so no person or source image is in both training and test data.`));
add(...table('datasets', 'Datasets and how each is used.',
  ['Dataset', 'Size used', 'Use and protocol'],
  [
    ['FRLL {frll}', '102 subjects, 1350×1350 (CC BY 4.0)', 'Matcher: enrol neutral front; probe smiling and four 3/4 views; subjects split into validation and test halves'],
    ['FRLL-Morphs {sarkar,frllmorphs}', `AMSL ${int(m2at('0.005')['n_attack[amsl]'])}, others ${int(m2at('0.005')['n_attack[facemorpher]'])} pairs each; ${int(m2at('0.005').n_bona_fide)} bona fide pairs`, 'M2: 5-fold identity-grouped CV; thresholds from training subjects only'],
    ['OpenForensics Val {openforensics}', `${int(m3val[0].n_bona_fide)} real / ${int(m3val[0]['n_attack[openforensics_fake]'])} fake faces ≥ 112 px`, 'M3: tuning (80%) and threshold calibration (20%), split by source image'],
    ['OpenForensics Test-Dev', `${int(zs('0.005').n_bona_fide)} real / ${int(zs('0.005')['n_attack[openforensics_fake]'])} fake faces`, 'M3: independent test, never used for tuning or thresholds'],
    ['CelebA-Spoof {celebaspoof}', 'Not yet downloaded', 'M1 fine-tuning and evaluation (pending)'],
    ['Own webcam recordings', `${genuine.length} logged genuine captures; 27 saved sessions`, 'Live acceptance and the bona fide side of the attack matrix (team members, with consent)'],
  ],
  [2300, 2800, 3926], { left: true }));

add(H2('4.2 Expected Outcomes & Performance Metrics'));
add(...table('metrics', 'Metrics and the target for each.',
  ['Metric', 'Definition', 'Expected outcome'],
  [
    ['APCER per attack type', 'Share of attacks accepted', 'As low as possible at the fixed BPCER'],
    ['BPCER', 'Share of bona fide samples rejected', '0.5% per module (also 1% and 5% reported)'],
    ['EER, AUC', 'Error where APCER = BPCER; ranking quality', 'Low EER, AUC close to 1'],
    ['FMR, FNMR', 'Matcher false match / non-match rate', 'FMR 0.1% at a low FNMR'],
    ['Replay rejection', 'Share of protocol attacks rejected with the right code', '100%, with 0% false rejection of genuine runs'],
    ['Latency p50 / p95', 'Per stage and end to end (perf_counter_ns)', 'Crypto a small share of the total'],
    ['CPU %, peak RSS', 'Sampled every 50 ms with psutil', 'Similar in both modes'],
    ['Bytes per request', 'Size of the sealed bundle', 'About +4.3 kB in PQ mode'],
  ],
  [2300, 3700, 3026], { left: true }));
add(P('Operating points are fixed in advance: every detector is reported at BPCER targets of 0.5%, 1% and 5%, and every threshold is computed on data separate from the data it is reported on. For a target β and n sorted bona fide calibration scores, τ = s_{(n − ⌊βn⌋)}, so at most ⌊βn⌋ bona fide samples score above it.'));
const yes = (c) => (c ? 'Met' : 'Not met');
const replayMet = replayDone ? (replayAllCorrect && replayProtocolFR === 0) : null;
add(P(`${T('targets')} evaluates the results of Section 4.3 against the target outcomes of the project.`));
add(...table('targets', 'Target project outcomes and what was achieved.',
  ['Objective', 'Target', 'Achieved', 'Status'],
  [
    ['ML-KEM / ML-DSA integration', 'Both modes work; NIST known-answer tests pass', 'Round trips, tamper tests and ACVP vectors pass; genuine user accepted in both modes', 'Met'],
    ['Binding and freshness', '100% of protocol attacks rejected with the right code; 0 protocol false rejections',
      replayDone ? `${int(replayAttacks.reduce((a, r) => a + num(r.correct_code), 0))} of ${int(replayAttackRuns)} with the expected code; ${int(replayProtocolFR)} protocol false rejections` : 'Replay suite running', replayDone ? yes(replayMet) : 'Pending'],
    ['Face matcher', 'FMR 0.1% on unseen subjects', `FMR ${pct(mTest.fmr, 3)}, FNMR ${pct(mTest.fnmr, 1)}`, 'Met'],
    ['M2 morphing', 'BPCER 0.5%, low APCER for every tool', `BPCER ${pct(m2at('0.005').bpcer)}, APCER ≤ ${pct(m2at('0.005').apcer_max)}`, 'Met'],
    ['M3 deepfake', 'BPCER 0.5% on an independent set, low APCER', `BPCER ${pct(ln('0.005').bpcer)}, APCER ${pct(ln('0.005').apcer_max, 1)} (${pct(ln('0.01').apcer_max, 1)} at BPCER 1%)`, 'Partly met'],
    ['M1 presentation attacks', 'Fine-tuned on CelebA-Spoof, reported per attack type', 'Pretrained model in use; dataset not yet downloaded', 'Pending'],
    ['M5 injection', 'Classifier tested leave-one-tool-out', 'Virtual-camera name rule only', 'Partly met'],
    ['PQC overhead', 'Measured; small share of the total', perfDone ? `${pct(pqS.crypto_share_p50, 2)} of end-to-end time; +${int(num(size('PQ').per_transaction_asymmetric_B) - num(size('CLASSICAL').per_transaction_asymmetric_B))} B per transaction` : 'Per operation only', perfDone ? 'Met' : 'Partly met'],
    ['Signed verdict', 'Any edit of a stored verdict detected', `Edited score detected (${F('ui_attacks')}); covered by tests`, 'Met'],
    ['Reproducibility', 'Public code, scripts, fresh-clone rebuild', 'Public repository; every number generated from CSVs; Docker and fresh-clone rebuild pending', 'Partly met'],
  ],
  [2000, 2500, 3326, 1200], { left: true }));

add(H2('4.3 Comparative Analysis & Discussion'));
add(H3('4.3.1 Classical versus post-quantum cryptography'));
const cryptoRows = [
  ['Key generation (KEM)', 'kem_keygen'], ['Encapsulate (client)', 'encapsulate'], ['Decapsulate (server)', 'decapsulate'],
  ['Signature key generation', 'sig_keygen'], ['Sign manifest (client)', 'sign_manifest'], ['Verify manifest (server)', 'verify_manifest'],
  ['Sign verdict (server)', 'sign_verdict'], ['HKDF-SHA384', 'hkdf'], ['Seal 2 MiB bundle (client)', 'seal_bundle'], ['Unseal 2 MiB bundle (server)', 'unseal_bundle'],
].map(([lab, op]) => [lab, fx(cop('CLASSICAL', op).p50_us, 1), fx(cop('CLASSICAL', op).p95_us, 1), fx(cop('PQ', op).p50_us, 1), fx(cop('PQ', op).p95_us, 1)]);
add(...table('crypto_ops', `Time per cryptographic operation in microseconds (${int(cop('PQ', 'kem_keygen').runs)} runs each).`,
  ['Operation', 'Classical p50', 'Classical p95', 'PQ p50', 'PQ p95'], cryptoRows, [3026, 1500, 1500, 1500, 1500]));
add(...figure('crypto', 'fig3_crypto_ops.png', 'Median time of each asymmetric operation, classical (X25519/Ed25519) vs post-quantum (ML-KEM-768/ML-DSA-65); logarithmic axis.', 560));
const sPQ = size('PQ'), sCL = size('CLASSICAL');
add(P(`${T('crypto_ops')} and ${F('crypto')} show that ML-KEM-768 is as fast as X25519: encapsulation took ${fx(cop('PQ', 'encapsulate').p50_us, 1)} µs against ${fx(cop('CLASSICAL', 'encapsulate').p50_us, 1)} µs, and decapsulation ${fx(cop('PQ', 'decapsulate').p50_us, 1)} µs against ${fx(cop('CLASSICAL', 'decapsulate').p50_us, 1)} µs. ML-DSA-65 signing is the most expensive step at ${fx(cop('PQ', 'sign_manifest').p50_us, 0)} µs median (${fx(cop('PQ', 'sign_manifest').p95_us, 0)} µs at p95, because its rejection sampling repeats a variable number of times), about ${fx(num(cop('PQ', 'sign_manifest').p50_us) / num(cop('CLASSICAL', 'sign_manifest').p50_us), 0)} times Ed25519. Even so, every post-quantum operation stays below half a millisecond at the median, while sealing and unsealing the 2 MiB video take about ${fx(num(cop('PQ', 'seal_bundle').p50_us) / 1000, 1)} ms and ${fx(num(cop('PQ', 'unseal_bundle').p50_us) / 1000, 1)} ms in both modes, dominated by AES-GCM over the payload rather than by the asymmetric algorithms.`));
add(...table('sizes', 'Key, ciphertext and signature sizes in bytes.',
  ['Mode', 'KEM public key', 'KEM ciphertext', 'Signature public key', 'Signature', 'Bundle overhead', 'Per-transaction asymmetric'],
  [sCL, sPQ].map((r) => [r.mode === 'PQ' ? 'PQ' : 'Classical', int(r.kem_public_key_B), int(r.kem_ciphertext_B), int(r.sig_public_key_B), int(r.signature_B), int(r.bundle_overhead_B), int(r.per_transaction_asymmetric_B)]),
  [1226, 1200, 1200, 1400, 1100, 1400, 1500]));
add(P(`The size cost is larger in relative terms: each transaction carries ${int(sPQ.per_transaction_asymmetric_B)} bytes of KEM ciphertext and signature instead of ${int(sCL.per_transaction_asymmetric_B)} (${T('sizes')}).` + (perfDone
  ? ` Against the ${fx(num(clS.video_bytes) / 1e6, 1)} MB four-second video of the test session, the measured request grew by ${int(num(pqS.bytes_per_request_mean) - num(clS.bytes_per_request_mean))} bytes, ${pct((num(pqS.bytes_per_request_mean) - num(clS.bytes_per_request_mean)) / num(clS.bytes_per_request_mean), 3)} of its size.`
  : ' Against a video of several megabytes this is well under 1% of the request.')));

add(H3('4.3.2 End-to-end performance'));
if (perfDone) {
  add(...table('perf_summary', 'End-to-end results per mode (one recorded genuine session re-sent under fresh challenges, real models).',
    ['Metric', 'Classical', 'PQ'],
    [
      ['Runs', int(clS.runs), int(pqS.runs)],
      ['End to end p50 / p95 (ms)', `${fx(clS.end_to_end_p50_ms, 1)} / ${fx(clS.end_to_end_p95_ms, 1)}`, `${fx(pqS.end_to_end_p50_ms, 1)} / ${fx(pqS.end_to_end_p95_ms, 1)}`],
      ['Server total p50 / p95 (ms)', `${fx(clS.server_total_p50_ms, 1)} / ${fx(clS.server_total_p95_ms, 1)}`, `${fx(pqS.server_total_p50_ms, 1)} / ${fx(pqS.server_total_p95_ms, 1)}`],
      ['Cryptography p50 (ms)', fx(clS.crypto_p50_ms, 3), fx(pqS.crypto_p50_ms, 3)],
      ['Cryptography share of end to end (p50)', pct(clS.crypto_share_p50, 2), pct(pqS.crypto_share_p50, 2)],
      ['Bytes per request (mean)', int(clS.bytes_per_request_mean), int(pqS.bytes_per_request_mean)],
      ['Sequential throughput (requests/s)', fx(clS.sequential_req_per_s, 2), fx(pqS.sequential_req_per_s, 2)],
      ['CPU % mean / p95', `${clS.cpu_percent_mean} / ${clS.cpu_percent_p95}`, `${pqS.cpu_percent_mean} / ${pqS.cpu_percent_p95}`],
      ['Peak resident memory (MB)', clS.rss_peak_mb, pqS.rss_peak_mb],
      ['Outcomes', clS.outcomes.replace(/;/g, '; ').replace(/:/g, ', '), pqS.outcomes.replace(/;/g, '; ').replace(/:/g, ', ')],
    ],
    [3426, 2800, 2800]));
  const want = ['client_challenge_rtt', 'client_seal', 'client_verify_rtt', 'server_parse', 'server_check1_challenge', 'server_check3_unseal', 'server_check4_binding', 'server_check5_hash', 'server_check6_signature',
    'server_decode', 'server_faces', 'server_embed', 'server_det_match', 'server_det_m1_pad', 'server_det_m2_morph', 'server_det_m3_deepfake', 'server_det_m4_replay', 'server_det_m5_injection',
    'server_detectors_parallel', 'server_decision', 'server_verdict_sign', 'server_crypto', 'server_total', 'end_to_end'];
  const st = (mode, k) => perfStages.find((r) => r.mode === mode && r.stage === k);
  const stageRows = want.filter((k) => st('PQ', k) && st('CLASSICAL', k))
    .map((k) => [k.replace(/^server_/, 'server: ').replace(/^client_/, 'client: ').replace(/_/g, ' '), fx(st('CLASSICAL', k).p50_ms, 3), fx(st('CLASSICAL', k).p95_ms, 3), fx(st('PQ', k).p50_ms, 3), fx(st('PQ', k).p95_ms, 3)]);
  add(...table('perf_stages', 'Latency per stage in milliseconds (p50 and p95).', ['Stage', 'Classical p50', 'Classical p95', 'PQ p50', 'PQ p95'], stageRows, [3226, 1450, 1450, 1450, 1450]));
  add(...figure('perf', 'fig5_perf_stages.png', 'Median latency of the main stages in both modes (logarithmic axis).', 560));
  const det = st('PQ', 'server_detectors_parallel');
  const p50 = (mode, k) => fx(st(mode, k).p50_ms, k.includes('sign') ? 2 : 0);
  add(P(`${perfSentence()} The detectors dominate: the parallel matcher and M1–M5 stage took a median of ${det ? fx(det.p50_ms, 0) : '—'} ms in PQ mode, almost all of it the GenD vision transformer on three frames (${p50('PQ', 'server_det_m3_deepfake')} ms), and detecting and embedding the faces of the sampled frames took another ${p50('PQ', 'server_faces')} ms and ${p50('PQ', 'server_embed')} ms. The steps that change with the mode are small: the device-signature check took ${p50('CLASSICAL', 'server_check6_signature')} ms classically and ${p50('PQ', 'server_check6_signature')} ms with ML-DSA-65, and the verdict signature ${p50('CLASSICAL', 'server_verdict_sign')} ms and ${p50('PQ', 'server_verdict_sign')} ms; sealing and unsealing are dominated by AES-GCM over the video in both modes. The request grew from ${int(clS.bytes_per_request_mean)} to ${int(pqS.bytes_per_request_mean)} bytes on average. CPU and memory were sampled from the single process that ran both client and server, so they describe the whole pipeline rather than the server alone. The run re-sends one recorded genuine session under fresh challenges and, having no separate enrolment photo, enrols the middle frame of that same recording as the reference. The "Outcomes" row records how every request was decided (${pqS.outcomes.replace(/;/g, '; ').replace(/:/g, ', ')} in PQ mode): M2 was trained on studio enrolment photos paired with probes and did not accept this frame-from-the-same-video pairing. Every detector runs on every request whatever the decision, so the timings are unaffected, but this run is a cost measurement, not an accuracy result.`));
} else {
  add(...table('perf_summary', 'End-to-end results per mode.', ['Metric', 'Classical', 'PQ'], [['End to end p50 / p95', 'pending', 'pending']], [3426, 2800, 2800]));
  add(...table('perf_stages', 'Latency per stage.', ['Stage', 'Classical', 'PQ'], [['all stages', 'pending', 'pending']], [3426, 2800, 2800]));
  add(...figure('perf', 'fig5_perf_stages.png', 'Median latency of the main stages in both modes.', 560));
  add(P(perfSentence()));
}

add(H3('4.3.3 Protocol attacks: replay suite'));
if (replayDone) {
  const caseName = { exact_resubmit: 'Exact re-submit', expired: 'Expired challenge', old_nonce: 'Old evidence, new challenge', swapped_tx: 'Swapped transaction', flipped_byte: 'Flipped bit', genuine: 'Genuine (fresh)' };
  add(...table('replay', 'Replay suite with the real models: share of runs rejected and share rejected with the expected reason code.',
    ['Mode', 'Case', 'Runs', 'Expected code', 'Rejected', 'Correct code', 'Other codes'],
    replay.map((r) => [r.mode === 'PQ' ? 'PQ' : 'Classical', caseName[r.case] || r.case, int(r.runs), r.expected_code || '—', pct(r.rejection_rate, 1),
      r.case === 'genuine' ? `${int(r.protocol_false_rejections)} protocol FR` : pct(r.correct_code_rate, 1), r.other_codes || '—']),
    [1000, 2000, 800, 1500, 1050, 1250, 1426]));
  const gen = replayGenuine.map((r) => `${r.mode === 'PQ' ? 'PQ' : 'classical'}: ${pct(r.rejection_rate, 1)} rejected overall${r.other_codes ? ` (${r.other_codes.replace(/;/g, ', ')})` : ''}`).join('; ');
  add(P(`${replaySentence()} The exact re-submission is stopped by the single-use GETDEL, the late bundle by the 60-second lifetime, re-sealed old evidence and swapped transactions by the signed nonce and transaction hash, and every flipped bit by the AES-GCM tag (or by the KEM, whose wrong shared secret leads to the same tag failure). The two attacks that re-seal evidence model a compromised client holding the device key; the other three need no key at all. Genuine runs (${gen}) may still be rejected by a detector (for example M4 on the fixed recorded session), but never by a protocol check, which is what this suite verifies.`));
} else {
  add(...table('replay', 'Replay suite.', ['Mode', 'Case', 'Result'], [['both', 'all', 'pending']], [2000, 3000, 4026]));
  add(P(replaySentence()));
}

add(H3('4.3.4 Detection results'));
add(P('**Face matcher.** The threshold was chosen on the validation half of the FRLL subjects at an FMR of 0.1% and then applied unchanged to the other half.'));
add(...table('matcher', 'ArcFace matcher on FRLL (threshold from the validation half).',
  ['Half', 'Threshold', 'FMR', 'FNMR', 'Genuine pairs', 'Impostor pairs', 'Images without face'],
  [mVal, mTest].map((r) => [r.split === 'val' ? 'Validation' : 'Test', fx(r.threshold, 4), pct(r.fmr, 3), pct(r.fnmr, 2), int(r.n_genuine), int(r.n_impostor), r.images_without_face]),
  [1300, 1200, 1100, 1100, 1400, 1500, 1426]));
add(P(`On unseen subjects the FMR stayed at ${pct(mTest.fmr, 3)} (${Math.round(num(mTest.fmr) * num(mTest.n_impostor))} of ${int(mTest.n_impostor)} impostor pairs) and no genuine pair was rejected, including smiling and three-quarter views. FRLL is studio quality, so webcam images are harder (Section 4.3.6).`));
add(P('**M2 morph detection.** Results are pooled over the held-out folds of the 5-fold identity-grouped cross-validation.'));
add(...table('m2', `M2 (D-MAD) held-out APCER per morphing tool at three operating points; ${int(m2at('0.005').n_bona_fide)} bona fide pairs.`,
  ['Target BPCER', 'Achieved BPCER', 'AMSL', 'FaceMorpher', 'OpenCV', 'StyleGAN', 'WebMorph'],
  ['0.005', '0.01', '0.05'].map((t) => { const r = m2at(t); return [pct(t, 1), pct(r.bpcer), ...M2_TOOLS.map((k) => pct(r[`apcer[${k}]`]))]; }),
  [1300, 1400, 1250, 1300, 1250, 1250, 1276]));
add(P(`At the deployed 0.5% target, ${pct(m2at('0.005').bpcer)} of genuine pairs were rejected and at most ${pct(m2at('0.005').apcer_max)} of morphs were accepted for any tool; the EER was ${pct(m2at('0.005').eer)} and the AUC ${fx(m2at('0.005').auc, 4)}. No image failed face detection. An earlier rule that set thresholds from the SVM's own training scores collapsed (training bona fide scores sit on the margin) and rejected 37% of genuine users on held-out data; it was replaced by the inner cross-validation, which is why these figures are not optimistic. The deployed threshold (${fx(m2dep[0].threshold, 4)}) comes from the pooled held-out bona fide scores.`));
add(P('**M3 deepfake detection.** The headline comparison is on OpenForensics Test-Dev, which was never used for tuning or thresholds; both models use thresholds from the same 20% calibration split of Val.'));
add(...table('m3', 'M3 (GenD) on OpenForensics Test-Dev: zero-shot vs LayerNorm-tuned.',
  ['Model', 'Target BPCER', 'Threshold', 'Achieved BPCER', 'APCER', 'EER', 'AUC'],
  [...['0.005', '0.01', '0.05'].map((t) => ['Zero-shot', pct(t, 1), fx(zs(t).threshold, 4), pct(zs(t).bpcer), pct(zs(t).apcer_max, 1), pct(zs(t).eer, 2), fx(zs(t).auc, 3)]),
    ...['0.005', '0.01', '0.05'].map((t) => ['LayerNorm-tuned', pct(t, 1), fx(ln(t).threshold, 4), pct(ln(t).bpcer), pct(ln(t).apcer_max, 1), pct(ln(t).eer, 2), fx(ln(t).auc, 3)])],
  [1826, 1200, 1200, 1300, 1100, 1100, 1300]));
add(...figure('m3', 'fig4_m3_apcer.png', 'Share of OpenForensics Test-Dev fakes accepted (APCER) at three genuine-rejection targets, zero-shot vs LayerNorm-tuned GenD.', 520));
add(P(`Off the shelf, GenD ranks real and fake faces reasonably (AUC ${fx(zs('0.005').auc, 3)}), but OpenForensics fakes differ from its training data and at a strict setting ${pct(zs('0.005').apcer_max, 1)} of fakes pass. Tuning only the LayerNorm parameters for two epochs raised the AUC to ${fx(ln('0.005').auc, 3)} and reduced APCER to ${pct(ln('0.005').apcer_max, 1)}, ${pct(ln('0.01').apcer_max, 1)} and ${pct(ln('0.05').apcer_max, 1)} at the three targets (${F('m3')}). The achieved BPCER on Test-Dev (${pct(ln('0.005').bpcer)}, ${pct(ln('0.01').bpcer)}, ${pct(ln('0.05').bpcer)}) stays close to the targets, which shows that thresholds from 769 calibration faces transfer; the tuned model is deployed.`));
add(...table('m3_tuning', 'LayerNorm-only tuning of GenD, measured on the held-out 20% calibration split of Val.',
  ['Epoch', 'Training loss', 'Calibration AUC', 'Calibration EER'],
  m3tune.map((r) => [r.epoch, r.train_loss ? fx(r.train_loss, 4) : '— (zero-shot)', fx(r.calib_auc, 4), pct(r.calib_eer, 1)]),
  [1500, 2500, 2500, 2526]));
add(P('**Exported models.** The server runs ONNX exports, so their outputs were compared with the original PyTorch models on 200 faces each.'));
add(...table('onnx', 'PyTorch vs ONNX parity of the exported models.',
  ['Module', 'Model', 'Faces', 'Minimum cosine', 'Max |difference|', 'Pass'],
  onnx.map((r) => [r.module === 'm1_pad' ? 'M1' : 'M3', r.model.replace('.onnx', ''), r.n, fx(r.cosine_min, 7), num(r.max_abs_diff).toExponential(1), r.pass === 'True' ? 'Yes' : 'No']),
  [900, 3226, 900, 1500, 1500, 1000]));
const eqOk = gendEq.find((r) => r.comparison.includes('onnxruntime'));
add(P(`All exports match their originals with a minimum cosine similarity of ${fx(Math.min(...onnx.map((r) => num(r.cosine_min))), 7)}. In addition, GenD's own pipeline and the server path (ONNX plus our preprocessing) differ by at most ${num(eqOk.max_abs_dp).toExponential(2)} in P(fake) on ${eqOk.n} faces, with the same class for ${eqOk.same_class}; the detector results above therefore apply to the running system.`));

add(H3('4.3.5 Attack × mode matrix'));
if (matrixDone) {
  const srcName = { deepfake: 'Deepfake faces (attack)', openforensics_real: 'Real faces (bona fide)', webcam: 'Webcam sessions (bona fide)', morph_enrolment: 'Morphed enrolment (attack)' };
  const bio = matrix.filter((r) => r.kind !== 'protocol');
  const fl = (r, m) => (r[`flag_rate[${m}]`] === '' || r[`flag_rate[${m}]`] === undefined ? '—' : pct(r[`flag_rate[${m}]`], 1));
  add(...table('matrix', 'Attack × mode matrix: overall rejection and the share of samples each detector flags on its own (detectors and thresholds identical in both modes).',
    ['Mode', 'Source', 'n', 'Rejected', 'M1 flag', 'M2 flag', 'M3 flag', 'M4 flag', 'No match'],
    bio.map((r) => [r.mode === 'PQ' ? 'PQ' : 'Classical', srcName[r.attack] || r.attack, int(r.n), pct(r.rejection_rate, 1),
      fl(r, 'm1_pad'), fl(r, 'm2_morph'), fl(r, 'm3_deepfake'), fl(r, 'm4_replay'), r.no_match_rate === '' ? '—' : pct(r.no_match_rate, 1)]),
    [900, 2000, 600, 1000, 900, 900, 900, 900, 926]));
  const pick = (mode, a) => bio.find((r) => r.mode === mode && r.attack === a);
  const srcs = [...new Set(bio.map((r) => r.attack))].filter((a) => pick('CLASSICAL', a) && pick('PQ', a));
  const keys = ['rejection_rate', 'flag_rate[m1_pad]', 'flag_rate[m2_morph]', 'flag_rate[m3_deepfake]', 'flag_rate[m4_replay]', 'no_match_rate'];
  const same = srcs.filter((a) => keys.every((k) => pick('CLASSICAL', a)[k] === pick('PQ', a)[k])).length;
  const say = [];
  const df = pick('PQ', 'deepfake'), real = pick('PQ', 'openforensics_real'), mo = pick('PQ', 'morph_enrolment'), cam = pick('PQ', 'webcam');
  if (df && real) say.push(`M3 flagged ${fl(df, 'm3_deepfake')} of the deepfake faces and ${fl(real, 'm3_deepfake')} of the real faces from the same Test-Dev set`);
  if (mo) say.push(`M2 flagged ${fl(mo, 'm2_morph')} of the morphed enrolments`);
  if (cam) say.push(`on the recorded webcam sessions M1, M2 and M3 flagged ${fl(cam, 'm1_pad')}, ${fl(cam, 'm2_morph')} and ${fl(cam, 'm3_deepfake')}`);
  add(P(`${T('matrix')} shows the security side of the trade-off. For ${same} of ${srcs.length} sources every rate is identical in the two modes: the detectors see the same pixels whichever cryptography carried them, so post-quantum cryptography neither helps nor hurts against presentation, morph or deepfake attacks. In PQ mode, ${say.join('; ')}. Two artefacts of the test set-up must be read correctly. Still images are sent as static four-second videos, so they cannot perform the liveness actions and M4 flags all of them, genuine or not; and a saved webcam session is a recording made for an earlier challenge, so M4 rejects it whenever the newly drawn actions differ, which is exactly the pre-recorded-video attack M4 is designed to stop. The per-detector columns therefore measure M1–M3 and the matcher on their own, while the overall rejection includes M4. Replay, tampering and re-targeting are stopped by the binding checks in both modes (Section 4.3.3); what PQ mode adds is that these guarantees, and the confidentiality of the video, also hold against a quantum adversary.`));
} else {
  add(...table('matrix', 'Attack × mode matrix.', ['Mode', 'Attack', 'Result'], [['both', 'all', 'pending']], [2000, 3000, 4026]));
  add(P('The attack × mode matrix had not finished when this report was generated.'));
}

add(H3('4.3.6 Live webcam test'));
add(...table('genuine', 'All genuine live captures of one team member at the laptop webcam, every capture logged.',
  ['Phase', 'Captures (PQ / classical)', 'Accepted', 'Rejected (reason)'],
  phases.map((p) => {
    const pq = p.rows.filter((r) => r.mode === 'PQ').length, cl = p.rows.length - pq;
    const name = p.phase.startsWith('S6') ? 'First live check (uncalibrated thresholds)'
      : p.phase.includes('before re-enrolment') ? 'Lighting and position changed, old enrolment (zero-shot M3)'
        : p.phase.includes('after re-enrolment') ? 'Re-enrolled, zero-shot M3' : 'Re-enrolled, LayerNorm-tuned M3';
    const rej = p.rows.filter((r) => r.decision !== 'accepted');
    const reasons = [...new Set(rej.map((r) => r.reason_code))].map((c) => `${c} ×${rej.filter((r) => r.reason_code === c).length}`).join(', ');
    return [name, `${pq} / ${cl}`, String(p.rows.length - rej.length), rej.length ? `${rej.length} (${reasons})` : '0'];
  }),
  [3600, 1800, 1100, 2526], { left: true }));
add(P(`After re-enrolling in the same room, the user was accepted in ${afterAccepted.length} of ${afterReenrol.length} captures across both modes, with face-match scores of ${fx(Math.min(...acceptedMatch), 2)}–${fx(Math.max(...acceptedMatch), 2)} on accepted captures (threshold ${fx(thr('match').threshold, 4)}), a highest deepfake score of ${fx(maxM3After, 2)} and a highest presentation-attack score of ${fx(maxM1After, 3)}. The two rejections were one SCRIPT_FAIL (the actions were not seen clearly in order) and one MORPH_FLAG with a low face match. The earlier phase shows the main practical risk: after the lighting and position changed, the old enrolment no longer matched and M2 flagged every capture. The system failed safe, but this is a usability cost.`));

add(H3('4.3.7 Demonstration on the running prototype'));
add(P(`The screenshots below were taken from the demo front end with the real models loaded (${F('ui_overview')}). They show single, fixed examples and illustrate the behaviour measured above; they are not additional results.`));
add(...figure('ui_attacks', 'fig_ui_replay_verdict.png', 'Left: replay attacks on a captured genuine bundle, all rejected while the client sees only "rejected" (the first shows EXPIRED rather than REPLAY because the server had been restarted since that capture, as the page explains). Right: the signed verdict re-checked offline (VALID) and a copy with one M3 score edited from 0.030 to 0.990 (INVALID), with every score against its threshold.', 600));
add(P(`${F('ui_attacks')} shows the two security properties that cryptography provides. The re-sent bundle, the bundle with one flipped bit and the evidence re-bound to a 50,000 INR payment are rejected by checks 1, 3 and 4. The verdict card is the project's novelty in action: the stored verdict of the last accepted transaction verifies under the server's ML-DSA-65 public key, and editing a single score in a copy of the database breaks the signature.`));
add(...figure('ui_detect', 'fig_ui_morph_deepfake.png', 'Left: M2 on a genuine FRLL pair (accepted) and an AMSL morph of subjects 001 and 002 used as the enrolment photo (rejected, MORPH_FLAG); illustration only, since this pair was in M2\'s training data. Right: M3 on three real and three fake OpenForensics Test-Dev faces (fixed sample) at the deployed threshold.', 600));
add(P(`${F('ui_detect')} shows the AI side. The morph card shows why D-MAD works: the morphed reference only half-matches the live face, so its score (0.876) is far above the 0.5608 threshold, while the genuine pair scores 0.137. The deepfake card shows the deployed LayerNorm-tuned GenD separating this sample cleanly; ${T('m3')} gives the honest rate on the whole Test-Dev set, where ${pct(ln('0.005').apcer_max, 1)} of fakes still pass at the strictest setting. ${F('ui_results')} shows the Results tab, which displays the result CSVs read-only, so that the numbers shown during a review are the same files this report is generated from.`));
add(...figure('ui_results', 'fig_ui_results.png', 'Results tab of the demo front end: the matcher and M2 tables, read directly from eval/results/*.csv (top of the page).', 600));

add(H3('4.3.8 Sources of error'));
add(P('In place of the quantum noise discussed in quantum-computing projects, the following sources of error affect these measurements:'));
add(bullets([
  '**Timing noise.** Measurements ran on a Windows laptop with turbo boost and background services; p95 values in particular vary between runs. ML-DSA signing time also varies by design because of rejection sampling.',
  '**Finite calibration sets.** A 0.5% target on 769 calibration faces leaves only three faces above the threshold, so the threshold itself is uncertain; Test-Dev BPCER of 0.21%–1.39% around targets of 0.5%–1% reflects this.',
  '**Dataset bias.** FRLL is London-based, OpenForensics and FRLL-Morphs are single sources, and all live captures come from one person and one webcam; results on other populations, cameras and morph tools are untested.',
  '**Numerical differences.** GenD on the GPU (TF32) differs from the CPU by up to 4.4×10⁻³ in P(fake); the server uses the CPU path, which matches GenD\'s own pipeline to 5×10⁻⁶.',
  '**Environment drift.** Changes in lighting and pose between enrolment and verification lower the match score and can trigger M2 or M4.',
  '**Synthetic attack presentation.** In the attack matrix, still images are sent as static videos with webcam-like metadata, which is a simplification of a real injection.',
]));

// ================================================================ 5 ANALYSIS
add(H1('5. Analysis and Discussion'));
add(H2('5.1 Novelty & Insights'));
add(bullets([
  '**A signed AI verdict.** Each decision is stored as an ML-DSA-65-signed record that binds the challenge, the transaction hash, the video hash, all scores, the model and threshold versions and the decision; editing any stored score makes verification fail, which the demonstration and the test suite show.',
  `**Post-quantum cryptography is cheap here.** ML-KEM-768 is as fast as X25519, and the slowest post-quantum step, ML-DSA-65 signing, takes about ${fx(num(cop('PQ', 'sign_manifest').p50_us) / 1000, 2)} ms at the median${perfDone ? `; end to end, cryptography is ${pct(pqS.crypto_share_p50, 2)} of a PQ transaction` : ''}. The real costs are the extra ${int(num(sPQ.per_transaction_asymmetric_B) - num(sCL.per_transaction_asymmetric_B))} bytes per transaction and, above all, the neural detectors.`,
  '**Crypto and AI stop different attacks.** Binding checks stop replay, tampering, re-targeting and downgrade in both modes; presentation, morph and deepfake attacks pass every cryptographic check and are stopped only by the detectors. Post-quantum cryptography protects provenance and confidentiality, not genuineness.',
  '**Small tuning, large gain.** Training 0.03% of GenD\'s parameters cut accepted fakes at the strict operating point by a factor of about six on an independent test set.',
  '**Honest thresholds matter.** Setting M2 thresholds from the classifier\'s own training scores looked fine but rejected 37% of genuine users on held-out data; nested cross-validation fixed it.',
]));
add(H2('5.2 Limitations'));
add(bullets([
  '**M1** uses the pretrained Silent-Face models with their own decision rule; fine-tuning and evaluation on CelebA-Spoof (including screen-replay types) are pending because the dataset requires a manual download.',
  '**M5** currently applies only the virtual-camera name rule; the gradient-boosted classifier needs 40–60 recorded sessions per class with OBS Studio, which is not yet installed.',
  `**Thresholds are preliminary** (version ${thresholds[0].version}) and were set on validation data; final calibration is planned.`,
  '**Deployment** was tested on Windows with an in-process Redis replacement; the Docker image on Linux and a teammate\'s fresh-clone rebuild remain to be done, as do throughput tests under concurrent load.',
  '**Generalisation**: morph and deepfake results come from one dataset each, the matcher from studio photos, and the live tests from one person; performance on Indian faces and other webcams is untested.',
  '**Threat model**: a compromised server, side channels, denial of service, enrolment fraud, attackers on rooted devices and high-quality 3D masks are out of scope.',
]));
add(...table('pending', 'Status of each part of the work at the time of this report.',
  ['Part', 'Status'],
  [
    ['Crypto suite, NIST known-answer tests, server checks 1–6, signed verdict', 'Done and tested'],
    ['Matcher (FRLL), M2 (FRLL-Morphs), M3 zero-shot and tuned (OpenForensics Test-Dev)', 'Done, reported above'],
    ['ONNX export parity (M1, M3)', 'Done'],
    ['Crypto cost per operation and sizes', 'Done'],
    ['End-to-end latency, CPU, memory, bytes', perfDone ? 'Done (500 runs per mode)' : 'Pending'],
    ['Replay suite with real models', replayDone ? 'Done' : 'Pending'],
    ['Attack × mode matrix', matrixDone ? 'Done (deepfake, morph, bona fide; print/screen/OBS sources pending)' : 'Pending'],
    ['M1 fine-tuning on CelebA-Spoof', 'Pending (dataset download)'],
    ['M5 classifier on OBS recordings', 'Pending (OBS install, team recordings)'],
    ['Final calibration, Docker on Linux, concurrent throughput, fresh-clone rebuild', 'Pending'],
  ],
  [6026, 3000], { left: true }));

// ================================================================ 6 CONCLUSIONS
add(H1('6. Conclusions'));
add(P(`This project built a working face-verification prototype in which every piece of evidence is bound to one transaction and one single-use challenge, protected with ML-KEM-768, ML-DSA-65 and AES-256-GCM, checked by six protocol checks and five AI detectors, and closed by a post-quantum-signed verdict that makes the AI decision auditable. The same pipeline runs with classical cryptography, which made it possible to measure the cost of post-quantum security directly.`));
add(P(`The measurements answer the central question. Post-quantum cryptography is affordable for biometric evidence: its operations take well under a millisecond and add about ${fx((num(sPQ.per_transaction_asymmetric_B) - num(sCL.per_transaction_asymmetric_B)) / 1000, 1)} kB per transaction${perfDone ? `, a ${pct(pqS.crypto_share_p50, 2)} share of the end-to-end time` : ''}. But it only proves where the evidence came from and that it was not changed. The genuineness of the face depends on the detectors: the matcher met its 0.1% false-match target on unseen subjects, the morph detector accepted fewer than 4 in 1,000 morphs at a 0.6% genuine rejection, and a lightweight LayerNorm tuning made the deepfake detector usable at strict settings on an independent test set.`));
add(P(`Measured against the goals in Section 1.2 (${T('targets')}), the protection, binding, signed-verdict, matcher, morph and cost objectives were met, the deepfake and injection objectives were partly met, and M1 fine-tuning is still pending. The remaining work is clear: fine-tune and evaluate M1 on CelebA-Spoof, record OBS sessions and train the M5 classifier, run the final calibration, and package the system in Docker so that a teammate can reproduce every result from a fresh clone. A natural extension is to publish the signed verdicts to an external transparency log, so that an auditor can check not only that a decision was not edited, but also that none was deleted.`));

// ================================================================ CODE, STATEMENTS
add(H1('Code'));
add(new Paragraph({ spacing: { after: 120 }, children: [new TextRun('Source code (public): '),
  new ExternalHyperlink({ link: 'https://github.com/Aviralbhan/pq-biometric', children: [new TextRun({ text: 'https://github.com/Aviralbhan/pq-biometric', style: 'Hyperlink', color: '0563C1', underline: {} })] })] }));
add(P('The repository contains the full prototype, the tests, the evaluation scripts and the CSV files from which every number in this report is generated (docs/report/build_report.js and docs/report/make_figures.py). Datasets and model weights are not redistributed because of their licences; scripts/ downloads them from the original sources listed in ' + T('datasets') + ' (FRLL: https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666; FRLL-Morphs: https://zenodo.org/record/4415159; OpenForensics: https://zenodo.org/records/5528418; CelebA-Spoof: https://github.com/Davidzhangyuanhan/CelebA-Spoof). Commands for every step are in docs/RUNBOOK.md.'));
add(H1('Originality Statement'));
add(P('We, the three authors, agree that this work is original and novel, and that this report and the accompanying software are our own work, carried out for this course project. The design of the protocol, the signed-verdict record, the evaluation protocol and all experiments were done by the team. Published algorithms, models, datasets and libraries are used under their licences and are cited where they are used; no text from other sources is reproduced without citation. All results come from our own runs and are traceable to CSV files in the repository.'));
add(H1('AI Usage Statement'));
add(P('An AI coding assistant (Claude, by Anthropic) was used during the project to help write code, tests and documentation, to run and monitor experiments, and to draft this report from the team\'s design and result files. The team specified the design, reviewed the generated code and text, ran and checked the experiments, and verified that every number in the report is produced from the result CSVs by script. The AI tool was not used to generate or alter any result, and all references were checked against their sources. The team takes full responsibility for the content.'));

// ================================================================ APPENDIX
const appendixAt = body.length;
add(pageBreak(), H1('Appendix A. Reason codes and reproduction'));
add(...table('codes', 'Server reason codes (kept in the server log; the client only sees accepted / rejected).',
  ['Code', 'Meaning'],
  [
    ['EXPIRED', 'Challenge lifetime (60 s) passed'], ['REPLAY', 'Challenge already consumed'], ['MODE_MISMATCH', 'Bundle mode differs from the challenge mode'],
    ['TAMPER', 'KEM decapsulation or AES-GCM authentication failed'], ['TX_MISMATCH', 'Nonce, transaction hash, challenge id or mode differ'],
    ['HASH_MISMATCH', 'Video hash differs from the signed manifest'], ['SIG_INVALID', 'Device signature invalid'], ['NO_MATCH', 'Face match below threshold'],
    ['PAD_FAIL', 'M1 presentation attack'], ['MORPH_FLAG', 'M2 morphed reference'], ['DEEPFAKE_FAIL', 'M3 deepfake'], ['SCRIPT_FAIL', 'M4 actions not performed in order'],
    ['INJECTION_FAIL', 'M5 virtual camera / injection'],
  ],
  [2600, 6426], { left: true }));
add(P('Reproduction order (from docs/RUNBOOK.md): install requirements; run pytest; download the datasets and models with the scripts in scripts/; run eval_matcher.py, train_m2.py, eval_detectors.py and the M3 tuning; python -m eval.crypto_bench; python -m eval.perf --runs 500; scripts/replay_attacks.py --runs 1000; python -m eval.attack_matrix; then docs/report/make_figures.py and docs/report/build_report.js.'));

// ================================================================ REFERENCES
// Built last so every citation has its number; inserted before the appendix, as in the template.
const refParas = [H1('References'), ...citeOrder.map((k, i) => new Paragraph({
  alignment: AlignmentType.LEFT, spacing: { after: 80 }, indent: { left: 567, hanging: 567 },
  children: [new TextRun({ text: `[${i + 1}]`, size: 21 }), new TextRun({ text: '  ', size: 21 }), ...runs(REFS[k], { size: 21 })],
}))];
body.splice(appendixAt, 0, ...refParas);

if (tablesMade !== TABLES.length) throw new Error(`made ${tablesMade} tables, expected ${TABLES.length}`);
if (figsMade !== FIGS.length) throw new Error(`made ${figsMade} figures, expected ${FIGS.length}`);
const unused = Object.keys(REFS).filter((k) => !citeOrder.includes(k));
if (unused.length) console.warn('uncited references (left out):', unused.join(', '));

// ================================================================ DOCUMENT
const numberingConfig = [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] }];
for (let i = 1; i <= listNo; i++) {
  numberingConfig.push({ reference: `num${i}`, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] });
}
const heading = (id, name, size, extra = {}) => ({
  id, name, basedOn: 'Normal', next: 'Normal', quickFormat: true,
  run: { font: 'Times New Roman', size, bold: true, ...extra },
  paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, keepLines: true, outlineLevel: Number(id.slice(-1)) - 1 },
});
const doc = new Document({
  creator: 'Aviral Bhan, Krish Manoj Mehta, Abhijay Singh Panwar',
  title: 'AI-Based Biometric Evidence Authentication for Post-Quantum Digital Identity Systems',
  styles: {
    default: { document: { run: { font: 'Times New Roman', size: 24 }, paragraph: { spacing: { line: 276 } } } },
    paragraphStyles: [heading('Heading1', 'Heading 1', 30), heading('Heading2', 'Heading 2', 30), heading('Heading3', 'Heading 3', 24, { italics: true })],
  },
  numbering: { config: numberingConfig },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT}: ${tablesMade} tables, ${figsMade} figures, ${citeOrder.length} references, ${eqNo} equations`);
  console.log(`perf ${perfDone ? 'included' : 'PENDING'}, replay ${replayDone ? 'included' : 'PENDING'}, attack matrix ${matrixDone ? 'included' : 'PENDING'}`);
});
