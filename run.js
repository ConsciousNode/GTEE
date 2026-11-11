#!/usr/bin/env node
'use strict';

// ============================================================================
// GTEE Headless Generational Runner (run.js)
// Usage: node run.js --arm <arm> --gens <N> --seed <uint32> --out <dir>
// ============================================================================

const fs = require('fs');
const path = require('path');


// ── Deterministic Float32 Math & Seeded PRNG (Cross-Engine Bit-Identical) ──
// Zero calls to Math.exp, Math.tanh, Math.log, Math.sqrt, Math.sin, or Math.random.
const f32 = Math.fround;
const _convBuf = new ArrayBuffer(4);
const _f32View = new Float32Array(_convBuf);
const _i32View = new Int32Array(_convBuf);
const _u32View = new Uint32Array(_convBuf);

function floatToBits(x) {
  _f32View[0] = f32(x);
  return _i32View[0];
}

function bitsToFloat(b) {
  _i32View[0] = b | 0;
  return _f32View[0];
}

// Deterministic square root via IEEE-754 bit seed + 4x Float32 Newton-Raphson
function detSqrt(x) {
  x = f32(x);
  if (!(x > 0) || !Number.isFinite(x)) return f32(0);
  const ix = floatToBits(x);
  let y = bitsToFloat((ix >> 1) + 0x1fbd1df5);
  const half = f32(0.5);
  y = f32(half * f32(y + f32(x / y)));
  y = f32(half * f32(y + f32(x / y)));
  y = f32(half * f32(y + f32(x / y)));
  y = f32(half * f32(y + f32(x / y)));
  return y;
}

// Deterministic exponential exp(x) via range reduction x = k*ln(2) + r and degree-6 polynomial
function detExp(x) {
  x = f32(x);
  if (Number.isNaN(x)) return f32(0);
  if (x <= f32(-20.0)) return f32(0.0);
  if (x >= f32(20.0)) x = f32(20.0);
  const invLn2 = f32(1.4426950408889634);
  const ln2Hi = f32(0.693145751953125);
  const ln2Lo = f32(1.428606765330187e-6);
  const k = Math.round(f32(x * invLn2)) | 0;
  const kf = f32(k);
  const r = f32(f32(x - f32(kf * ln2Hi)) - f32(kf * ln2Lo));
  const c6 = f32(0.001388888888888889);
  const c5 = f32(0.008333333333333333);
  const c4 = f32(0.041666666666666664);
  const c3 = f32(0.16666666666666666);
  const c2 = f32(0.5);
  const c1 = f32(1.0);
  let p = f32(f32(c6 * r) + c5);
  p = f32(f32(p * r) + c4);
  p = f32(f32(p * r) + c3);
  p = f32(f32(p * r) + c2);
  p = f32(f32(p * r) + c1);
  p = f32(f32(p * r) + c1);
  const scaleBits = ((k + 127) & 0xff) << 23;
  return f32(p * bitsToFloat(scaleBits));
}

// Deterministic natural logarithm log(x) via IEEE-754 exponent/mantissa extraction + odd Padé/Atanh series
function detLog(x) {
  x = f32(x);
  if (!(x > 0) || !Number.isFinite(x)) x = f32(1e-12);
  if (x < f32(1e-12)) x = f32(1e-12);
  let ix = floatToBits(x);
  let exp = ((ix >>> 23) & 0xff) - 127;
  ix = (ix & 0x007fffff) | 0x3f800000;
  let m = bitsToFloat(ix);
  if (m > f32(1.41421356)) {
    m = f32(m * f32(0.5));
    exp = (exp + 1) | 0;
  }
  const u = f32(f32(m - f32(1.0)) / f32(m + f32(1.0)));
  const u2 = f32(u * u);
  const b7 = f32(0.2857142857142857);
  const b5 = f32(0.4);
  const b3 = f32(0.6666666666666666);
  const b1 = f32(2.0);
  let poly = f32(f32(b7 * u2) + b5);
  poly = f32(f32(poly * u2) + b3);
  poly = f32(f32(poly * u2) + b1);
  const lnm = f32(u * poly);
  const ln2 = f32(0.6931471805599453);
  return f32(f32(f32(exp) * ln2) + lnm);
}

// Deterministic hyperbolic tangent tanh(x)
function detTanh(x) {
  x = f32(x);
  if (Number.isNaN(x)) return f32(0);
  if (x >= f32(9.0)) return f32(1.0);
  if (x <= f32(-9.0)) return f32(-1.0);
  const e2x = detExp(f32(f32(2.0) * x));
  return f32(f32(e2x - f32(1.0)) / f32(e2x + f32(1.0)));
}

// Deterministic logistic sigmoid(x)
function detSigmoid(x) {
  x = f32(x);
  if (Number.isNaN(x)) return f32(0.5);
  if (x >= f32(14.0)) return f32(1.0);
  if (x <= f32(-14.0)) return f32(0.0);
  return f32(f32(1.0) / f32(f32(1.0) + detExp(f32(-x))));
}

// Deterministic sine sin(x) via range reduction to [-pi/2, pi/2] + 7th degree polynomial
function detSin(x) {
  x = f32(x);
  if (!Number.isFinite(x)) return f32(0);
  const twoPi = f32(6.283185307179586);
  const invTwoPi = f32(0.15915494309189535);
  const pi = f32(3.141592653589793);
  const halfPi = f32(1.5707963267948966);
  const k = Math.round(f32(x * invTwoPi));
  let r = f32(x - f32(f32(k) * twoPi));
  if (r > halfPi) r = f32(pi - r);
  else if (r < f32(-halfPi)) r = f32(f32(-pi) - r);
  const r2 = f32(r * r);
  const s7 = f32(-0.0001984126984126984);
  const s5 = f32(0.008333333333333333);
  const s3 = f32(-0.16666666666666666);
  let p = f32(f32(s7 * r2) + s5);
  p = f32(f32(p * r2) + s3);
  p = f32(f32(p * r2) + f32(1.0));
  return f32(r * p);
}

// ── Seeded PRNG: xoshiro128** initialized with SplitMix32 ──────────────────
class DeterministicPRNG {
  constructor(seed = 1) {
    this.s = new Uint32Array(4);
    this.reseed(seed >>> 0);
  }
  reseed(seed) {
    let z = (seed >>> 0) || 0x9e3779b9;
    for (let i = 0; i < 4; i++) {
      z = (z + 0x9e3779b9) >>> 0;
      let t = z ^ (z >>> 16);
      t = Math.imul(t, 0x21f0aaad) >>> 0;
      t = t ^ (t >>> 15);
      t = Math.imul(t, 0x735a2d97) >>> 0;
      this.s[i] = (t ^ (t >>> 15)) >>> 0;
    }
    if (this.s[0] === 0 && this.s[1] === 0 && this.s[2] === 0 && this.s[3] === 0) {
      this.s[0] = 1;
    }
  }
  getState() {
    return [this.s[0] >>> 0, this.s[1] >>> 0, this.s[2] >>> 0, this.s[3] >>> 0];
  }
  setState(st) {
    for (let i = 0; i < 4; i++) this.s[i] = st[i] >>> 0;
  }
  clone() {
    const c = new DeterministicPRNG(1);
    c.setState(this.getState());
    return c;
  }
  nextUint32() {
    const s0 = this.s[0], s1 = this.s[1], s2 = this.s[2], s3 = this.s[3];
    const s1_5 = Math.imul(s1, 5) >>> 0;
    const rot = ((s1_5 << 7) | (s1_5 >>> 25)) >>> 0;
    const result = Math.imul(rot, 9) >>> 0;
    const t = (s1 << 9) >>> 0;
    let n2 = (s2 ^ s0) >>> 0;
    let n3 = (s3 ^ s1) >>> 0;
    let n1 = (s1 ^ n2) >>> 0;
    let n0 = (s0 ^ n3) >>> 0;
    n2 = (n2 ^ t) >>> 0;
    n3 = ((n3 << 11) | (n3 >>> 21)) >>> 0;
    this.s[0] = n0; this.s[1] = n1; this.s[2] = n2; this.s[3] = n3;
    return result >>> 0;
  }
  nextFloat() {
    return f32((this.nextUint32() >>> 8) * (1.0 / 16777216.0));
  }
  nextSymNoise(bound) {
    const u = f32(this.nextFloat() * f32(2.0) - f32(1.0));
    return f32(u * f32(bound));
  }
  nextInt(maxExclusive) {
    if (maxExclusive <= 1) return 0;
    return (this.nextUint32() % (maxExclusive >>> 0)) | 0;
  }
}

// ── Pure-JS Deterministic SHA-256 (No WebCrypto/Node crypto required) ──────
const _SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]);

function utf8ToBytes(str) {
  const out = [];
  for (let i = 0; i < str.length; i++) {
    let c = str.charCodeAt(i);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) {
      out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
    } else if (c < 0xd800 || c >= 0xe000) {
      out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
    } else {
      i++;
      c = 0x10000 + (((c & 0x3ff) << 10) | (str.charCodeAt(i) & 0x3ff));
      out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 0x3f), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
    }
  }
  return new Uint8Array(out);
}

function sha256Bytes(dataBytes) {
  const bytes = typeof dataBytes === 'string' ? utf8ToBytes(dataBytes) : dataBytes;
  const len = bytes.length;
  const bitLenHi = Math.floor((len * 8) / 0x100000000) >>> 0;
  const bitLenLo = (len * 8) >>> 0;
  const padLen = ((len + 9 + 63) & ~63);
  const buf = new Uint8Array(padLen);
  buf.set(bytes, 0);
  buf[len] = 0x80;
  const dv = new DataView(buf.buffer);
  dv.setUint32(padLen - 8, bitLenHi, false);
  dv.setUint32(padLen - 4, bitLenLo, false);

  let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
  let h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
  const w = new Uint32Array(64);

  for (let offset = 0; offset < padLen; offset += 64) {
    for (let i = 0; i < 16; i++) w[i] = dv.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const v0 = w[i - 15], v1 = w[i - 2];
      const s0 = (((v0 >>> 7) | (v0 << 25)) ^ ((v0 >>> 18) | (v0 << 14)) ^ (v0 >>> 3)) >>> 0;
      const s1 = (((v1 >>> 17) | (v1 << 15)) ^ ((v1 >>> 19) | (v1 << 13)) ^ (v1 >>> 10)) >>> 0;
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
    for (let i = 0; i < 64; i++) {
      const S1 = (((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7))) >>> 0;
      const ch = ((e & f) ^ ((~e) & g)) >>> 0;
      const temp1 = (h + S1 + ch + _SHA256_K[i] + w[i]) >>> 0;
      const S0 = (((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10))) >>> 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const temp2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + temp1) >>> 0;
      d = c; c = b; b = a; a = (temp1 + temp2) >>> 0;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0; h5 = (h5 + f) >>> 0; h6 = (h6 + g) >>> 0; h7 = (h7 + h) >>> 0;
  }
  const hex = n => (n >>> 0).toString(16).padStart(8, '0');
  return hex(h0) + hex(h1) + hex(h2) + hex(h3) + hex(h4) + hex(h5) + hex(h6) + hex(h7);
}

// Deterministic Canonical JSON Serializer (sorts object keys recursively)
function canonicalJSON(val) {
  if (val === null || val === undefined) return 'null';
  if (typeof val === 'boolean') return val ? 'true' : 'false';
  if (typeof val === 'number') {
    if (!Number.isFinite(val)) return 'null';
    if (Object.is(val, -0)) return '0';
    return JSON.stringify(val);
  }
  if (typeof val === 'string') return JSON.stringify(val);
  if (Array.isArray(val)) {
    return '[' + val.map(v => canonicalJSON(v)).join(',') + ']';
  }
  if (ArrayBuffer.isView(val)) {
    return canonicalJSON(Array.from(val));
  }
  if (typeof val === 'object') {
    const keys = Object.keys(val).filter(k => val[k] !== undefined).sort();
    return '{' + keys.map(k => JSON.stringify(k) + ':' + canonicalJSON(val[k])).join(',') + '}';
  }
  return 'null';
}

// Base64 helpers (pure JS, no window.btoa/Buffer dependency required)
const _B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const _B64_LOOKUP = new Uint8Array(128);
for (let i = 0; i < _B64_CHARS.length; i++) _B64_LOOKUP[_B64_CHARS.charCodeAt(i)] = i;

function bytesToBase64(u8) {
  let out = '';
  const len = u8.length;
  for (let i = 0; i < len; i += 3) {
    const b0 = u8[i];
    const b1 = i + 1 < len ? u8[i + 1] : 0;
    const b2 = i + 2 < len ? u8[i + 2] : 0;
    const trip = (b0 << 16) | (b1 << 8) | b2;
    out += _B64_CHARS[(trip >> 18) & 63];
    out += _B64_CHARS[(trip >> 12) & 63];
    out += (i + 1 < len) ? _B64_CHARS[(trip >> 6) & 63] : '=';
    out += (i + 2 < len) ? _B64_CHARS[trip & 63] : '=';
  }
  return out;
}

function base64ToBytes(b64) {
  const clean = b64.replace(/[^A-Za-z0-9+/=]/g, '');
  const len = clean.length;
  let pad = 0;
  if (clean.endsWith('==')) pad = 2;
  else if (clean.endsWith('=')) pad = 1;
  const outLen = ((len * 3) >> 2) - pad;
  const out = new Uint8Array(outLen);
  let p = 0;
  for (let i = 0; i < len; i += 4) {
    const c0 = _B64_LOOKUP[clean.charCodeAt(i)];
    const c1 = _B64_LOOKUP[clean.charCodeAt(i + 1)];
    const c2 = _B64_LOOKUP[clean.charCodeAt(i + 2)];
    const c3 = _B64_LOOKUP[clean.charCodeAt(i + 3)];
    const trip = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
    if (p < outLen) out[p++] = (trip >> 16) & 0xff;
    if (p < outLen) out[p++] = (trip >> 8) & 0xff;
    if (p < outLen) out[p++] = trip & 0xff;
  }
  return out;
}

// ----------------------------------------------------------------------------
// 2. PROVENANCE: TERNARY QUANTIZATION & RWKV-7 ENGINE LIFTED FROM pips-room.html
// ----------------------------------------------------------------------------

// [PROVENANCE: Lifted from pips-room.html lines 3915-3983 (class TMAC)]
// Modifications for GTEE:
// 1. Replaced native float arithmetic with Math.fround (f32) for cross-engine bit-identity.
// 2. Added configurable thresholdFactor (default 0.5, matching Math.round) in quantizeWeights.
class TMAC {
  static UNPACK_TABLE = new Int8Array(81 * 4);

  static initTable() {
    for (let i = 0; i < 81; i++) {
      let rem = i;
      TMAC.UNPACK_TABLE[i * 4] = (rem % 3) - 1; rem = Math.floor(rem / 3);
      TMAC.UNPACK_TABLE[i * 4 + 1] = (rem % 3) - 1; rem = Math.floor(rem / 3);
      TMAC.UNPACK_TABLE[i * 4 + 2] = (rem % 3) - 1; rem = Math.floor(rem / 3);
      TMAC.UNPACK_TABLE[i * 4 + 3] = (rem % 3) - 1;
    }
  }

  static packTernary(t0, t1, t2, t3) {
    return (t0 + 1) + 3 * (t1 + 1) + 9 * (t2 + 1) + 27 * (t3 + 1);
  }

  static quantizeStep(val, invGamma, thresholdFactor = 0.5) {
    const scaled = f32(f32(val) * f32(invGamma));
    const thr = f32(thresholdFactor);
    if (scaled >= thr) return 1;
    if (scaled <= f32(-thr)) return -1;
    return 0;
  }

  static quantizeWeights(masterWeights, packedOut, thresholdFactor = 0.5) {
    const n = masterWeights.length;
    const blocks = Math.ceil(n / 4);

    let sumAbs = f32(0);
    for (let i = 0; i < n; i++) sumAbs = f32(sumAbs + Math.abs(masterWeights[i]));
    const gamma = f32(sumAbs / f32(n)) || f32(1e-8);
    const invGamma = f32(f32(1.0) / gamma);

    for (let b = 0; b < blocks; b++) {
      const base = b * 4;
      const t0 = base < n ? TMAC.quantizeStep(masterWeights[base], invGamma, thresholdFactor) : 0;
      const t1 = base + 1 < n ? TMAC.quantizeStep(masterWeights[base + 1], invGamma, thresholdFactor) : 0;
      const t2 = base + 2 < n ? TMAC.quantizeStep(masterWeights[base + 2], invGamma, thresholdFactor) : 0;
      const t3 = base + 3 < n ? TMAC.quantizeStep(masterWeights[base + 3], invGamma, thresholdFactor) : 0;
      packedOut[b] = TMAC.packTernary(t0, t1, t2, t3);
    }
    return gamma;
  }

  static unpackToFloat32(packedOut, length) {
    const out = new Float32Array(length);
    const blocks = Math.ceil(length / 4);
    for (let b = 0; b < blocks; b++) {
      const code = packedOut[b];
      const uBase = code * 4;
      const base = b * 4;
      for (let k = 0; k < 4 && base + k < length; k++) {
        out[base + k] = TMAC.UNPACK_TABLE[uBase + k];
      }
    }
    return out;
  }

  static execute(packedWeights, input, outDim, inDim, gamma, outputArr, scratchLut) {
    const blocksPerRow = Math.ceil(inDim / 4);

    for (let b = 0; b < blocksPerRow; b++) {
      const base = b * 4;
      const v0 = base < inDim ? f32(input[base]) : f32(0);
      const v1 = base + 1 < inDim ? f32(input[base + 1]) : f32(0);
      const v2 = base + 2 < inDim ? f32(input[base + 2]) : f32(0);
      const v3 = base + 3 < inDim ? f32(input[base + 3]) : f32(0);

      const lutOffset = b * 81;
      for (let i = 0; i < 81; i++) {
        const uBase = i * 4;
        const p0 = f32(TMAC.UNPACK_TABLE[uBase] * v0);
        const p1 = f32(TMAC.UNPACK_TABLE[uBase + 1] * v1);
        const p2 = f32(TMAC.UNPACK_TABLE[uBase + 2] * v2);
        const p3 = f32(TMAC.UNPACK_TABLE[uBase + 3] * v3);
        scratchLut[lutOffset + i] = f32(f32(p0 + p1) + f32(p2 + p3));
      }
    }

    for (let o = 0; o < outDim; o++) {
      let acc = f32(0);
      const rowBase = o * blocksPerRow;
      for (let b = 0; b < blocksPerRow; b++) {
        acc = f32(acc + scratchLut[b * 81 + packedWeights[rowBase + b]]);
      }
      outputArr[o] = f32(acc * f32(gamma));
    }
  }
}
TMAC.initTable();

// [PROVENANCE: Lifted from pips-room.html lines 3985-4041 (class BitLinear)]
// Modifications for GTEE:
// 1. Uses seeded DeterministicPRNG and detSqrt instead of Math.random / Math.sqrt.
class BitLinear {
  constructor(inDim, outDim, rng = new DeterministicPRNG(1)) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.weight = new Float32Array(outDim * inDim);
    this.grad = new Float32Array(outDim * inDim);
    this.momentum = new Float32Array(outDim * inDim);

    this.packed = new Uint8Array(Math.ceil((outDim * inDim) / 4));
    this.gamma = f32(1.0);
    this.dirty = true;
    this.thresholdFactor = f32(0.5);

    this.lutScratch = new Float32Array(Math.ceil(inDim / 4) * 81);

    const scale = detSqrt(f32(3.0 / inDim));
    for (let i = 0; i < this.weight.length; i++) {
      this.weight[i] = f32(f32(rng.nextFloat() * f32(2.0) - f32(1.0)) * scale);
    }
  }

  pack() {
    if (this.dirty) {
      this.gamma = TMAC.quantizeWeights(this.weight, this.packed, this.thresholdFactor);
      this.dirty = false;
    }
  }

  forward(x) {
    this.pack();
    const out = new Float32Array(this.outDim);
    TMAC.execute(this.packed, x, this.outDim, this.inDim, this.gamma, out, this.lutScratch);
    return out;
  }

  backward(gradOut, x) {
    for (let o = 0; o < this.outDim; o++) {
      const rowOffset = o * this.inDim;
      const go = f32(gradOut[o]);
      for (let i = 0; i < this.inDim; i++) {
        this.grad[rowOffset + i] = f32(this.grad[rowOffset + i] + f32(go * f32(x[i])));
      }
    }

    const gradIn = new Float32Array(this.inDim);
    for (let i = 0; i < this.inDim; i++) {
      let acc = f32(0);
      for (let o = 0; o < this.outDim; o++) {
        acc = f32(acc + f32(this.weight[o * this.inDim + i] * f32(gradOut[o])));
      }
      gradIn[i] = acc;
    }
    return gradIn;
  }

  zeroGrad() { this.grad.fill(0); }
  markDirty() { this.dirty = true; }
}

// [PROVENANCE: Lifted from pips-room.html lines 1342-1648 (class RWKV7 "Goose" x070)]
// Canonical source in pips-room.html: HTMLNLM/engine/rwkv7-engine.js (BlinkDL rwkv7 math + exact BPTT).
// Modifications for GTEE:
// 1. Replaced Math.exp, Math.tanh, Math.sqrt, Math.random with deterministic Float32 equivalents
//    (detExp, detTanh, detSqrt, detSigmoid, and seeded DeterministicPRNG).
// 2. Integrated TMAC ternary quantization (lifted from pips-room.html:3915-3983) into _lin()
//    controlled by model.ternaryMix and model.ternaryThreshold via Straight-Through Estimator (STE).
const SIG = detSigmoid;
const W_SCALE = f32(0.606531); // exp(-0.5), official constant (pips-room.html:1343)

const DEFAULT_MODEL_CONFIG = Object.freeze({
  vocabSize: 24,
  dim: 12,
  headSize: 6,
  numLayers: 2,
  dimFFN: 24,
  dDecay: 6,
  dAAA: 6,
  dMV: 6,
  dGate: 6
});

class RWKV7 {
  constructor(cfg = DEFAULT_MODEL_CONFIG, opts = {}) {
    const F = this.Float = Float32Array;
    const rng = opts.rng || new DeterministicPRNG(opts.seed || 1);
    const C = this.C = cfg.dim;
    const N = this.N = cfg.headSize || 6;
    const H = this.H = C / N;
    const L = this.L = cfg.numLayers;
    const V = this.V = cfg.vocabSize;
    if (!Number.isInteger(H)) throw new Error(`dim ${C} is not a multiple of headSize ${N}`);
    const FF = this.FF = cfg.dimFFN || 2 * C;
    const DW = cfg.dDecay || 6, DA = cfg.dAAA || 6, DV = cfg.dMV || 6, DG = cfg.dGate || 6;
    this.cfg = { vocabSize: V, dim: C, numLayers: L, headSize: N, dimFFN: FF, dDecay: DW, dAAA: DA, dMV: DV, dGate: DG };
    this.ternaryMix = f32(opts.ternaryMix !== undefined ? opts.ternaryMix : 0.2);
    this.ternaryThreshold = f32(opts.ternaryThreshold !== undefined ? opts.ternaryThreshold : 0.5);

    this.p = {};
    this.shape = {};
    this.g = {};
    this.m = {}; // Momentum buffer per parameter tensor
    this.v = {}; // Variance buffer per parameter tensor
    this.paramOrder = [];

    const def = (name, shape, init) => {
      const n = shape.reduce((a, b) => a * b, 1);
      const arr = new F(n);
      if (init) for (let i = 0; i < n; i++) arr[i] = f32(init(i));
      this.p[name] = arr;
      this.shape[name] = shape;
      this.g[name] = new F(n);
      this.m[name] = new F(n);
      this.v[name] = new F(n);
      this.paramOrder.push(name);
    };

    const rnd = s => () => f32(f32(rng.nextFloat() * f32(2.0) - f32(1.0)) * f32(s));
    const one = () => f32(1.0);
    const half = () => f32(0.5);

    def('emb.weight', [V, C], rnd(0.08));
    def('blocks.0.ln0.weight', [C], one);
    def('blocks.0.ln0.bias', [C]);
    for (let l = 0; l < L; l++) {
      const b = `blocks.${l}.`, a = b + 'att.', f = b + 'ffn.';
      def(b + 'ln1.weight', [C], one); def(b + 'ln1.bias', [C]);
      def(b + 'ln2.weight', [C], one); def(b + 'ln2.bias', [C]);
      for (const x of ['x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g']) def(a + x, [1, 1, C], half);
      def(a + 'w0', [1, 1, C], () => f32(-1.0));
      def(a + 'w1', [C, DW], rnd(0.08));
      def(a + 'w2', [DW, C], rnd(0.1));
      def(a + 'a0', [1, 1, C]);
      def(a + 'a1', [C, DA], rnd(0.08));
      def(a + 'a2', [DA, C], rnd(0.1));
      if (l > 0) {
        def(a + 'v0', [1, 1, C], one);
        def(a + 'v1', [C, DV], rnd(0.08));
        def(a + 'v2', [DV, C], rnd(0.1));
      }
      def(a + 'g1', [C, DG], rnd(0.08));
      def(a + 'g2', [DG, C], rnd(0.1));
      def(a + 'k_k', [1, 1, C], () => f32(0.85));
      def(a + 'k_a', [1, 1, C], one);
      def(a + 'r_k', [H, N], rnd(0.05));
      for (const m of ['receptance', 'key', 'value', 'output']) {
        def(a + m + '.weight', [C, C], rnd(detSqrt(f32(3.0 / C))));
      }
      def(a + 'ln_x.weight', [C], one);
      def(a + 'ln_x.bias', [C]);
      def(f + 'x_k', [1, 1, C], half);
      def(f + 'key.weight', [FF, C], rnd(detSqrt(f32(3.0 / C))));
      def(f + 'value.weight', [C, FF], rnd(detSqrt(f32(3.0 / FF))));
    }
    def('ln_out.weight', [C], one);
    def('ln_out.bias', [C]);
    def('head.weight', [V, C], rnd(detSqrt(f32(3.0 / C))));
  }

  zeroGrad() {
    for (const name of this.paramOrder) this.g[name].fill(0);
  }

  getEmptyState() {
    const F = this.Float, s = [];
    for (let l = 0; l < this.L; l++) {
      s.push({ xa: new F(this.C), S: new F(this.H * this.N * this.N), xf: new F(this.C) });
    }
    return s;
  }

  exportWeightsFlat() {
    let total = 0;
    for (const name of this.paramOrder) total += this.p[name].length;
    const flat = new Float32Array(total);
    let offset = 0;
    for (const name of this.paramOrder) {
      flat.set(this.p[name], offset);
      offset += this.p[name].length;
    }
    return flat;
  }

  importWeightsFlat(flat) {
    let offset = 0;
    for (const name of this.paramOrder) {
      const len = this.p[name].length;
      for (let i = 0; i < len; i++) this.p[name][i] = f32(flat[offset + i]);
      offset += len;
    }
  }

  exportBuffersFlat() {
    let total = 0;
    for (const name of this.paramOrder) total += this.p[name].length;
    const mFlat = new Float32Array(total);
    const vFlat = new Float32Array(total);
    let offset = 0;
    for (const name of this.paramOrder) {
      const len = this.p[name].length;
      mFlat.set(this.m[name], offset);
      vFlat.set(this.v[name], offset);
      offset += len;
    }
    return { mFlat, vFlat };
  }

  importBuffersFlat(bufs) {
    let offset = 0;
    for (const name of this.paramOrder) {
      const len = this.p[name].length;
      for (let i = 0; i < len; i++) {
        this.m[name][i] = f32(bufs.mFlat[offset + i]);
        this.v[name][i] = f32(bufs.vFlat[offset + i]);
      }
      offset += len;
    }
  }

  weightsHash() {
    const flat = this.exportWeightsFlat();
    const u8 = new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength);
    return sha256Bytes(u8);
  }

  weightsToBase64() {
    const flat = this.exportWeightsFlat();
    const u8 = new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength);
    return bytesToBase64(u8);
  }

  loadWeightsFromBase64(b64) {
    const u8 = base64ToBytes(b64);
    if (u8.byteLength % 4 !== 0) throw new Error('Corrupt weight buffer alignment');
    const flat = new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
    let expected = 0;
    for (const name of this.paramOrder) expected += this.p[name].length;
    if (flat.length !== expected) throw new Error(`Weight length mismatch: expected ${expected}, got ${flat.length}`);
    this.importWeightsFlat(flat);
  }

  _lin(name, x) {
    const W = this.p[name], [outD, inD] = this.shape[name], y = new this.Float(outD);
    for (let o = 0; o < outD; o++) {
      let s = f32(0);
      const r = o * inD;
      for (let i = 0; i < inD; i++) s = f32(s + f32(W[r + i] * x[i]));
      y[o] = s;
    }
    const mix = f32(this.ternaryMix);
    if (mix > f32(0)) {
      const packed = new Uint8Array(Math.ceil((outD * inD) / 4));
      const lutScratch = new Float32Array(Math.ceil(inD / 4) * 81);
      const yTern = new Float32Array(outD);
      const gamma = TMAC.quantizeWeights(W, packed, this.ternaryThreshold);
      TMAC.execute(packed, x, outD, inD, gamma, yTern, lutScratch);
      const oneMinus = f32(f32(1.0) - mix);
      for (let o = 0; o < outD; o++) {
        y[o] = f32(f32(oneMinus * y[o]) + f32(mix * yTern[o]));
      }
    }
    return y;
  }

  _linBack(name, x, gy) {
    const W = this.p[name], G = this.g[name], [outD, inD] = this.shape[name], gx = new this.Float(inD);
    for (let o = 0; o < outD; o++) {
      const go = f32(gy[o]);
      if (go === 0) continue;
      const r = o * inD;
      for (let i = 0; i < inD; i++) {
        gx[i] = f32(gx[i] + f32(W[r + i] * go));
        G[r + i] = f32(G[r + i] + f32(go * x[i]));
      }
    }
    return gx;
  }

  _xM(name, x) {
    const M = this.p[name], [Cc, D] = this.shape[name], h = new this.Float(D);
    for (let c = 0; c < Cc; c++) {
      const xc = f32(x[c]);
      if (xc === 0) continue;
      const r = c * D;
      for (let d = 0; d < D; d++) h[d] = f32(h[d] + f32(xc * M[r + d]));
    }
    return h;
  }

  _xMBack(name, x, gh, gx) {
    const M = this.p[name], G = this.g[name], [Cc, D] = this.shape[name];
    for (let c = 0; c < Cc; c++) {
      const r = c * D;
      let s = f32(0);
      for (let d = 0; d < D; d++) {
        s = f32(s + f32(M[r + d] * gh[d]));
        G[r + d] = f32(G[r + d] + f32(x[c] * gh[d]));
      }
      gx[c] = f32(gx[c] + s);
    }
  }

  _ln(x, wn, bn, eps = 1e-5) {
    const n = x.length, w = this.p[wn], b = this.p[bn];
    let m = f32(0);
    for (let i = 0; i < n; i++) m = f32(m + x[i]);
    m = f32(m / f32(n));
    let v = f32(0);
    for (let i = 0; i < n; i++) {
      const d = f32(x[i] - m);
      v = f32(v + f32(d * d));
    }
    v = f32(v / f32(n));
    const rstd = f32(f32(1.0) / detSqrt(f32(v + f32(eps))));
    const xh = new this.Float(n), y = new this.Float(n);
    for (let i = 0; i < n; i++) {
      xh[i] = f32(f32(x[i] - m) * rstd);
      y[i] = f32(f32(xh[i] * w[i]) + b[i]);
    }
    return { y, c: { xh, rstd, wn, bn } };
  }

  _lnBack(gy, c) {
    const n = gy.length, w = this.p[c.wn], gw = this.g[c.wn], gb = this.g[c.bn], gxh = new this.Float(n);
    let m1 = f32(0), m2 = f32(0);
    for (let i = 0; i < n; i++) {
      gw[i] = f32(gw[i] + f32(gy[i] * c.xh[i]));
      gb[i] = f32(gb[i] + gy[i]);
      gxh[i] = f32(gy[i] * w[i]);
      m1 = f32(m1 + gxh[i]);
      m2 = f32(m2 + f32(gxh[i] * c.xh[i]));
    }
    m1 = f32(m1 / f32(n));
    m2 = f32(m2 / f32(n));
    const gx = new this.Float(n);
    for (let i = 0; i < n; i++) {
      gx[i] = f32(c.rstd * f32(f32(gxh[i] - m1) - f32(c.xh[i] * m2)));
    }
    return gx;
  }

  _tmix(l, x, xp, vFirst, S0) {
    const F = this.Float, C = this.C, H = this.H, N = this.N, a = `blocks.${l}.att.`, P = this.p, c = { l, x, xp };
    const mix = n => {
      const m = P[a + n], o = new F(C);
      for (let i = 0; i < C; i++) o[i] = f32(x[i] + f32(f32(xp[i] - x[i]) * m[i]));
      return o;
    };
    const xr = mix('x_r'), xw = mix('x_w'), xk = mix('x_k'), xv = mix('x_v'), xa = mix('x_a'), xg = mix('x_g');
    const r = this._lin(a + 'receptance.weight', xr);
    const k0 = this._lin(a + 'key.weight', xk);
    const vb = this._lin(a + 'value.weight', xv);
    const hw = this._xM(a + 'w1', xw);
    const tw = new F(hw.length);
    for (let i = 0; i < hw.length; i++) tw[i] = detTanh(hw[i]);
    const wl = this._xM(a + 'w2', tw);
    const ha = this._xM(a + 'a1', xa), aa = this._xM(a + 'a2', ha), av = new F(C);
    for (let i = 0; i < C; i++) av[i] = SIG(f32(P[a + 'a0'][i] + aa[i]));
    const hg = this._xM(a + 'g1', xg), sg = new F(hg.length);
    for (let i = 0; i < hg.length; i++) sg[i] = SIG(hg[i]);
    const g = this._xM(a + 'g2', sg);

    const kkr = new F(C), kk = new F(C), knrm = new F(H);
    for (let i = 0; i < C; i++) kkr[i] = f32(k0[i] * P[a + 'k_k'][i]);
    for (let h = 0; h < H; h++) {
      let s = f32(0);
      for (let j = 0; j < N; j++) {
        const v_ = kkr[h * N + j];
        s = f32(s + f32(v_ * v_));
      }
      const nrm = detSqrt(s);
      knrm[h] = nrm > f32(1e-6) ? nrm : f32(1e-6);
      for (let j = 0; j < N; j++) kk[h * N + j] = f32(kkr[h * N + j] / knrm[h]);
    }
    const k = new F(C);
    for (let i = 0; i < C; i++) k[i] = f32(k0[i] * f32(f32(1.0) + f32(f32(av[i] - f32(1.0)) * P[a + 'k_a'][i])));
    let v, gate = null, hv = null;
    if (l === 0) {
      v = vb; vFirst = vb;
    } else {
      hv = this._xM(a + 'v1', xv);
      const vv = this._xM(a + 'v2', hv);
      gate = new F(C); v = new F(C);
      for (let i = 0; i < C; i++) {
        gate[i] = SIG(f32(P[a + 'v0'][i] + vv[i]));
        v[i] = f32(vb[i] + f32(f32(vFirst[i] - vb[i]) * gate[i]));
      }
    }
    const sw = new F(C), w = new F(C);
    for (let i = 0; i < C; i++) {
      sw[i] = SIG(f32(P[a + 'w0'][i] + wl[i]));
      w[i] = detExp(f32(f32(-W_SCALE) * sw[i]));
    }
    const S = new F(H * N * N), q = new F(C), o = new F(C);
    for (let h = 0; h < H; h++) {
      const base = h * N * N, hb = h * N;
      for (let i = 0; i < N; i++) {
        let s = f32(0);
        for (let m = 0; m < N; m++) s = f32(s + f32(S0[base + i * N + m] * kk[hb + m]));
        q[hb + i] = s;
      }
      for (let i = 0; i < N; i++) {
        let s = f32(0);
        for (let j = 0; j < N; j++) {
          const term1 = f32(S0[base + i * N + j] * w[hb + j]);
          const term2 = f32(f32(q[hb + i] * kk[hb + j]) * av[hb + j]);
          const term3 = f32(v[hb + i] * k[hb + j]);
          const val = f32(f32(term1 - term2) + term3);
          S[base + i * N + j] = val;
          s = f32(s + f32(val * r[hb + j]));
        }
        o[hb + i] = s;
      }
    }
    const lw = P[a + 'ln_x.weight'], lb = P[a + 'ln_x.bias'], on = new F(C), grstd = new F(H), y = new F(C), bon = new F(H);
    for (let h = 0; h < H; h++) {
      const hb = h * N;
      let m = f32(0);
      for (let j = 0; j < N; j++) m = f32(m + o[hb + j]);
      m = f32(m / f32(N));
      let vr = f32(0);
      for (let j = 0; j < N; j++) {
        const d = f32(o[hb + j] - m);
        vr = f32(vr + f32(d * d));
      }
      vr = f32(vr / f32(N));
      grstd[h] = f32(f32(1.0) / detSqrt(f32(vr + f32(64e-5))));
      for (let j = 0; j < N; j++) {
        on[hb + j] = f32(f32(o[hb + j] - m) * grstd[h]);
        y[hb + j] = f32(f32(on[hb + j] * lw[hb + j]) + lb[hb + j]);
      }
      let s = f32(0);
      for (let j = 0; j < N; j++) s = f32(s + f32(f32(r[hb + j] * k[hb + j]) * P[a + 'r_k'][hb + j]));
      bon[h] = s;
      for (let j = 0; j < N; j++) y[hb + j] = f32(y[hb + j] + f32(s * v[hb + j]));
    }
    const z = new F(C);
    for (let i = 0; i < C; i++) z[i] = f32(y[i] * g[i]);
    const out = this._lin(a + 'output.weight', z);
    Object.assign(c, { xr, xw, xk, xv, xa, xg, r, k0, vb, hw, tw, ha, av, hg, sg, g, kkr, kk, knrm, k, v, gate, hv, vFirst, sw, w, S0, S, q, on, grstd, bon, y, z });
    return { out, S, vFirst, c };
  }

  _tmixBack(c, gout, gS, gvFirst) {
    const F = this.Float, C = this.C, H = this.H, N = this.N, a = `blocks.${c.l}.att.`, P = this.p, G = this.g;
    const gz = this._linBack(a + 'output.weight', c.z, gout);
    const gy = new F(C), gg = new F(C);
    for (let i = 0; i < C; i++) {
      gy[i] = f32(gz[i] * c.g[i]);
      gg[i] = f32(gz[i] * c.y[i]);
    }
    const gr = new F(C), gk = new F(C), gv = new F(C), go = new F(C);
    const lw = P[a + 'ln_x.weight'];
    for (let h = 0; h < H; h++) {
      const hb = h * N;
      let gbon = f32(0);
      for (let j = 0; j < N; j++) {
        gbon = f32(gbon + f32(gy[hb + j] * c.v[hb + j]));
        gv[hb + j] = f32(gv[hb + j] + f32(gy[hb + j] * c.bon[h]));
      }
      for (let j = 0; j < N; j++) {
        const rk = P[a + 'r_k'][hb + j];
        gr[hb + j] = f32(gr[hb + j] + f32(f32(gbon * c.k[hb + j]) * rk));
        gk[hb + j] = f32(gk[hb + j] + f32(f32(gbon * c.r[hb + j]) * rk));
        G[a + 'r_k'][hb + j] = f32(G[a + 'r_k'][hb + j] + f32(f32(gbon * c.r[hb + j]) * c.k[hb + j]));
      }
      let m1 = f32(0), m2 = f32(0);
      const gon = new F(N);
      for (let j = 0; j < N; j++) {
        G[a + 'ln_x.weight'][hb + j] = f32(G[a + 'ln_x.weight'][hb + j] + f32(gy[hb + j] * c.on[hb + j]));
        G[a + 'ln_x.bias'][hb + j] = f32(G[a + 'ln_x.bias'][hb + j] + gy[hb + j]);
        gon[j] = f32(gy[hb + j] * lw[hb + j]);
        m1 = f32(m1 + gon[j]);
        m2 = f32(m2 + f32(gon[j] * c.on[hb + j]));
      }
      m1 = f32(m1 / f32(N));
      m2 = f32(m2 / f32(N));
      for (let j = 0; j < N; j++) {
        go[hb + j] = f32(c.grstd[h] * f32(f32(gon[j] - m1) - f32(c.on[hb + j] * m2)));
      }
    }
    const gS0 = new F(H * N * N), gw = new F(C), gkk = new F(C), ga = new F(C);
    for (let h = 0; h < H; h++) {
      const base = h * N * N, hb = h * N, gq = new F(N);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const idx = base + i * N + j;
          const gSn = f32(f32(go[hb + i] * c.r[hb + j]) + (gS ? gS[idx] : f32(0)));
          gr[hb + j] = f32(gr[hb + j] + f32(c.S[idx] * go[hb + i]));
          if (gSn === 0) continue;
          gS0[idx] = f32(gS0[idx] + f32(gSn * c.w[hb + j]));
          gw[hb + j] = f32(gw[hb + j] + f32(gSn * c.S0[idx]));
          const b = f32(c.kk[hb + j] * c.av[hb + j]);
          gq[i] = f32(gq[i] - f32(gSn * b));
          const gb = f32(f32(-gSn) * c.q[hb + i]);
          gkk[hb + j] = f32(gkk[hb + j] + f32(gb * c.av[hb + j]));
          ga[hb + j] = f32(ga[hb + j] + f32(gb * c.kk[hb + j]));
          gv[hb + i] = f32(gv[hb + i] + f32(gSn * c.k[hb + j]));
          gk[hb + j] = f32(gk[hb + j] + f32(gSn * c.v[hb + i]));
        }
      }
      for (let i = 0; i < N; i++) {
        const g_ = gq[i];
        if (g_ === 0) continue;
        for (let m = 0; m < N; m++) {
          gS0[base + i * N + m] = f32(gS0[base + i * N + m] + f32(g_ * c.kk[hb + m]));
          gkk[hb + m] = f32(gkk[hb + m] + f32(g_ * c.S0[base + i * N + m]));
        }
      }
    }
    const gu = new F(C);
    for (let i = 0; i < C; i++) {
      gu[i] = f32(f32(f32(gw[i] * c.w[i]) * f32(-W_SCALE)) * f32(c.sw[i] * f32(f32(1.0) - c.sw[i])));
      G[a + 'w0'][i] = f32(G[a + 'w0'][i] + gu[i]);
    }
    const gxw = new F(C), gtw = new F(c.tw.length);
    this._xMBack(a + 'w2', c.tw, gu, gtw);
    const ghw = new F(gtw.length);
    for (let d = 0; d < gtw.length; d++) ghw[d] = f32(gtw[d] * f32(f32(1.0) - f32(c.tw[d] * c.tw[d])));
    this._xMBack(a + 'w1', c.xw, ghw, gxw);

    const gvb = new F(C), gxv = new F(C);
    if (c.l === 0) {
      for (let i = 0; i < C; i++) gvb[i] = f32(gv[i] + gvFirst[i]);
    } else {
      const gpre = new F(C);
      for (let i = 0; i < C; i++) {
        gvb[i] = f32(gv[i] * f32(f32(1.0) - c.gate[i]));
        gvFirst[i] = f32(gvFirst[i] + f32(gv[i] * c.gate[i]));
        gpre[i] = f32(f32(gv[i] * f32(c.vFirst[i] - c.vb[i])) * f32(c.gate[i] * f32(f32(1.0) - c.gate[i])));
        G[a + 'v0'][i] = f32(G[a + 'v0'][i] + gpre[i]);
      }
      const ghv = new F(c.hv.length);
      this._xMBack(a + 'v2', c.hv, gpre, ghv);
      this._xMBack(a + 'v1', c.xv, ghv, gxv);
    }
    const gk0 = new F(C);
    for (let i = 0; i < C; i++) {
      const ka = P[a + 'k_a'][i];
      gk0[i] = f32(gk0[i] + f32(gk[i] * f32(f32(1.0) + f32(f32(c.av[i] - f32(1.0)) * ka))));
      ga[i] = f32(ga[i] + f32(f32(gk[i] * c.k0[i]) * ka));
      G[a + 'k_a'][i] = f32(G[a + 'k_a'][i] + f32(f32(gk[i] * c.k0[i]) * f32(c.av[i] - f32(1.0))));
    }
    for (let h = 0; h < H; h++) {
      const hb = h * N;
      let dot = f32(0);
      for (let j = 0; j < N; j++) dot = f32(dot + f32(c.kk[hb + j] * gkk[hb + j]));
      for (let j = 0; j < N; j++) {
        const gkr = f32(f32(gkk[hb + j] - f32(c.kk[hb + j] * dot)) / c.knrm[h]);
        gk0[hb + j] = f32(gk0[hb + j] + f32(gkr * P[a + 'k_k'][hb + j]));
        G[a + 'k_k'][hb + j] = f32(G[a + 'k_k'][hb + j] + f32(gkr * c.k0[hb + j]));
      }
    }
    const gpa = new F(C);
    for (let i = 0; i < C; i++) {
      gpa[i] = f32(ga[i] * f32(c.av[i] * f32(f32(1.0) - c.av[i])));
      G[a + 'a0'][i] = f32(G[a + 'a0'][i] + gpa[i]);
    }
    const gha = new F(c.ha.length), gxa = new F(C);
    this._xMBack(a + 'a2', c.ha, gpa, gha);
    this._xMBack(a + 'a1', c.xa, gha, gxa);

    const gsg = new F(c.sg.length), gxg = new F(C);
    this._xMBack(a + 'g2', c.sg, gg, gsg);
    const ghg = new F(gsg.length);
    for (let d = 0; d < gsg.length; d++) ghg[d] = f32(gsg[d] * f32(c.sg[d] * f32(f32(1.0) - c.sg[d])));
    this._xMBack(a + 'g1', c.xg, ghg, gxg);

    const gxr = this._linBack(a + 'receptance.weight', c.xr, gr);
    const gxk = this._linBack(a + 'key.weight', c.xk, gk0);
    const gxv2 = this._linBack(a + 'value.weight', c.xv, gvb);
    for (let i = 0; i < C; i++) gxv[i] = f32(gxv[i] + gxv2[i]);

    const gx = new F(C), gxp = new F(C);
    for (const [n, gsrc] of [['x_r', gxr], ['x_w', gxw], ['x_k', gxk], ['x_v', gxv], ['x_a', gxa], ['x_g', gxg]]) {
      const m = P[a + n], gm = G[a + n];
      for (let i = 0; i < C; i++) {
        gx[i] = f32(gx[i] + f32(gsrc[i] * f32(f32(1.0) - m[i])));
        gxp[i] = f32(gxp[i] + f32(gsrc[i] * m[i]));
        gm[i] = f32(gm[i] + f32(gsrc[i] * f32(c.xp[i] - c.x[i])));
      }
    }
    return { gx, gxp, gS0 };
  }

  _cmix(l, x, xp) {
    const F = this.Float, C = this.C, f = `blocks.${l}.ffn.`, m = this.p[f + 'x_k'], kin = new F(C);
    for (let i = 0; i < C; i++) kin[i] = f32(x[i] + f32(f32(xp[i] - x[i]) * m[i]));
    const hk = this._lin(f + 'key.weight', kin);
    const rel = new F(hk.length);
    for (let i = 0; i < hk.length; i++) rel[i] = hk[i] > f32(0) ? f32(hk[i] * hk[i]) : f32(0);
    return { out: this._lin(f + 'value.weight', rel), c: { l, x, xp, kin, hk, rel } };
  }

  _cmixBack(c, gout) {
    const F = this.Float, C = this.C, f = `blocks.${c.l}.ffn.`;
    const grel = this._linBack(f + 'value.weight', c.rel, gout);
    const ghk = new F(grel.length);
    for (let i = 0; i < grel.length; i++) ghk[i] = c.hk[i] > f32(0) ? f32(f32(grel[i] * f32(2.0)) * c.hk[i]) : f32(0);
    const gkin = this._linBack(f + 'key.weight', c.kin, ghk);
    const m = this.p[f + 'x_k'], gm = this.g[f + 'x_k'], gx = new F(C), gxp = new F(C);
    for (let i = 0; i < C; i++) {
      gx[i] = f32(gkin[i] * f32(f32(1.0) - m[i]));
      gxp[i] = f32(gkin[i] * m[i]);
      gm[i] = f32(gm[i] + f32(gkin[i] * f32(c.xp[i] - c.x[i])));
    }
    return { gx, gxp };
  }

  forward(tokenId, state) {
    const C = this.C, E = this.p['emb.weight'];
    const e = E.subarray(tokenId * C, tokenId * C + C);
    const ln0 = this._ln(e, 'blocks.0.ln0.weight', 'blocks.0.ln0.bias');
    let x = ln0.y, vFirst = null;
    const layers = [], newState = [];
    for (let l = 0; l < this.L; l++) {
      const b = `blocks.${l}.`, st = state[l];
      const l1 = this._ln(x, b + 'ln1.weight', b + 'ln1.bias');
      const tm = this._tmix(l, l1.y, st.xa, vFirst, st.S);
      vFirst = tm.vFirst;
      const x1 = new this.Float(C);
      for (let i = 0; i < C; i++) x1[i] = f32(x[i] + tm.out[i]);
      const l2 = this._ln(x1, b + 'ln2.weight', b + 'ln2.bias');
      const cm = this._cmix(l, l2.y, st.xf);
      const xNext = new this.Float(C);
      for (let i = 0; i < C; i++) xNext[i] = f32(x1[i] + cm.out[i]);
      x = xNext;
      layers.push({ l1: l1.c, tm: tm.c, l2: l2.c, cm: cm.c });
      newState.push({ xa: l1.y, S: tm.S, xf: l2.y });
    }
    const lo = this._ln(x, 'ln_out.weight', 'ln_out.bias');
    const logits = this._lin('head.weight', lo.y);
    return { logits, state: newState, cache: { tokenId, ln0: ln0.c, layers, lo: lo.c, xo: lo.y } };
  }

  backward(gLogits, cache, gState = null) {
    const F = this.Float, C = this.C;
    let gx = this._lnBack(this._linBack('head.weight', cache.xo, gLogits), cache.lo);
    const gvFirst = new F(C), out = new Array(this.L);
    for (let l = this.L - 1; l >= 0; l--) {
      const lc = cache.layers[l], gs = gState ? gState[l] : null;
      const cm = this._cmixBack(lc.cm, gx);
      const ga2 = cm.gx;
      if (gs) for (let i = 0; i < C; i++) ga2[i] = f32(ga2[i] + gs.xf[i]);
      const gx1 = this._lnBack(ga2, lc.l2);
      for (let i = 0; i < C; i++) gx1[i] = f32(gx1[i] + gx[i]);
      const tm = this._tmixBack(lc.tm, gx1, gs ? gs.S : null, gvFirst);
      const ga1 = tm.gx;
      if (gs) for (let i = 0; i < C; i++) ga1[i] = f32(ga1[i] + gs.xa[i]);
      const gx0 = this._lnBack(ga1, lc.l1);
      for (let i = 0; i < C; i++) gx0[i] = f32(gx0[i] + gx1[i]);
      gx = gx0;
      out[l] = { xa: tm.gxp, S: tm.gS0, xf: cm.gxp };
    }
    const ge = this._lnBack(gx, cache.ln0);
    const G = this.g['emb.weight'], o = cache.tokenId * C;
    for (let i = 0; i < C; i++) G[o + i] = f32(G[o + i] + ge[i]);
    return out;
  }
}

// ----------------------------------------------------------------------------
// 3. PROCEDURAL TASK FAMILIES (4 Families, Standard & Long OOD Settings)
// ----------------------------------------------------------------------------
const TASK_FAMILIES = ['carry_add', 'brackets', 'kv_recall', 'grammar'];

function generateTaskSample(familyIndex, seed, isLong = false) {
  const rng = new DeterministicPRNG(seed >>> 0);
  const fam = TASK_FAMILIES[familyIndex % 4];
  const tokens = [1]; // BOS
  const mask = [0];

  if (fam === 'carry_add') {
    const digits = isLong ? 4 : 2;
    const a = [], b = [];
    for (let i = 0; i < digits; i++) a.push(rng.nextInt(4));
    for (let i = 0; i < digits; i++) b.push(rng.nextInt(4));
    for (let i = 0; i < digits; i++) { tokens.push(4 + a[i]); mask.push(0); }
    tokens.push(8); mask.push(0);
    for (let i = 0; i < digits; i++) { tokens.push(4 + b[i]); mask.push(0); }
    tokens.push(2); mask.push(0);
    let carry = 0;
    for (let i = 0; i < digits; i++) {
      const sum = a[i] + b[i] + carry;
      tokens.push(4 + (sum % 4));
      mask.push(1);
      carry = Math.floor(sum / 4);
    }
    tokens.push(4 + carry);
    mask.push(1);
  } else if (fam === 'brackets') {
    const steps = isLong ? 6 : 3;
    const stack = [];
    for (let i = 0; i < steps; i++) {
      if (stack.length === 0 || (stack.length < (isLong ? 5 : 3) && rng.nextFloat() < 0.7)) {
        const openTok = rng.nextInt(2) === 0 ? 9 : 11;
        tokens.push(openTok);
        mask.push(0);
        stack.push(openTok === 9 ? 10 : 12);
      } else {
        const closeTok = stack.pop();
        tokens.push(closeTok);
        mask.push(0);
      }
    }
    if (stack.length === 0) {
      const openTok = rng.nextInt(2) === 0 ? 9 : 11;
      tokens.push(openTok);
      mask.push(0);
      stack.push(openTok === 9 ? 10 : 12);
    }
    tokens.push(2); mask.push(0);
    while (stack.length > 0) {
      tokens.push(stack.pop());
      mask.push(1);
    }
  } else if (fam === 'kv_recall') {
    const pairs = isLong ? 5 : 3;
    const availKeys = [13, 14, 15, 16, 17, 18];
    for (let i = availKeys.length - 1; i > 0; i--) {
      const j = rng.nextInt(i + 1);
      const tmp = availKeys[i]; availKeys[i] = availKeys[j]; availKeys[j] = tmp;
    }
    const chosenKeys = availKeys.slice(0, pairs);
    const chosenVals = chosenKeys.map(() => 4 + rng.nextInt(4));
    for (let i = 0; i < pairs; i++) {
      tokens.push(chosenKeys[i]); mask.push(0);
      tokens.push(chosenVals[i]); mask.push(0);
    }
    const queries = isLong ? 2 : 1;
    for (let q = 0; q < queries; q++) {
      const qIdx = rng.nextInt(pairs);
      tokens.push(3); mask.push(0);
      tokens.push(chosenKeys[qIdx]); mask.push(0);
      tokens.push(2); mask.push(0);
      tokens.push(chosenVals[qIdx]); mask.push(1);
    }
  } else {
    const len = isLong ? 6 : 3;
    const rule = rng.nextInt(4);
    tokens.push(19 + rule); mask.push(0);
    const seq = [];
    for (let i = 0; i < len; i++) {
      const s = rng.nextInt(4);
      seq.push(s);
      tokens.push(4 + s);
      mask.push(0);
    }
    tokens.push(2); mask.push(0);
    let outSeq;
    if (rule === 0) outSeq = seq.slice();
    else if (rule === 1) outSeq = seq.map(x => (x + 1) % 4);
    else if (rule === 2) outSeq = seq.slice().reverse();
    else outSeq = seq.map(x => 3 - x);
    for (let i = 0; i < outSeq.length; i++) {
      tokens.push(4 + outSeq[i]);
      mask.push(1);
    }
  }

  return {
    family: fam,
    familyIndex: familyIndex % 4,
    isLong: !!isLong,
    seed: seed >>> 0,
    tokens: new Uint8Array(tokens),
    mask: new Uint8Array(mask)
  };
}

// ----------------------------------------------------------------------------
// 4. AST DSL FOR UPDATE RULES (C) AND SCORERS (S)
// ----------------------------------------------------------------------------
const PRIMITIVE_SPECS = Object.freeze({
  neg:  { arity: 1 },
  abs:  { arity: 1 },
  tanh: { arity: 1 },
  exp:  { arity: 1 },
  log:  { arity: 1 },
  sqrt: { arity: 1 },
  sin:  { arity: 1 },
  clip: { arity: 1 },
  add:  { arity: 2 },
  sub:  { arity: 2 },
  mul:  { arity: 2 },
  div:  { arity: 2 },
  min:  { arity: 2 },
  max:  { arity: 2 },
  pow:  { arity: 2 },
  ite:  { arity: 3 }
});

const ALL_PRIMITIVE_NAMES = Object.freeze(Object.keys(PRIMITIVE_SPECS));
const UNARY_PRIMITIVES = Object.freeze(ALL_PRIMITIVE_NAMES.filter(k => PRIMITIVE_SPECS[k].arity === 1));
const BINARY_PRIMITIVES = Object.freeze(ALL_PRIMITIVE_NAMES.filter(k => PRIMITIVE_SPECS[k].arity === 2));
const TERNARY_PRIMITIVES = Object.freeze(ALL_PRIMITIVE_NAMES.filter(k => PRIMITIVE_SPECS[k].arity === 3));

const UPDATE_RULE_VARS = Object.freeze(['g', 'w', 'm', 'v', 'lr']);
const SCORER_VARS = Object.freeze(['acc', 'loss']);
const AST_MAX_NODES = 256;

function cloneAST(node) {
  if (!node || typeof node !== 'object') throw new Error('Invalid AST node');
  if (node.op === 'var') return { op: 'var', name: String(node.name) };
  if (node.op === 'const') return { op: 'const', val: f32(node.val) };
  return { op: String(node.op), args: (node.args || []).map(cloneAST) };
}

function countASTNodes(node) {
  if (!node || typeof node !== 'object') return 0;
  if (node.op === 'var' || node.op === 'const') return 1;
  let sum = 1;
  for (const a of (node.args || [])) sum += countASTNodes(a);
  return sum;
}

function validateAST(node, allowedVars = UPDATE_RULE_VARS, maxNodes = AST_MAX_NODES) {
  let count = 0;
  function walk(n, depth) {
    if (!n || typeof n !== 'object') throw new Error('AST node must be an object');
    if (depth > 32) throw new Error('AST exceeds maximum depth of 32');
    count++;
    if (count > maxNodes) throw new Error(`AST exceeds node budget (${maxNodes})`);
    if (n.op === 'var') {
      if (typeof n.name !== 'string' || !allowedVars.includes(n.name)) {
        throw new Error(`Disallowed variable in AST: ${n.name}`);
      }
      return;
    }
    if (n.op === 'const') {
      if (typeof n.val !== 'number' || !Number.isFinite(n.val)) {
        throw new Error('AST constant must be a finite number');
      }
      return;
    }
    const spec = PRIMITIVE_SPECS[n.op];
    if (!spec) throw new Error(`Unknown AST primitive: ${n.op}`);
    if (!Array.isArray(n.args) || n.args.length !== spec.arity) {
      throw new Error(`Primitive ${n.op} expects ${spec.arity} args, got ${n.args?.length}`);
    }
    for (const child of n.args) walk(child, depth + 1);
  }
  walk(node, 0);
  return true;
}

function formatAST(node) {
  if (node.op === 'var') return node.name;
  if (node.op === 'const') {
    const v = f32(node.val);
    if (Object.is(v, -0)) return '0';
    return String(v);
  }
  return `${node.op}(${node.args.map(formatAST).join(', ')})`;
}

function parseAST(exprStr, allowedVars = UPDATE_RULE_VARS) {
  const s = String(exprStr || '').trim();
  let pos = 0;

  function skipWS() {
    while (pos < s.length && /\s/.test(s[pos])) pos++;
  }

  function parseExpr() {
    skipWS();
    if (pos >= s.length) throw new Error('Unexpected end of expression');
    const numMatch = s.slice(pos).match(/^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/);
    if (numMatch && (pos + numMatch[0].length >= s.length || /[\s,)]/.test(s[pos + numMatch[0].length]))) {
      pos += numMatch[0].length;
      const val = f32(parseFloat(numMatch[0]));
      if (!Number.isFinite(val)) throw new Error('Non-finite constant in expression');
      return { op: 'const', val };
    }
    const idMatch = s.slice(pos).match(/^[a-zA-Z_][a-zA-Z0-9_]*/);
    if (!idMatch) throw new Error(`Unexpected token at position ${pos}: ${s.slice(pos, pos + 16)}`);
    const id = idMatch[0];
    pos += id.length;
    skipWS();
    if (pos < s.length && s[pos] === '(') {
      pos++;
      const spec = PRIMITIVE_SPECS[id];
      if (!spec) throw new Error(`Unknown primitive in parseAST: ${id}`);
      const args = [];
      for (let i = 0; i < spec.arity; i++) {
        if (i > 0) {
          skipWS();
          if (s[pos] !== ',') throw new Error(`Expected ',' in ${id}() at pos ${pos}`);
          pos++;
        }
        args.push(parseExpr());
      }
      skipWS();
      if (s[pos] !== ')') throw new Error(`Expected ')' after ${id}() at pos ${pos}`);
      pos++;
      return { op: id, args };
    } else {
      if (!allowedVars.includes(id)) throw new Error(`Unknown variable in parseAST: ${id}`);
      return { op: 'var', name: id };
    }
  }

  const ast = parseExpr();
  skipWS();
  if (pos !== s.length) throw new Error(`Trailing characters in expression at pos ${pos}: ${s.slice(pos)}`);
  validateAST(ast, allowedVars);
  return ast;
}

function evalAST(node, env) {
  switch (node.op) {
    case 'var': return f32(env[node.name] || 0);
    case 'const': return f32(node.val);
    case 'neg': return f32(-evalAST(node.args[0], env));
    case 'abs': return f32(Math.abs(evalAST(node.args[0], env)));
    case 'tanh': return detTanh(evalAST(node.args[0], env));
    case 'exp': {
      const a = evalAST(node.args[0], env);
      const clamped = a < f32(-6.0) ? f32(-6.0) : (a > f32(6.0) ? f32(6.0) : a);
      return detExp(clamped);
    }
    case 'log': {
      const a = Math.abs(evalAST(node.args[0], env));
      return detLog(a < f32(1e-6) ? f32(1e-6) : a);
    }
    case 'sqrt': return detSqrt(Math.abs(evalAST(node.args[0], env)));
    case 'sin': return detSin(evalAST(node.args[0], env));
    case 'clip': {
      const a = evalAST(node.args[0], env);
      if (Number.isNaN(a)) return f32(0);
      return a < f32(-1.0) ? f32(-1.0) : (a > f32(1.0) ? f32(1.0) : a);
    }
    case 'add': return f32(evalAST(node.args[0], env) + evalAST(node.args[1], env));
    case 'sub': return f32(evalAST(node.args[0], env) - evalAST(node.args[1], env));
    case 'mul': return f32(evalAST(node.args[0], env) * evalAST(node.args[1], env));
    case 'div': {
      const num = evalAST(node.args[0], env);
      const den = evalAST(node.args[1], env);
      const safeDen = Math.abs(den) < f32(1e-4) ? (den < 0 ? f32(-1e-4) : f32(1e-4)) : den;
      return f32(num / safeDen);
    }
    case 'min': {
      const a = evalAST(node.args[0], env), b = evalAST(node.args[1], env);
      return a <= b ? a : b;
    }
    case 'max': {
      const a = evalAST(node.args[0], env), b = evalAST(node.args[1], env);
      return a >= b ? a : b;
    }
    case 'pow': {
      const a = Math.abs(evalAST(node.args[0], env));
      const b = evalAST(node.args[1], env);
      const safeA = a < f32(1e-4) ? f32(1e-4) : (a > f32(10.0) ? f32(10.0) : a);
      const safeB = b < f32(-3.0) ? f32(-3.0) : (b > f32(3.0) ? f32(3.0) : b);
      return detExp(f32(safeB * detLog(safeA)));
    }
    case 'ite': {
      const cond = evalAST(node.args[0], env);
      return cond > f32(0) ? evalAST(node.args[1], env) : evalAST(node.args[2], env);
    }
    default:
      throw new Error(`Unknown AST op: ${node.op}`);
  }
}

// ----------------------------------------------------------------------------
// 5. GENOME DEFAULTS, EVALUATION, AND HASHING
// ----------------------------------------------------------------------------
function createGen0UpdateRule() {
  return {
    op: 'mul',
    args: [
      { op: 'var', name: 'lr' },
      { op: 'var', name: 'g' }
    ]
  };
}

function createGen0Scorer() {
  return {
    op: 'sub',
    args: [
      { op: 'var', name: 'acc' },
      {
        op: 'mul',
        args: [
          { op: 'const', val: f32(0.25) },
          { op: 'var', name: 'loss' }
        ]
      }
    ]
  };
}

function createGen0Genome() {
  return {
    hyper: {
      lr: f32(0.04),
      momentum: f32(0.9),
      steps: 18,
      gradClip: f32(1.0),
      ternaryMix: f32(0.2),
      ternaryThreshold: f32(0.5),
      curriculum: [f32(0.25), f32(0.25), f32(0.25), f32(0.25)]
    },
    updateRule: createGen0UpdateRule(),
    scorer: createGen0Scorer()
  };
}

function cloneGenome(g) {
  return {
    hyper: {
      lr: f32(g.hyper.lr),
      momentum: f32(g.hyper.momentum),
      steps: g.hyper.steps | 0,
      gradClip: f32(g.hyper.gradClip),
      ternaryMix: f32(g.hyper.ternaryMix),
      ternaryThreshold: f32(g.hyper.ternaryThreshold),
      curriculum: g.hyper.curriculum.map(x => f32(x))
    },
    updateRule: cloneAST(g.updateRule),
    scorer: cloneAST(g.scorer)
  };
}

function computeGenomeHash(genome) {
  return sha256Bytes(canonicalJSON(genome));
}

function computeGenomeDiff(parentGenome, childGenome) {
  const diff = {};
  for (const k of ['lr', 'momentum', 'steps', 'gradClip', 'ternaryMix', 'ternaryThreshold']) {
    if (parentGenome.hyper[k] !== childGenome.hyper[k]) {
      diff[k] = { from: parentGenome.hyper[k], to: childGenome.hyper[k] };
    }
  }
  const pCurr = canonicalJSON(parentGenome.hyper.curriculum);
  const cCurr = canonicalJSON(childGenome.hyper.curriculum);
  if (pCurr !== cCurr) {
    diff.curriculum = { from: parentGenome.hyper.curriculum, to: childGenome.hyper.curriculum };
  }
  const pUpd = formatAST(parentGenome.updateRule);
  const cUpd = formatAST(childGenome.updateRule);
  if (pUpd !== cUpd) {
    diff.updateRule = { from: pUpd, to: cUpd };
  }
  const pScr = formatAST(parentGenome.scorer);
  const cScr = formatAST(childGenome.scorer);
  if (pScr !== cScr) {
    diff.scorer = { from: pScr, to: cScr };
  }
  return diff;
}

function softmaxProbs(logits) {
  const V = logits.length;
  let maxLogit = f32(-1e30);
  for (let i = 0; i < V; i++) {
    if (logits[i] > maxLogit) maxLogit = f32(logits[i]);
  }
  const probs = new Float32Array(V);
  let sum = f32(0);
  for (let i = 0; i < V; i++) {
    const e = detExp(f32(logits[i] - maxLogit));
    probs[i] = e;
    sum = f32(sum + e);
  }
  const invSum = f32(f32(1.0) / (sum > f32(1e-12) ? sum : f32(1e-12)));
  for (let i = 0; i < V; i++) probs[i] = f32(probs[i] * invSum);
  return probs;
}

function evaluateModelOnSamples(model, samples, scorerAST) {
  let totalTargets = 0;
  let totalCorrect = 0;
  let totalLoss = f32(0);
  const famStats = [
    { correct: 0, total: 0, loss: f32(0) },
    { correct: 0, total: 0, loss: f32(0) },
    { correct: 0, total: 0, loss: f32(0) },
    { correct: 0, total: 0, loss: f32(0) }
  ];

  for (let s = 0; s < samples.length; s++) {
    const sample = samples[s];
    const toks = sample.tokens;
    const mask = sample.mask;
    let state = model.getEmptyState();
    for (let t = 0; t < toks.length - 1; t++) {
      const res = model.forward(toks[t], state);
      state = res.state;
      if (mask[t + 1] === 1) {
        const target = toks[t + 1];
        const probs = softmaxProbs(res.logits);
        let bestIdx = 0;
        let bestVal = probs[0];
        for (let v = 1; v < probs.length; v++) {
          if (probs[v] > bestVal) {
            bestVal = probs[v];
            bestIdx = v;
          }
        }
        const hit = bestIdx === target ? 1 : 0;
        const pTarget = probs[target] > f32(1e-6) ? probs[target] : f32(1e-6);
        const stepLoss = f32(-detLog(pTarget));
        totalTargets++;
        totalCorrect += hit;
        totalLoss = f32(totalLoss + stepLoss);
        const fs = famStats[sample.familyIndex];
        fs.total++;
        fs.correct += hit;
        fs.loss = f32(fs.loss + stepLoss);
      }
    }
  }

  const acc = totalTargets > 0 ? f32(totalCorrect / totalTargets) : f32(0);
  const loss = totalTargets > 0 ? f32(totalLoss / totalTargets) : f32(0);
  const gen0AST = createGen0Scorer();
  const gen0Score = evalAST(gen0AST, { acc, loss });
  const score = scorerAST ? evalAST(scorerAST, { acc, loss }) : gen0Score;

  const perFamily = {};
  for (let i = 0; i < 4; i++) {
    const fs = famStats[i];
    const fAcc = fs.total > 0 ? f32(fs.correct / fs.total) : f32(0);
    const fLoss = fs.total > 0 ? f32(fs.loss / fs.total) : f32(0);
    perFamily[TASK_FAMILIES[i]] = {
      acc: fAcc,
      loss: fLoss,
      gen0Score: evalAST(gen0AST, { acc: fAcc, loss: fLoss })
    };
  }

  return { acc, loss, score, gen0Score, perFamily };
}

function computeRecordHash(record) {
  const canonicalPayload = {
    seqIndex: record.seqIndex,
    gen: record.gen,
    branchId: record.branchId,
    isForkRoot: !!record.isForkRoot,
    parentSeqIndex: record.parentSeqIndex,
    parentGen: record.parentGen,
    parentHash: record.parentHash,
    operator: record.operator,
    operatorDetail: record.operatorDetail,
    genomeDiff: record.genomeDiff,
    candidateGenome: record.candidateGenome,
    activeGenome: record.activeGenome,
    genomeHash: record.genomeHash,
    seeds: record.seeds,
    devAcc: record.devAcc,
    devLoss: record.devLoss,
    devScore: record.devScore,
    gen0DevScore: record.gen0DevScore,
    accepted: record.accepted,
    gateReason: record.gateReason,
    activeGen: record.activeGen,
    candidateWeightsHash: record.candidateWeightsHash,
    weightsHash: record.weightsHash
  };
  return sha256Bytes(canonicalJSON(canonicalPayload));
}


// ── Trainer Seed Domain (Strictly [0x00000000, 0x7FFFFFFF], MSB = 0) ────────
const TRAINER_DOMAIN_MASK = 0x7fffffff;

function deriveTrainerSeed(rawUint32) {
  return ((rawUint32 >>> 0) & TRAINER_DOMAIN_MASK) >>> 0;
}

function deriveTrainerDevSeeds(rootSeed, countPerFamily = 4) {
  const rng = new DeterministicPRNG(((rootSeed >>> 0) ^ 0x243f6a88) >>> 0);
  const devSeeds = [];
  for (let fam = 0; fam < 4; fam++) {
    for (let i = 0; i < countPerFamily; i++) {
      const s = deriveTrainerSeed(rng.nextUint32());
      devSeeds.push({ familyIndex: fam, seed: s });
    }
  }
  return devSeeds;
}

// ── Random Expression Generator & AST Operators (C and S) ───────────────────
const AST_MUTATION_OPS = Object.freeze([
  'const_perturb',
  'subtree_replace',
  'wrap_primitive',
  'arg_swap',
  'ancestor_splice'
]);

function randomTerminal(rng, allowedVars) {
  if (rng.nextFloat() < 0.65 && allowedVars.length > 0) {
    const v = allowedVars[rng.nextInt(allowedVars.length)];
    return { op: 'var', name: v };
  }
  const c = f32((rng.nextFloat() * f32(3.0)) - f32(1.5));
  return { op: 'const', val: Math.abs(c) < f32(0.01) ? f32(0.1) : c };
}

function randomExpression(rng, allowedVars = UPDATE_RULE_VARS, maxDepth = 3, forcePrimitive = false) {
  if (maxDepth <= 0 || (!forcePrimitive && rng.nextFloat() < 0.35)) {
    return randomTerminal(rng, allowedVars);
  }
  const prim = ALL_PRIMITIVE_NAMES[rng.nextInt(ALL_PRIMITIVE_NAMES.length)];
  const arity = PRIMITIVE_SPECS[prim].arity;
  const args = [];
  for (let i = 0; i < arity; i++) {
    args.push(randomExpression(rng, allowedVars, maxDepth - 1, false));
  }
  return { op: prim, args };
}

function collectASTPaths(root) {
  const list = [];
  function walk(node, path) {
    list.push({ node, path });
    if (node.args) {
      for (let i = 0; i < node.args.length; i++) {
        walk(node.args[i], path.concat([i]));
      }
    }
  }
  walk(root, []);
  return list;
}

function replaceAtASTPath(root, path, replacement) {
  const cloned = cloneAST(root);
  if (path.length === 0) return cloneAST(replacement);
  let cur = cloned;
  for (let i = 0; i < path.length - 1; i++) {
    cur = cur.args[path[i]];
  }
  cur.args[path[path.length - 1]] = cloneAST(replacement);
  return cloned;
}

function mutateAST(ast, rng, allowedVars, ancestorASTs = [], forcedSubOp = null) {
  const subOp = forcedSubOp || AST_MUTATION_OPS[rng.nextInt(AST_MUTATION_OPS.length)];
  const root = cloneAST(ast);
  const allNodes = collectASTPaths(root);
  let candidate;

  if (subOp === 'const_perturb') {
    const constNodes = allNodes.filter(x => x.node.op === 'const');
    if (constNodes.length > 0) {
      const target = constNodes[rng.nextInt(constNodes.length)];
      const delta = rng.nextSymNoise(0.35);
      const newConst = { op: 'const', val: f32(target.node.val + delta) };
      candidate = replaceAtASTPath(root, target.path, newConst);
    } else {
      const target = allNodes[rng.nextInt(allNodes.length)];
      const delta = rng.nextSymNoise(0.25);
      const wrapped = { op: 'add', args: [cloneAST(target.node), { op: 'const', val: delta }] };
      candidate = replaceAtASTPath(root, target.path, wrapped);
    }
  } else if (subOp === 'subtree_replace') {
    const target = allNodes[rng.nextInt(allNodes.length)];
    const sub = randomExpression(rng, allowedVars, 2, true);
    candidate = replaceAtASTPath(root, target.path, sub);
  } else if (subOp === 'wrap_primitive') {
    const target = allNodes[rng.nextInt(allNodes.length)];
    const prim = ALL_PRIMITIVE_NAMES[rng.nextInt(ALL_PRIMITIVE_NAMES.length)];
    const arity = PRIMITIVE_SPECS[prim].arity;
    const insertPos = rng.nextInt(arity);
    const args = [];
    for (let i = 0; i < arity; i++) {
      if (i === insertPos) args.push(cloneAST(target.node));
      else args.push(randomTerminal(rng, allowedVars));
    }
    candidate = replaceAtASTPath(root, target.path, { op: prim, args });
  } else if (subOp === 'arg_swap') {
    const multiNodes = allNodes.filter(x => x.node.args && x.node.args.length >= 2);
    if (multiNodes.length > 0) {
      const target = multiNodes[rng.nextInt(multiNodes.length)];
      const nCopy = cloneAST(target.node);
      if (nCopy.args.length === 2) {
        const tmp = nCopy.args[0]; nCopy.args[0] = nCopy.args[1]; nCopy.args[1] = tmp;
      } else {
        const tmp = nCopy.args[1]; nCopy.args[1] = nCopy.args[2]; nCopy.args[2] = tmp;
      }
      candidate = replaceAtASTPath(root, target.path, nCopy);
    } else {
      const binPrim = BINARY_PRIMITIVES[rng.nextInt(BINARY_PRIMITIVES.length)];
      candidate = { op: binPrim, args: [randomTerminal(rng, allowedVars), cloneAST(root)] };
    }
  } else if (subOp === 'ancestor_splice') {
    const pool = ancestorASTs && ancestorASTs.length > 0 ? ancestorASTs : [root];
    const donorAST = pool[rng.nextInt(pool.length)];
    const donorNodes = collectASTPaths(donorAST);
    const donorSub = donorNodes[rng.nextInt(donorNodes.length)].node;
    const target = allNodes[rng.nextInt(allNodes.length)];
    candidate = replaceAtASTPath(root, target.path, donorSub);
  } else {
    throw new Error(`Unknown AST mutation subOp: ${subOp}`);
  }

  try {
    validateAST(candidate, allowedVars, 64);
  } catch (e) {
    candidate = randomExpression(rng, allowedVars, 2, true);
  }
  return { ast: candidate, detail: subOp };
}

// ── Hyperparameter Bounds & Symmetric Zero-Centered Perturbation (H) ────────
const HYPER_BOUNDS = Object.freeze({
  lr:               { min: f32(0.001), max: f32(0.15),  scale: f32(0.015) },
  momentum:         { min: f32(0.0),   max: f32(0.98),  scale: f32(0.08)  },
  steps:            { min: 6,          max: 40,         scale: 4          },
  gradClip:         { min: f32(0.1),   max: f32(5.0),   scale: f32(0.35)  },
  ternaryMix:       { min: f32(0.0),   max: f32(1.0),   scale: f32(0.15)  },
  ternaryThreshold: { min: f32(0.15),  max: f32(0.85),  scale: f32(0.10)  },
  curriculumWeight: { min: f32(0.05),  max: f32(0.85),  scale: f32(0.10)  }
});

function clampVal(x, minV, maxV) {
  return x < minV ? minV : (x > maxV ? maxV : x);
}

function mutateHyperparameters(hyper, rng) {
  const next = {
    lr: f32(hyper.lr),
    momentum: f32(hyper.momentum),
    steps: hyper.steps | 0,
    gradClip: f32(hyper.gradClip),
    ternaryMix: f32(hyper.ternaryMix),
    ternaryThreshold: f32(hyper.ternaryThreshold),
    curriculum: hyper.curriculum.map(x => f32(x))
  };
  const keys = ['lr', 'momentum', 'steps', 'gradClip', 'ternaryMix', 'ternaryThreshold', 'curriculum'];
  const numToPerturb = 1 + rng.nextInt(2);
  const touched = [];
  for (let n = 0; n < numToPerturb; n++) {
    const k = keys[rng.nextInt(keys.length)];
    touched.push(k);
    if (k === 'steps') {
      const b = HYPER_BOUNDS.steps;
      const delta = Math.round(rng.nextSymNoise(b.scale));
      next.steps = clampVal((next.steps + (delta === 0 ? (rng.nextInt(2) === 0 ? -1 : 1) : delta)) | 0, b.min, b.max);
    } else if (k === 'curriculum') {
      const b = HYPER_BOUNDS.curriculumWeight;
      let sum = f32(0);
      for (let i = 0; i < 4; i++) {
        const w = clampVal(f32(next.curriculum[i] + rng.nextSymNoise(b.scale)), b.min, b.max);
        next.curriculum[i] = w;
        sum = f32(sum + w);
      }
      for (let i = 0; i < 4; i++) next.curriculum[i] = f32(next.curriculum[i] / sum);
    } else {
      const b = HYPER_BOUNDS[k];
      next[k] = f32(clampVal(f32(next[k] + rng.nextSymNoise(b.scale)), b.min, b.max));
    }
  }
  return { hyper: next, detail: touched.join('+') };
}

// ── Arms & Operator Selection (§4 & §5: Zero Arm-Specific Side Effects) ─────
const DEFAULT_OPERATOR_WEIGHTS = Object.freeze({
  W: 0.30,
  H: 0.30,
  C: 0.20,
  S: 0.20
});

const ARM_ENABLED_OPERATORS = Object.freeze({
  'train':       Object.freeze(['W']),
  'hyper':       Object.freeze(['W', 'H']),
  'code':        Object.freeze(['W', 'H', 'C']),
  'code+scorer': Object.freeze(['W', 'H', 'C', 'S'])
});

function getArmProbabilities(armName, customWeights = null) {
  const enabled = ARM_ENABLED_OPERATORS[armName];
  if (!enabled) throw new Error(`Unknown arm "${armName}". Expected train, hyper, code, or code+scorer.`);
  const base = Object.assign({}, DEFAULT_OPERATOR_WEIGHTS, customWeights || {});
  const raw = { W: 0, H: 0, C: 0, S: 0 };
  let total = 0;
  for (const op of enabled) {
    const w = Math.max(0, Number(base[op]) || 0);
    raw[op] = w;
    total += w;
  }
  if (total <= 0) {
    return { W: 1.0, H: 0.0, C: 0.0, S: 0.0 };
  }
  return {
    W: raw.W / total,
    H: raw.H / total,
    C: raw.C / total,
    S: raw.S / total
  };
}

function selectOperator(rng, armName, customWeights = null) {
  const probs = getArmProbabilities(armName, customWeights);
  const u = rng.nextFloat();
  let acc = 0;
  for (const op of ['W', 'H', 'C', 'S']) {
    acc += probs[op];
    if (u < acc && probs[op] > 0) return op;
  }
  for (const op of ['S', 'C', 'H', 'W']) {
    if (probs[op] > 0) return op;
  }
  return 'W';
}

function applyChangeOperator(operator, parentGenome, rng, ancestorGenomes = []) {
  const next = cloneGenome(parentGenome);
  if (operator === 'W') {
    return { genome: next, detail: 'train_only' };
  }
  if (operator === 'H') {
    const res = mutateHyperparameters(next.hyper, rng);
    next.hyper = res.hyper;
    return { genome: next, detail: res.detail };
  }
  if (operator === 'C') {
    const ancestors = ancestorGenomes.map(g => g.updateRule);
    const res = mutateAST(next.updateRule, rng, UPDATE_RULE_VARS, ancestors);
    next.updateRule = res.ast;
    return { genome: next, detail: res.detail };
  }
  if (operator === 'S') {
    const ancestors = ancestorGenomes.map(g => g.scorer);
    const res = mutateAST(next.scorer, rng, SCORER_VARS, ancestors);
    next.scorer = res.ast;
    return { genome: next, detail: res.detail };
  }
  throw new Error(`Invalid operator: ${operator}`);
}

// ── Training One Sequence with BPTT & Genome Update Rule ────────────────────
function trainOneStep(model, sample, genome) {
  const toks = sample.tokens;
  const mask = sample.mask;
  const T = toks.length - 1;
  const states = [model.getEmptyState()];
  const fwdTraces = [];
  const stepLogits = [];

  model.ternaryMix = f32(genome.hyper.ternaryMix);
  model.ternaryThreshold = f32(genome.hyper.ternaryThreshold);

  for (let t = 0; t < T; t++) {
    const res = model.forward(toks[t], states[t]);
    states.push(res.state);
    fwdTraces.push(res.cache);
    stepLogits.push(res.logits);
  }

  let targetCount = 0;
  for (let t = 0; t < T; t++) {
    if (mask[t + 1] === 1) targetCount++;
  }
  const invTargets = targetCount > 0 ? f32(1.0 / targetCount) : f32(1.0);

  model.zeroGrad();
  let gState = null;
  let stepLoss = f32(0);

  for (let t = T - 1; t >= 0; t--) {
    const gLogits = new Float32Array(model.V);
    if (mask[t + 1] === 1) {
      const target = toks[t + 1];
      const probs = softmaxProbs(stepLogits[t]);
      const pTarget = probs[target] > f32(1e-6) ? probs[target] : f32(1e-6);
      stepLoss = f32(stepLoss - detLog(pTarget));
      for (let v = 0; v < model.V; v++) {
        const ind = v === target ? f32(1.0) : f32(0.0);
        gLogits[v] = f32(f32(probs[v] - ind) * invTargets);
      }
    }
    gState = model.backward(gLogits, fwdTraces[t], gState);
  }

  const clip = f32(genome.hyper.gradClip);
  const lr = f32(genome.hyper.lr);
  const mom = f32(genome.hyper.momentum);
  const oneMinusMom = f32(f32(1.0) - mom);
  const beta2 = f32(0.99);
  const oneMinusBeta2 = f32(0.01);
  const updateAST = genome.updateRule;

  let hasNaNOrInf = false;
  let stepAbsSum = f32(0);
  let maxStepAbs = f32(0);
  let maxWeightAbs = f32(0);
  let gradAbsSum = f32(0);

  for (const name of model.paramOrder) {
    const W = model.p[name];
    const G = model.g[name];
    const M = model.m[name];
    const V = model.v[name];
    const len = W.length;
    for (let i = 0; i < len; i++) {
      let g = f32(G[i]);
      if (!Number.isFinite(g)) {
        hasNaNOrInf = true;
        g = f32(0);
      }
      gradAbsSum = f32(gradAbsSum + Math.abs(g));
      if (g > clip) g = clip;
      else if (g < f32(-clip)) g = f32(-clip);

      const mNew = f32(f32(mom * M[i]) + f32(oneMinusMom * g));
      const vNew = f32(f32(beta2 * V[i]) + f32(oneMinusBeta2 * f32(g * g)));
      M[i] = mNew;
      V[i] = vNew;

      const wCur = W[i];
      const dw = evalAST(updateAST, { g, w: wCur, m: mNew, v: vNew, lr });
      if (!Number.isFinite(dw)) {
        hasNaNOrInf = true;
        continue;
      }
      const absDw = Math.abs(dw);
      stepAbsSum = f32(stepAbsSum + absDw);
      if (absDw > maxStepAbs) maxStepAbs = f32(absDw);

      const wNext = f32(wCur - dw);
      if (!Number.isFinite(wNext)) {
        hasNaNOrInf = true;
        continue;
      }
      W[i] = wNext;
      const absW = Math.abs(wNext);
      if (absW > maxWeightAbs) maxWeightAbs = f32(absW);
    }
  }

  return {
    loss: targetCount > 0 ? f32(stepLoss / targetCount) : f32(0),
    hasNaNOrInf,
    stepAbsSum,
    maxStepAbs,
    maxWeightAbs,
    gradAbsSum
  };
}

function sampleCurriculumFamily(rng, curriculum) {
  const u = rng.nextFloat();
  let acc = 0;
  for (let i = 0; i < 4; i++) {
    acc += curriculum[i];
    if (u <= acc) return i;
  }
  return 3;
}

// Gate evaluation rule (§3.5 & §9.10)
function evaluateGate({
  isGen0,
  hasNaNOrInf,
  totalStepAbsSum,
  totalGradAbsSum,
  maxStepAbs,
  maxWeightAbs,
  candidateDevScore,
  gen0DevScore,
  parentDevScore
}) {
  if (maxStepAbs > f32(50.0) || maxWeightAbs > f32(100.0)) {
    return { accepted: false, reason: 'reject_explosive_step' };
  }
  if (hasNaNOrInf || !Number.isFinite(candidateDevScore) || !Number.isFinite(gen0DevScore)) {
    return { accepted: false, reason: 'reject_nan_or_inf' };
  }
  if (totalGradAbsSum > f32(1e-8) && totalStepAbsSum <= f32(1e-9)) {
    return { accepted: false, reason: 'reject_zero_step' };
  }
  if (isGen0) {
    return { accepted: true, reason: 'accept_gen0_baseline' };
  }
  if (candidateDevScore > parentDevScore) {
    return { accepted: true, reason: 'accept_improved_dev_score' };
  }
  return { accepted: false, reason: 'reject_score_not_beaten' };
}

// ── Generational Trainer Session ────────────────────────────────────────────
class GTEETrainer {
  constructor(opts = {}) {
    this.rootSeed = deriveTrainerSeed(opts.seed !== undefined ? opts.seed : 42);
    this.arm = opts.arm || 'train';
    this.customOperatorWeights = opts.operatorWeights || null;
    this.modelConfig = Object.assign({}, DEFAULT_MODEL_CONFIG, opts.modelConfig || {});
    this.devSeedSpecs = deriveTrainerDevSeeds(this.rootSeed, 4);
    this.devSamples = this.devSeedSpecs.map(sp => generateTaskSample(sp.familyIndex, sp.seed, false));

    this.rng = new DeterministicPRNG(this.rootSeed);
    this.model = new RWKV7(this.modelConfig, { rng: this.rng });
    this.lineage = [];
    this.activeSeqIndex = -1;
    this.branchCounter = 0;
    this.activeBranchId = 'main';
    this.pendingFork = false;
  }

  initGen0() {
    if (this.lineage.length > 0) return this.lineage[0];
    const t0 = Date.now();
    const rngBefore = this.rng.getState();
    const gen0Genome = createGen0Genome();
    const trainSeeds = [];

    let hasNaNOrInf = false;
    let totalStepAbsSum = f32(0);
    let totalGradAbsSum = f32(0);
    let maxStepAbs = f32(0);
    let maxWeightAbs = f32(0);

    for (let s = 0; s < gen0Genome.hyper.steps; s++) {
      const fam = sampleCurriculumFamily(this.rng, gen0Genome.hyper.curriculum);
      const stepSeed = deriveTrainerSeed(this.rng.nextUint32());
      trainSeeds.push({ familyIndex: fam, seed: stepSeed });
      const sample = generateTaskSample(fam, stepSeed, false);
      const st = trainOneStep(this.model, sample, gen0Genome);
      if (st.hasNaNOrInf) hasNaNOrInf = true;
      totalStepAbsSum = f32(totalStepAbsSum + st.stepAbsSum);
      totalGradAbsSum = f32(totalGradAbsSum + st.gradAbsSum);
      if (st.maxStepAbs > maxStepAbs) maxStepAbs = st.maxStepAbs;
      if (st.maxWeightAbs > maxWeightAbs) maxWeightAbs = st.maxWeightAbs;
    }

    const evalRes = evaluateModelOnSamples(this.model, this.devSamples, gen0Genome.scorer);
    const weightsHash = this.model.weightsHash();
    const weightsB64 = this.model.weightsToBase64();
    const bufs = this.model.exportBuffersFlat();
    const elapsedMs = Math.max(1, Date.now() - t0);

    const record = {
      seqIndex: 0,
      gen: 0,
      branchId: 'main',
      isForkRoot: false,
      parentSeqIndex: null,
      parentGen: null,
      parentHash: '0'.repeat(64),
      operator: 'INIT',
      operatorDetail: 'gen0_clipped_sgd',
      genomeDiff: {},
      candidateGenome: cloneGenome(gen0Genome),
      activeGenome: cloneGenome(gen0Genome),
      genomeHash: computeGenomeHash(gen0Genome),
      seeds: {
        train: trainSeeds,
        dev: this.devSeedSpecs
      },
      devAcc: evalRes.acc,
      devLoss: evalRes.loss,
      devScore: evalRes.score,
      gen0DevScore: evalRes.gen0Score,
      accepted: true,
      gateReason: 'accept_gen0_baseline',
      activeGen: 0,
      candidateWeightsHash: weightsHash,
      weightsHash: weightsHash,
      weightsB64: weightsB64,
      mFlatB64: bytesToBase64(new Uint8Array(bufs.mFlat.buffer)),
      vFlatB64: bytesToBase64(new Uint8Array(bufs.vFlat.buffer)),
      rngStateBefore: rngBefore,
      rngStateAfter: this.rng.getState(),
      timing: { elapsedMs }
    };
    record.recordHash = computeRecordHash(record);

    this.lineage.push(record);
    this.activeSeqIndex = 0;
    return record;
  }

  getAncestorRecords(seqIndex) {
    const chain = [];
    let cur = seqIndex;
    while (cur !== null && cur !== undefined && cur >= 0) {
      const rec = this.lineage[cur];
      if (!rec) break;
      chain.push(rec);
      cur = rec.parentSeqIndex;
    }
    return chain.reverse();
  }

  stepGeneration(opts = {}) {
    if (this.lineage.length === 0) this.initGen0();
    const t0 = Date.now();
    const parentSeqIndex = this.activeSeqIndex;
    const parentRec = this.lineage[parentSeqIndex];
    const parentGenome = cloneGenome(parentRec.activeGenome);
    const parentWeightsFlat = this.model.exportWeightsFlat();
    const parentBuffers = this.model.exportBuffersFlat();
    const rngBefore = this.rng.getState();

    const operator = opts.forcedOperator || selectOperator(this.rng, this.arm, this.customOperatorWeights);

    const ancestorChain = this.getAncestorRecords(parentSeqIndex);
    const ancestorGenomes = ancestorChain.map(r => r.activeGenome);
    const mutRes = applyChangeOperator(operator, parentGenome, this.rng, ancestorGenomes);
    const candidateGenome = opts.injectedGenome ? cloneGenome(opts.injectedGenome) : mutRes.genome;
    const operatorDetail = opts.injectedGenome ? 'injected_rule' : mutRes.detail;

    const trainSeeds = [];
    let hasNaNOrInf = false;
    let totalStepAbsSum = f32(0);
    let totalGradAbsSum = f32(0);
    let maxStepAbs = f32(0);
    let maxWeightAbs = f32(0);

    for (let s = 0; s < candidateGenome.hyper.steps; s++) {
      const fam = sampleCurriculumFamily(this.rng, candidateGenome.hyper.curriculum);
      const stepSeed = deriveTrainerSeed(this.rng.nextUint32());
      trainSeeds.push({ familyIndex: fam, seed: stepSeed });
      const sample = generateTaskSample(fam, stepSeed, false);
      const st = trainOneStep(this.model, sample, candidateGenome);
      if (st.hasNaNOrInf) hasNaNOrInf = true;
      totalStepAbsSum = f32(totalStepAbsSum + st.stepAbsSum);
      totalGradAbsSum = f32(totalGradAbsSum + st.gradAbsSum);
      if (st.maxStepAbs > maxStepAbs) maxStepAbs = st.maxStepAbs;
      if (st.maxWeightAbs > maxWeightAbs) maxWeightAbs = st.maxWeightAbs;
    }

    let evalRes;
    if (hasNaNOrInf) {
      evalRes = { acc: f32(0), loss: f32(99.0), score: NaN, gen0Score: NaN };
    } else {
      evalRes = evaluateModelOnSamples(this.model, this.devSamples, candidateGenome.scorer);
    }
    const candidateWeightsHash = hasNaNOrInf ? 'invalid_nan_weights' : this.model.weightsHash();

    const gate = evaluateGate({
      isGen0: false,
      hasNaNOrInf,
      totalStepAbsSum,
      totalGradAbsSum,
      maxStepAbs,
      maxWeightAbs,
      candidateDevScore: evalRes.score,
      gen0DevScore: evalRes.gen0Score,
      parentDevScore: parentRec.devScore
    });

    let activeGenome, activeGen, weightsHash;
    if (gate.accepted) {
      activeGenome = cloneGenome(candidateGenome);
      activeGen = parentRec.gen + 1;
      weightsHash = candidateWeightsHash;
    } else {
      this.model.importWeightsFlat(parentWeightsFlat);
      this.model.importBuffersFlat(parentBuffers);
      this.model.ternaryMix = f32(parentGenome.hyper.ternaryMix);
      this.model.ternaryThreshold = f32(parentGenome.hyper.ternaryThreshold);
      activeGenome = cloneGenome(parentGenome);
      activeGen = parentRec.activeGen;
      weightsHash = parentRec.weightsHash;
    }

    const activeWeightsB64 = this.model.weightsToBase64();
    const activeBufs = this.model.exportBuffersFlat();
    const seqIndex = this.lineage.length;
    const isForkRoot = !!this.pendingFork;
    this.pendingFork = false;
    const elapsedMs = Math.max(1, Date.now() - t0);

    const record = {
      seqIndex,
      gen: parentRec.gen + 1,
      branchId: this.activeBranchId,
      isForkRoot,
      parentSeqIndex,
      parentGen: parentRec.gen,
      parentHash: parentRec.recordHash,
      operator,
      operatorDetail,
      genomeDiff: computeGenomeDiff(parentGenome, candidateGenome),
      candidateGenome: cloneGenome(candidateGenome),
      activeGenome: cloneGenome(activeGenome),
      genomeHash: computeGenomeHash(candidateGenome),
      seeds: {
        train: trainSeeds,
        dev: this.devSeedSpecs
      },
      devAcc: Number.isFinite(evalRes.acc) ? evalRes.acc : f32(0),
      devLoss: Number.isFinite(evalRes.loss) ? evalRes.loss : f32(99.0),
      devScore: Number.isFinite(evalRes.score) ? evalRes.score : f32(-999.0),
      gen0DevScore: Number.isFinite(evalRes.gen0Score) ? evalRes.gen0Score : f32(-999.0),
      accepted: gate.accepted,
      gateReason: gate.reason,
      activeGen,
      candidateWeightsHash,
      weightsHash,
      weightsB64: activeWeightsB64,
      mFlatB64: bytesToBase64(new Uint8Array(activeBufs.mFlat.buffer)),
      vFlatB64: bytesToBase64(new Uint8Array(activeBufs.vFlat.buffer)),
      rngStateBefore: rngBefore,
      rngStateAfter: this.rng.getState(),
      timing: { elapsedMs }
    };
    record.recordHash = computeRecordHash(record);

    this.lineage.push(record);
    this.activeSeqIndex = seqIndex;
    return record;
  }

  rewindTo(targetSeqIndex) {
    if (targetSeqIndex < 0 || targetSeqIndex >= this.lineage.length) {
      throw new Error(`Invalid rewind targetSeqIndex: ${targetSeqIndex}`);
    }
    const chain = this.getAncestorRecords(targetSeqIndex);
    if (chain.length === 0 || chain[0].seqIndex !== 0) {
      throw new Error('Broken ancestor chain to gen 0');
    }

    const replayRng = new DeterministicPRNG(this.rootSeed);
    const replayModel = new RWKV7(this.modelConfig, { rng: replayRng });

    const rec0 = chain[0];
    replayRng.setState(rec0.rngStateBefore);
    const gen0Genome = cloneGenome(rec0.candidateGenome);
    for (let s = 0; s < rec0.seeds.train.length; s++) {
      const ts = rec0.seeds.train[s];
      const sample = generateTaskSample(ts.familyIndex, ts.seed, false);
      trainOneStep(replayModel, sample, gen0Genome);
    }

    let activeGenome = cloneGenome(rec0.activeGenome);

    for (let idx = 1; idx < chain.length; idx++) {
      const rec = chain[idx];
      const parentWeights = replayModel.exportWeightsFlat();
      const parentBufs = replayModel.exportBuffersFlat();
      const candGenome = cloneGenome(rec.candidateGenome);

      let hasNaN = false;
      for (let s = 0; s < rec.seeds.train.length; s++) {
        const ts = rec.seeds.train[s];
        const sample = generateTaskSample(ts.familyIndex, ts.seed, false);
        const st = trainOneStep(replayModel, sample, candGenome);
        if (st.hasNaNOrInf) hasNaN = true;
      }

      if (rec.accepted && !hasNaN) {
        activeGenome = cloneGenome(candGenome);
      } else {
        replayModel.importWeightsFlat(parentWeights);
        replayModel.importBuffersFlat(parentBufs);
        replayModel.ternaryMix = f32(activeGenome.hyper.ternaryMix);
        replayModel.ternaryThreshold = f32(activeGenome.hyper.ternaryThreshold);
      }
    }

    const targetRec = chain[chain.length - 1];
    const rebuiltWeightsHash = replayModel.weightsHash();
    const rebuiltGenomeHash = computeGenomeHash(targetRec.candidateGenome);

    this.model = replayModel;
    this.rng.setState(targetRec.rngStateAfter);
    this.activeSeqIndex = targetSeqIndex;
    this.activeBranchId = targetRec.branchId;
    this.pendingFork = false;

    return {
      seqIndex: targetSeqIndex,
      gen: targetRec.gen,
      branchId: targetRec.branchId,
      weightsHash: rebuiltWeightsHash,
      storedWeightsHash: targetRec.weightsHash,
      genomeHash: rebuiltGenomeHash,
      storedGenomeHash: targetRec.genomeHash,
      matches: rebuiltWeightsHash === targetRec.weightsHash && rebuiltGenomeHash === targetRec.genomeHash
    };
  }

  forkFrom(sourceSeqIndex, branchName = null) {
    const rw = this.rewindTo(sourceSeqIndex);
    this.branchCounter++;
    const newBranchId = branchName || `fork-${this.branchCounter}-from-g${rw.gen}`;
    this.activeBranchId = newBranchId;
    this.pendingFork = true;
    const st = this.rng.getState();
    const branchHash = parseInt(sha256Bytes(newBranchId).slice(0, 8), 16) >>> 0;
    this.rng.setState([
      (st[0] ^ branchHash) >>> 0,
      (st[1] + 0x9e3779b9) >>> 0,
      (st[2] ^ (branchHash >>> 1)) >>> 0,
      (st[3] + this.branchCounter) >>> 0
    ]);
    return {
      branchId: newBranchId,
      forkedFromSeqIndex: sourceSeqIndex,
      forkedFromGen: rw.gen,
      parentHash: this.lineage[sourceSeqIndex].recordHash
    };
  }

  exportStateBundle() {
    if (this.lineage.length === 0) this.initGen0();
    const activeRec = this.lineage[this.activeSeqIndex];
    return {
      format: 'GTEE_DESCENDANT_V1',
      rootSeed: this.rootSeed,
      arm: this.arm,
      modelConfig: this.modelConfig,
      activeSeqIndex: this.activeSeqIndex,
      activeBranchId: this.activeBranchId,
      branchCounter: this.branchCounter,
      activeGenome: cloneGenome(activeRec.activeGenome),
      weightsHash: activeRec.weightsHash,
      weightsB64: this.model.weightsToBase64(),
      lineage: this.lineage
    };
  }

  loadStateBundle(bundle) {
    if (!bundle || bundle.format !== 'GTEE_DESCENDANT_V1') {
      throw new Error('Invalid GTEE state bundle format');
    }
    this.rootSeed = bundle.rootSeed >>> 0;
    this.arm = bundle.arm || 'train';
    this.modelConfig = Object.assign({}, DEFAULT_MODEL_CONFIG, bundle.modelConfig || {});
    this.devSeedSpecs = deriveTrainerDevSeeds(this.rootSeed, 4);
    this.devSamples = this.devSeedSpecs.map(sp => generateTaskSample(sp.familyIndex, sp.seed, false));
    this.lineage = bundle.lineage.map(r => Object.assign({}, r));
    this.activeSeqIndex = bundle.activeSeqIndex;
    this.activeBranchId = bundle.activeBranchId || 'main';
    this.branchCounter = bundle.branchCounter || 0;
    this.model = new RWKV7(this.modelConfig, { seed: this.rootSeed });
    this.model.loadWeightsFromBase64(bundle.weightsB64);
    const activeRec = this.lineage[this.activeSeqIndex];
    if (activeRec && activeRec.rngStateAfter) {
      this.rng.setState(activeRec.rngStateAfter);
    }
  }
}

function exportDescendantHTML(templateHTML, stateBundle) {
  const jsonPayload = canonicalJSON(stateBundle);
  const startTag = '<script id="gtee-state" type="application/json">';
  const endTag = '</script>';
  const startIdx = templateHTML.indexOf(startTag);
  if (startIdx === -1) throw new Error('Template HTML is missing <script id="gtee-state" type="application/json">');
  const contentStart = startIdx + startTag.length;
  const endIdx = templateHTML.indexOf(endTag, contentStart);
  if (endIdx === -1) throw new Error('Template HTML is missing closing </script> for #gtee-state');
  return templateHTML.slice(0, contentStart) + jsonPayload + templateHTML.slice(endIdx);
}


function parseArgs(argv) {
  const opts = {
    arm: 'train',
    gens: 3,
    seed: 42,
    out: './out'
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--arm' && argv[i + 1]) opts.arm = argv[++i];
    else if ((a === '--gens' || a === '--generations') && argv[i + 1]) opts.gens = parseInt(argv[++i], 10);
    else if (a === '--seed' && argv[i + 1]) opts.seed = parseInt(argv[++i], 10) >>> 0;
    else if (a === '--out' && argv[i + 1]) opts.out = argv[++i];
  }
  return opts;
}

function findTrainerTemplate() {
  const candidates = [
    path.join(__dirname, 'trainer.html'),
    path.join(process.cwd(), 'trainer.html'),
    '/working_dir/c_3d01dfb1440d0c96/artifacts/file_generation/ttl=63d/output/trainer.html'
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return fs.readFileSync(c, 'utf8');
  }
  throw new Error('Could not locate trainer.html template for descendant export');
}

function runHeadless(opts) {
  const trainer = new GTEETrainer({
    seed: opts.seed,
    arm: opts.arm,
    operatorWeights: opts.operatorWeights || null
  });
  trainer.initGen0();
  for (let g = 0; g < opts.gens; g++) {
    trainer.stepGeneration();
  }
  const bundle = trainer.exportStateBundle();
  const templateHTML = findTrainerTemplate();
  const descendantHTML = exportDescendantHTML(templateHTML, bundle);

  if (opts.out) {
    const resolvedOut = path.resolve(opts.out);
    const isFile = /\.(json|jsonl|html)$/i.test(resolvedOut);
    const outDir = isFile ? path.dirname(resolvedOut) : resolvedOut;
    fs.mkdirSync(outDir, { recursive: true });

    const jsonlLines = trainer.lineage.map(rec => canonicalJSON(rec)).join('\n') + '\n';
    const jsonlPath = path.join(outDir, 'lineage.jsonl');
    const htmlPath = path.join(outDir, 'descendant.html');
    const jsonPath = path.join(outDir, 'lineage.json');

    fs.writeFileSync(jsonlPath, jsonlLines, 'utf8');
    fs.writeFileSync(htmlPath, descendantHTML, 'utf8');
    fs.writeFileSync(jsonPath, canonicalJSON(bundle) + '\n', 'utf8');

    if (isFile) {
      if (resolvedOut.endsWith('.jsonl')) fs.writeFileSync(resolvedOut, jsonlLines, 'utf8');
      else if (resolvedOut.endsWith('.html')) fs.writeFileSync(resolvedOut, descendantHTML, 'utf8');
      else fs.writeFileSync(resolvedOut, canonicalJSON(bundle) + '\n', 'utf8');
    }
  }

  return { trainer, bundle, descendantHTML };
}

if (require.main === module) {
  const opts = parseArgs(process.argv.slice(2));
  const t0 = Date.now();
  const { trainer } = runHeadless(opts);
  const elapsed = Date.now() - t0;
  const last = trainer.lineage[trainer.lineage.length - 1];
  console.log(JSON.stringify({
    status: 'ok',
    arm: opts.arm,
    seed: trainer.rootSeed,
    generationsRun: opts.gens,
    totalRecords: trainer.lineage.length,
    finalDevScore: last.devScore,
    finalGen0DevScore: last.gen0DevScore,
    finalWeightsHash: last.weightsHash,
    elapsedMs: elapsed,
    out: opts.out
  }));
}

module.exports = {
  f32,
  detSqrt,
  detExp,
  detLog,
  detTanh,
  detSigmoid,
  detSin,
  DeterministicPRNG,
  sha256Bytes,
  canonicalJSON,
  bytesToBase64,
  base64ToBytes,
  TMAC,
  BitLinear,
  RWKV7,
  DEFAULT_MODEL_CONFIG,
  TASK_FAMILIES,
  generateTaskSample,
  PRIMITIVE_SPECS,
  ALL_PRIMITIVE_NAMES,
  UNARY_PRIMITIVES,
  BINARY_PRIMITIVES,
  TERNARY_PRIMITIVES,
  UPDATE_RULE_VARS,
  SCORER_VARS,
  AST_MUTATION_OPS,
  validateAST,
  formatAST,
  parseAST,
  evalAST,
  randomExpression,
  mutateAST,
  createGen0UpdateRule,
  createGen0Scorer,
  createGen0Genome,
  computeGenomeHash,
  computeGenomeDiff,
  evaluateModelOnSamples,
  computeRecordHash,
  TRAINER_DOMAIN_MASK,
  deriveTrainerSeed,
  deriveTrainerDevSeeds,
  HYPER_BOUNDS,
  mutateHyperparameters,
  DEFAULT_OPERATOR_WEIGHTS,
  ARM_ENABLED_OPERATORS,
  getArmProbabilities,
  selectOperator,
  applyChangeOperator,
  trainOneStep,
  evaluateGate,
  GTEETrainer,
  exportDescendantHTML,
  runHeadless
};