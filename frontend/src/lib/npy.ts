/**
 * Minimal NumPy .npy file parser for Node.js.
 * Supports: float32 (<f4), little-endian, 2D C-order arrays (version 1.0 + 2.0).
 */

export interface NpyArray {
  data: Float32Array;
  shape: number[];
  dtype: string;
  fortranOrder: boolean;
}

export function parseNpy(buffer: Buffer): NpyArray {
  // ── Magic ──────────────────────────────────────────────────────────────────
  if (
    buffer[0] !== 0x93 ||
    buffer[1] !== 0x4e || // N
    buffer[2] !== 0x55 || // U
    buffer[3] !== 0x4d || // M
    buffer[4] !== 0x50 || // P
    buffer[5] !== 0x59    // Y
  ) {
    throw new Error("Not a valid .npy file: bad magic bytes");
  }

  const majorVersion = buffer[6];

  // ── Header length & data offset ────────────────────────────────────────────
  let headerStart: number;
  let headerLen: number;

  if (majorVersion === 1) {
    headerLen   = buffer.readUInt16LE(8);
    headerStart = 10;
  } else if (majorVersion === 2) {
    headerLen   = buffer.readUInt32LE(8);
    headerStart = 12;
  } else {
    throw new Error(`Unsupported .npy version: ${majorVersion}`);
  }

  const dataOffset = headerStart + headerLen;
  const headerStr  = buffer.slice(headerStart, dataOffset).toString("ascii");

  // ── Parse shape ────────────────────────────────────────────────────────────
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error("Could not parse shape from .npy header");
  const shapeStr = shapeMatch[1].trim();
  const shape    = shapeStr
    ? shapeStr.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n))
    : [];

  // ── Parse dtype ────────────────────────────────────────────────────────────
  const dtypeMatch  = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const dtype       = dtypeMatch ? dtypeMatch[1] : "<f4";

  const fortranMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);
  const fortranOrder = fortranMatch ? fortranMatch[1] === "True" : false;

  if (dtype !== "<f4") {
    throw new Error(`Unsupported dtype: "${dtype}". This parser only handles <f4 (float32 LE).`);
  }

  // ── Build Float32Array ─────────────────────────────────────────────────────
  const rawSlice   = buffer.slice(dataOffset);
  const arrayBuf   = rawSlice.buffer.slice(
    rawSlice.byteOffset,
    rawSlice.byteOffset + rawSlice.byteLength,
  );
  const data = new Float32Array(arrayBuf);

  return { data, shape, dtype, fortranOrder };
}

export interface Bounds {
  south: number;
  north: number;
  west:  number;
  east:  number;
}

export function gridToLatLon(
  row:      number,
  col:      number,
  bounds:   Bounds,
  gridSize = 500,
): { lat: number; lon: number } {
  const lat = bounds.north - (row / gridSize) * (bounds.north - bounds.south);
  const lon = bounds.west  + (col / gridSize) * (bounds.east  - bounds.west);
  return { lat, lon };
}

/** Reservoir sampling with a simple LCG — deterministic, no external RNG needed */
export function reservoirSample<T>(arr: T[], n: number, seed = 42): T[] {
  if (arr.length <= n) return arr;
  let s = seed >>> 0;
  const rand = () => {
    s = Math.imul(s, 1664525) + 1013904223;
    return (s >>> 0) / 0xffffffff;
  };
  const res = arr.slice(0, n);
  for (let i = n; i < arr.length; i++) {
    const j = Math.floor(rand() * (i + 1));
    if (j < n) res[j] = arr[i];
  }
  return res;
}
