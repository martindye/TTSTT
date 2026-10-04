// One-off: decode a DSH session.jsonl.zstd (concatenated zstd frames, one per
// JSONL batch) using Node's built-in zstd. Mirrors scanZstdFrames from
// packages/session/session-persistence-jsonl/src/zstd.ts.
import { readFileSync, writeFileSync } from 'node:fs'
import { zstdDecompressSync } from 'node:zlib'

const [srcPath, dstPath] = process.argv.slice(2)
const buf = readFileSync(srcPath)

const MAGIC = 0xfd2fb528
const frames = []
let offset = 0
while (offset < buf.length) {
  const start = offset
  if (buf.length - offset < 4) break
  if (buf.readUInt32LE(offset) !== MAGIC) throw new Error(`bad magic at ${offset}`)
  offset += 4
  if (offset >= buf.length) break
  const descriptor = buf.readUInt8(offset)
  offset += 1
  if ((descriptor & 0x18) !== 0) throw new Error('reserved bit set')
  const contentSizeFlag = descriptor >>> 6
  const singleSegment = (descriptor & 0x20) !== 0
  const checksum = (descriptor & 0x04) !== 0
  const dictionaryFlag = descriptor & 0x03
  const dictionaryBytes = dictionaryFlag === 3 ? 4 : dictionaryFlag
  const contentSizeBytes =
    contentSizeFlag === 0 ? (singleSegment ? 1 : 0) : 1 << contentSizeFlag
  offset += (singleSegment ? 0 : 1) + dictionaryBytes + contentSizeBytes
  for (;;) {
    if (buf.length - offset < 3) break
    const blockHeader = buf.readUIntLE(offset, 3)
    offset += 3
    const lastBlock = (blockHeader & 1) !== 0
    const blockType = (blockHeader >>> 1) & 0x03
    const blockSize = blockHeader >>> 3
    if (blockType === 0x03) throw new Error('reserved block type')
    const payloadBytes = blockType === 0x01 ? 1 : blockSize
    if (buf.length - offset < payloadBytes) break
    offset += payloadBytes
    if (lastBlock) break
  }
  if (buf.length - offset < 4) break
  if (checksum) offset += 4
  frames.push({ start, end: offset })
}

let out = []
let bytes = 0
for (const f of frames) {
  const piece = zstdDecompressSync(buf.subarray(f.start, f.end))
  out.push(piece)
  bytes += piece.length
}
writeFileSync(dstPath, Buffer.concat(out))
console.log(`frames=${frames.length} plaintext=${bytes} bytes -> ${dstPath}`)
