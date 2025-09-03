#!/usr/bin/env node
import {parseSrt} from '@remotion/captions';
import fs from 'node:fs/promises';

const [,, srtPath, outJsonPath] = process.argv;
if (!srtPath || !outJsonPath) {
  console.error('Usage: node convert-srt-to-json.mjs <input.srt> <output.json>');
  process.exit(1);
}

const srt = await fs.readFile(srtPath, 'utf8');
const {captions} = parseSrt({input: srt});
await fs.writeFile(outJsonPath, JSON.stringify(captions));
console.log(`Wrote ${captions.length} captions to ${outJsonPath}`);
