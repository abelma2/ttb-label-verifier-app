/**
 * Tests for the batch runner (src/lib/batch.ts) with an injected verify
 * function — pinning the behaviors the port's parity depends on: results
 * keyed by INDEX (products can share a stem), one bad product never sinking
 * the batch, cancel rejecting the whole run, and the concurrency cap.
 *
 * Run: npm run test:web
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { BATCH_CONCURRENCY, errorShort, runBatch } from "../batch.ts";
import { VerifyError } from "../api.ts";
import type { Product } from "../stem.ts";
import type { VerifyResponse } from "../types.ts";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function fakeFile(name: string): File {
  return new File([new Uint8Array(8)], name, { type: "image/png" });
}

function product(label: string, ...names: string[]): Product<File> {
  return { label, files: names.map(fakeFile) };
}

function fakeResponse(overall: "pass" | "fail"): VerifyResponse {
  return { mode: "rules_only", overall, beverage_type: "spirits", fields: [],
           additional_statements: [], image_quality_notes: null,
           extracted: {} as VerifyResponse["extracted"] };
}

test("results are keyed by index — two products sharing a stem stay distinct, in order", async () => {
  const seen: string[] = [];
  const items = await runBatch(
    [product("a", "a_front.jpg"), product("a", "a_back.jpg")],
    () => null,
    () => {},
    undefined,
    async (files) => {
      seen.push(files[0].name);
      return fakeResponse("pass");
    },
  );
  assert.equal(items.length, 2);
  assert.deepEqual(items.map((i) => i.fileNames), [["a_front.jpg"], ["a_back.jpg"]]);
  assert.deepEqual(seen.sort(), ["a_back.jpg", "a_front.jpg"]);
});

test("one failing product becomes an error item; the rest of the batch survives", async () => {
  const items = await runBatch(
    [product("bad", "bad.jpg"), product("good", "good.jpg")],
    () => null,
    () => {},
    undefined,
    async (files) => {
      if (files[0].name === "bad.jpg") {
        throw new VerifyError("bad_response", "The vision model returned an unusable response.");
      }
      return fakeResponse("pass");
    },
  );
  assert.equal(items[0].errorKind, "bad_response");
  assert.equal(items[0].result, null);
  assert.equal(errorShort(items[0].errorKind), "bad service reply — try again");
  assert.equal(items[1].errorKind, null);
  assert.equal(items[1].result?.overall, "pass");
});

test("matched reflects whether a non-blank application row was passed", async () => {
  const row = { brand_name: "X", class_type: "", alcohol_content: "", net_contents: "",
                name_and_address: "", country_of_origin: "" };
  const received: (object | null)[] = [];
  const items = await runBatch(
    [product("with", "with.jpg"), product("without", "without.jpg")],
    (label) => (label === "with" ? row : null),
    () => {},
    undefined,
    async (_files, application) => {
      received.push(application);
      return fakeResponse("pass");
    },
  );
  assert.equal(items[0].matched, true);
  assert.equal(items[1].matched, false);
  assert.ok(received.includes(row) && received.includes(null));
});

test("aborting rejects the whole run with kind 'cancelled'", async () => {
  const controller = new AbortController();
  // more products than workers, so workers must loop back to the abort check
  // (in production the in-flight verifyImages calls also reject on abort)
  const run = runBatch(
    Array.from({ length: BATCH_CONCURRENCY + 4 }, (_, i) => product(`p${i}`, `p${i}.jpg`)),
    () => null,
    () => {},
    controller.signal,
    async () => {
      await sleep(20);
      return fakeResponse("pass");
    },
  );
  setTimeout(() => controller.abort(), 5);
  await assert.rejects(run, (err: unknown) =>
    err instanceof VerifyError && err.kind === "cancelled");
});

test("in-flight verifications never exceed BATCH_CONCURRENCY", async () => {
  let inFlight = 0;
  let peak = 0;
  await runBatch(
    Array.from({ length: 12 }, (_, i) => product(`p${i}`, `p${i}.jpg`)),
    () => null,
    () => {},
    undefined,
    async () => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await sleep(5);
      inFlight -= 1;
      return fakeResponse("pass");
    },
  );
  assert.ok(peak <= BATCH_CONCURRENCY, `peak in-flight was ${peak}`);
  assert.ok(peak >= 2, "expected actual concurrency");
});

test("progress reports done/total as items complete", async () => {
  const seen: number[] = [];
  await runBatch(
    [product("a", "a.jpg"), product("b", "b.jpg")],
    () => null,
    (p) => seen.push(p.done),
    undefined,
    async () => fakeResponse("pass"),
  );
  assert.deepEqual(seen.sort(), [1, 2]);
});

test("errorShort: neutral fallback that never photo-blames, proto-chain-safe", () => {
  assert.equal(errorShort("unknown"), "service error — try again");
  assert.equal(errorShort("file_too_large"), "an image is too large");
  // an unmapped kind gets a neutral message, NOT "could not read image"
  assert.equal(errorShort("some_future_kind"), "verification failed — try again");
  assert.equal(errorShort(null), "verification failed — try again");
  // a kind that collides with an Object.prototype member returns a string, not a Function
  assert.equal(errorShort("constructor"), "verification failed — try again");
  assert.equal(errorShort("toString"), "verification failed — try again");
});
