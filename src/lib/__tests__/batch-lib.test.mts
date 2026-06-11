/**
 * Unit tests for the TypeScript ports of app.py's batch helpers (_stem,
 * _group_uploads, _parse_applications, _app_row_for, _pick_application_row).
 * The two implementations are one contract in two languages — these tests
 * mirror the documented app.py behavior case for case.
 *
 * Run: npm run test:web   (node --experimental-strip-types --test; no deps)
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { stem, groupUploads } from "../stem.ts";
import {
  appRowFor,
  normHeader,
  parseApplications,
  parseCsv,
  pickApplicationRow,
} from "../applications.ts";

// --- stem ---------------------------------------------------------------------

test("stem strips trailing side markers, any separator, any case", () => {
  assert.equal(stem("oldtom_front.jpg"), "oldtom");
  assert.equal(stem("oldtom_Back.png"), "oldtom");
  assert.equal(stem("oldtom-other.jpeg"), "oldtom");
  assert.equal(stem("oldtom label.png"), "oldtom");
  assert.equal(stem("baseline_1_Front.png"), "baseline_1");
});

test("stem strips a copy number after the marker", () => {
  assert.equal(stem("brandX_front_2.jpg"), "brandX");
  assert.equal(stem("brandX_front (2).jpg"), "brandX");
  assert.equal(stem("brandX_label 3.png"), "brandX");
});

test("stem never eats marker words inside the product name", () => {
  assert.equal(stem("back_forty_ipa.png"), "back_forty_ipa");
  assert.equal(stem("frontier_whiskey.jpg"), "frontier_whiskey");
});

test("stem falls back to the unstripped stem rather than empty", () => {
  assert.equal(stem("front.jpg"), "front");
  assert.equal(stem("_front.jpg"), "_front");
});

test("stem keeps camera-style names distinct", () => {
  assert.equal(stem("IMG_1234.jpg"), "IMG_1234");
});

// --- groupUploads ----------------------------------------------------------------

const f = (name: string) => ({ name });

test("grouping on: same stem reads as one product, order preserved", () => {
  const products = groupUploads(
    [f("b_front.jpg"), f("a_front.jpg"), f("b_back.jpg")],
    true,
  );
  assert.deepEqual(
    products.map((p) => [p.label, p.files.map((x) => x.name)]),
    [
      ["b", ["b_front.jpg", "b_back.jpg"]],
      ["a", ["a_front.jpg"]],
    ],
  );
});

test("grouping off: each file its own product, label is still the stem", () => {
  const products = groupUploads([f("a_front.jpg"), f("a_back.jpg")], false);
  assert.deepEqual(
    products.map((p) => [p.label, p.files.length]),
    [
      ["a", 1],
      ["a", 1],
    ],
  );
});

// --- CSV parsing -------------------------------------------------------------------

test("parseCsv handles quoted commas, escaped quotes, CRLF, blank lines", () => {
  const grid = parseCsv('a,b\r\n"x, y","he said ""hi"""\n\n,\nlast,row\n');
  assert.deepEqual(grid, [
    ["a", "b"],
    ["x, y", 'he said "hi"'],
    ["", ""], // ",": two empty cells is NOT a blank line
    ["last", "row"],
  ]);
});

test("normHeader normalizes spacing, case, and dashes", () => {
  assert.equal(normHeader(" Brand Name "), "brand_name");
  assert.equal(normHeader("brand-name"), "brand_name");
  assert.equal(normHeader("NET   CONTENTS"), "net_contents");
});

// --- parseApplications: CSV ----------------------------------------------------------

test("CSV: normalized headers match, values trimmed, keyed by lowercase product", () => {
  const csv = "Product,Brand Name,Net Contents\nOldTom, Old Tom Reserve , 750 mL \n";
  const { mapping, error, warnings } = parseApplications(csv, "apps.csv");
  assert.equal(error, null);
  assert.deepEqual(warnings, []);
  assert.equal(mapping.oldtom.brand_name, "Old Tom Reserve");
  assert.equal(mapping.oldtom.net_contents, "750 mL");
  assert.equal(mapping.oldtom.class_type, ""); // all six keys always present
});

test("CSV: duplicate products are last-row-wins and reported", () => {
  const csv = "product,brand_name\na,First\na,Second\n";
  const { mapping, warnings } = parseApplications(csv, "apps.csv");
  assert.equal(mapping.a.brand_name, "Second");
  assert.ok(warnings.some((w) => w.includes("duplicate product row(s): a")));
});

test("CSV: no recognized field columns warns (rules-only screening)", () => {
  const csv = "product,notes\na,hello\n";
  const { mapping, error, warnings } = parseApplications(csv, "apps.csv");
  assert.equal(error, null);
  assert.ok(mapping.a);
  assert.ok(warnings.some((w) => w.includes("no recognized application-field columns")));
});

test("CSV: rows without a product value are skipped; none at all is an error", () => {
  const { error } = parseApplications("product,brand_name\n,NoProduct\n", "apps.csv");
  assert.equal(error, "no rows with a 'product' value were found");
});

test("CSV: a UTF-8 BOM on the header row is tolerated", () => {
  const { mapping } = parseApplications("﻿product,brand_name\na,X\n", "apps.csv");
  assert.equal(mapping.a.brand_name, "X");
});

// --- parseApplications: JSON --------------------------------------------------------

test("JSON mapping form: the mapping key beats an inner 'Product' field", () => {
  const json = JSON.stringify({ realprod: { Product: "decoy", "Brand Name": "X" } });
  const { mapping } = parseApplications(json, "apps.json");
  assert.ok(mapping.realprod);
  assert.equal(mapping.decoy, undefined);
  assert.equal(mapping.realprod.brand_name, "X");
});

test("JSON list and {products: [...]} forms both work", () => {
  const list = JSON.stringify([{ product: "a", brand_name: "X" }]);
  assert.equal(parseApplications(list, "apps.json").mapping.a.brand_name, "X");
  const wrapped = JSON.stringify({ products: [{ product: "b", brand_name: "Y" }] });
  assert.equal(parseApplications(wrapped, "apps.json").mapping.b.brand_name, "Y");
});

test("JSON: a scalar document is a clear error, not a crash", () => {
  const { error } = parseApplications('"just a string"', "apps.json");
  assert.equal(error, "JSON must be a mapping of product -> fields, or a list of objects");
});

test("JSON: invalid JSON reports the friendly read error", () => {
  const { error } = parseApplications("{not json", "apps.json");
  assert.equal(error, "the JSON file could not be read — it isn't valid JSON");
});

// --- appRowFor / pickApplicationRow ----------------------------------------------------

test("appRowFor: a present-but-blank row screens rules-only (null)", () => {
  const { mapping } = parseApplications("product,brand_name\na,\n", "apps.csv");
  assert.equal(appRowFor(mapping, "a"), null);
  assert.equal(appRowFor(mapping, "missing"), null);
});

test("appRowFor: matching ignores capitalization", () => {
  const { mapping } = parseApplications("product,brand_name\nOldTom,X\n", "apps.csv");
  assert.ok(appRowFor(mapping, "OLDTOM"));
});

// --- review-fix regressions -----------------------------------------------------

test("products named after Object.prototype members don't hit the prototype chain", () => {
  // no application file at all: 'constructor' must NOT read as a present row
  assert.equal(appRowFor({}, "constructor"), null);
  assert.equal(pickApplicationRow({}, "constructor").row, null);
  // first row named 'Constructor' must not be falsely reported as a duplicate
  const { mapping, warnings } = parseApplications(
    "product,brand_name\nConstructor,X\n", "apps.csv");
  assert.equal(warnings.length, 0);
  assert.ok(appRowFor(mapping, "constructor"));
});

test("a product named __proto__ is stored as a real row, not prototype pollution", () => {
  const { mapping } = parseApplications(
    "product,brand_name\n__proto__,Evil\noldtom,Old Tom\n", "apps.csv");
  // both rows present and counted; the __proto__ row is a real, matchable product
  assert.deepEqual(Object.keys(mapping).sort(), ["__proto__", "oldtom"]);
  assert.equal(appRowFor(mapping, "__proto__")?.brand_name, "Evil");
  // and no prototype was polluted: a normal key inherits nothing
  assert.equal(appRowFor(mapping, "oldtom")?.brand_name, "Old Tom");
});

test("mid-field quotes are literal, matching Python's csv reader", () => {
  const grid = parseCsv('product,brand_name\na,BRAND "X" WHISKEY\n');
  assert.equal(grid[1][1], 'BRAND "X" WHISKEY');
  // an inch-mark must not flip the parser into quoted mode for the rest of the file
  const grid2 = parseCsv('product,net_contents\na,12" tall\nb,750 mL\n');
  assert.deepEqual(grid2[2], ["b", "750 mL"]);
});

test("falsy JSON values collapse like Python's `or` (0/false are blank, not '0'/'false')", () => {
  const json = JSON.stringify([
    { product: 0, brand_name: "skipped — blank product" },
    { product: "a", alcohol_content: 0, brand_name: false },
  ]);
  const { mapping, error } = parseApplications(json, "apps.json");
  assert.equal(error, null);
  assert.equal(Object.keys(mapping).length, 1);
  assert.equal(mapping.a.alcohol_content, "");
  assert.equal(mapping.a.brand_name, "");
  assert.equal(appRowFor(mapping, "a"), null); // all-blank row -> rules-only
});

test("binary content gets the save-as-CSV guidance, not a misleading 'no rows' error", () => {
  const { error } = parseApplications("PK\u0003\u0004\u0000binary junk", "apps.csv");
  assert.equal(error, "the file isn't plain text — save it as a regular CSV or JSON file");
});

test("pickApplicationRow: stem match, single-row fallback, ambiguity -> null", () => {
  const { mapping } = parseApplications(
    "product,brand_name\na,X\nb,Y\n",
    "apps.csv",
  );
  assert.equal(pickApplicationRow(mapping, "A").row?.brand_name, "X");
  assert.equal(pickApplicationRow(mapping, "zzz").row, null);
  const single = parseApplications("product,brand_name\nonly,Z\n", "apps.csv").mapping;
  assert.equal(pickApplicationRow(single, "nomatch").row?.brand_name, "Z");
});
