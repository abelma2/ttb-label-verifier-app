/**
 * Unit tests for the batch helpers (stem, groupUploads, parseApplicationsFile,
 * appRowFor, pickApplicationRow) — the row-matching semantics are TypeScript
 * ports of the retired Streamlit prototype's helpers (dev-archive branch);
 * these tests pin the documented behavior case for case. The application file
 * is Excel-only, so the parse tests build real .xlsx workbooks in memory.
 *
 * Run: npm run test:web   (node --experimental-strip-types --test)
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { stem, groupUploads } from "../stem.ts";
import {
  appRowFor,
  buildApplicationsTemplate,
  normHeader,
  parseApplicationsFile,
  pickApplicationRow,
  type AppRow,
} from "../applications.ts";

// SheetJS is CJS: under plain Node the exports may only be on `.default`.
import * as XLSXmod from "xlsx";
const XLSX = (
  (XLSXmod as { default?: unknown }).default ?? XLSXmod
) as typeof import("xlsx");

/** Build an in-memory workbook File from [sheetName, rows[][]] pairs. */
function wbFile(
  sheets: [string, unknown[][]][],
  filename = "apps.xlsx",
  patch?: (wb: import("xlsx").WorkBook) => void,
): File {
  const wb = XLSX.utils.book_new();
  for (const [name, aoa] of sheets) {
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(aoa), name);
  }
  patch?.(wb);
  const bytes = XLSX.write(wb, { type: "array", bookType: "xlsx" }) as ArrayBuffer;
  return new File([bytes], filename);
}

/** Single-sheet shorthand. */
function sheetFile(aoa: unknown[][], filename = "apps.xlsx"): File {
  return wbFile([["Sheet1", aoa]], filename);
}

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

test("a stitched single image matches the same product as a front/back pair", () => {
  // the application row 'oldtom' must match all three upload styles
  assert.equal(stem("oldtom.jpg"), "oldtom");
  assert.equal(stem("oldtom_front.jpg"), stem("oldtom.jpg"));
  assert.equal(stem("oldtom_label.png"), stem("oldtom.jpg"));
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

// --- header normalization ----------------------------------------------------------

test("normHeader normalizes spacing, case, and dashes", () => {
  assert.equal(normHeader(" Brand Name "), "brand_name");
  assert.equal(normHeader("brand-name"), "brand_name");
  assert.equal(normHeader("NET   CONTENTS"), "net_contents");
});

// --- parseApplicationsFile: happy paths --------------------------------------------

test("Excel: normalized headers match, values trimmed, keyed by lowercase product", async () => {
  const file = sheetFile([
    ["Product", "Brand Name", "Net Contents"],
    ["OldTom", " Old Tom Reserve ", " 750 mL "],
  ]);
  const { mapping, error, warnings } = await parseApplicationsFile(file);
  assert.equal(error, null);
  assert.deepEqual(warnings, []);
  assert.equal(mapping.oldtom.brand_name, "Old Tom Reserve");
  assert.equal(mapping.oldtom.net_contents, "750 mL");
  assert.equal(mapping.oldtom.class_type, ""); // all six keys always present
});

test("Excel: numeric and percent-formatted cells arrive as the text Excel shows", async () => {
  const file = wbFile([
    [
      "Apps",
      [
        ["product", "alcohol_content", "net_contents"],
        ["a", 0.45, 750],
      ],
    ],
  ], "apps.xlsx", (wb) => {
    wb.Sheets.Apps.B2.z = "0%"; // a percent-formatted ABV cell must read "45%", not "0.45"
  });
  const { mapping } = await parseApplicationsFile(file);
  assert.equal(mapping.a.alcohol_content, "45%");
  assert.equal(mapping.a.net_contents, "750");
});

test("Excel: duplicate products are last-row-wins and reported", async () => {
  const file = sheetFile([
    ["product", "brand_name"],
    ["a", "First"],
    ["a", "Second"],
  ]);
  const { mapping, warnings } = await parseApplicationsFile(file);
  assert.equal(mapping.a.brand_name, "Second");
  assert.ok(warnings.some((w) => w.includes("duplicate product row(s): a")));
});

test("Excel: no recognized field columns warns (rules-only screening)", async () => {
  const file = sheetFile([
    ["product", "notes"],
    ["a", "hello"],
  ]);
  const { mapping, error, warnings } = await parseApplicationsFile(file);
  assert.equal(error, null);
  assert.ok(mapping.a);
  assert.ok(warnings.some((w) => w.includes("no recognized application-field columns")));
});

test("Excel: rows without a product value are skipped; none at all is an error", async () => {
  const file = sheetFile([
    ["product", "brand_name"],
    ["", "NoProduct"],
  ]);
  const { error } = await parseApplicationsFile(file);
  assert.equal(error, "no rows with a 'product' value were found");
});

// --- parseApplicationsFile: sheet selection -----------------------------------------

test("the first sheet WITH a 'product' column is read — instruction sheets are skipped", async () => {
  const file = wbFile([
    ["How to fill this in", [["Put your data on the next sheet"]]],
    ["Applications", [["product", "brand_name"], ["a", "X"]]],
  ]);
  const { mapping, error, warnings } = await parseApplicationsFile(file);
  assert.equal(error, null);
  assert.deepEqual(warnings, []);
  assert.equal(mapping.a.brand_name, "X");
});

test("a header-only decoy sheet does not shadow the real data sheet", async () => {
  // e.g. a cleared/duplicated template sheet left in front of the real one
  const file = wbFile([
    ["Archive", [["product", "brand_name"]]],
    ["Applications", [["product", "brand_name"], ["a", "X"]]],
  ]);
  const { mapping, error, warnings } = await parseApplicationsFile(file);
  assert.equal(error, null);
  assert.equal(mapping.a.brand_name, "X");
  assert.ok(warnings.some((w) => w.includes("more than one sheet") && w.includes("'Applications'")));
});

test("two data-bearing sheets: first wins, with a warning naming it", async () => {
  const file = wbFile([
    ["First", [["product", "brand_name"], ["a", "X"]]],
    ["Second", [["product", "brand_name"], ["b", "Y"]]],
  ]);
  const { mapping, warnings } = await parseApplicationsFile(file);
  assert.ok(mapping.a);
  assert.equal(mapping.b, undefined);
  assert.ok(warnings.some((w) => w.includes("more than one sheet") && w.includes("'First'")));
});

test("no sheet with a 'product' column is a clear error", async () => {
  const file = sheetFile([
    ["name", "brand_name"],
    ["a", "X"],
  ]);
  const { error } = await parseApplicationsFile(file);
  assert.ok(error?.includes("no sheet has a 'product' column"));
});

// --- parseApplicationsFile: rejected inputs -----------------------------------------

test("non-Excel extensions (the retired CSV/JSON formats) get the Excel-only error", async () => {
  for (const name of ["apps.csv", "apps.json", "apps.txt"]) {
    const { error } = await parseApplicationsFile(new File(["product,brand\na,X"], name));
    assert.ok(error?.includes("only Excel files are accepted"), `${name}: ${error}`);
  }
});

test("a corrupt .xlsx reports a read error, not a crash", async () => {
  // a broken ZIP container (the real "corrupt download" case)
  const corrupt = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 9, 9, 9, 9, 9, 9]);
  const { error } = await parseApplicationsFile(new File([corrupt], "apps.xlsx"));
  assert.ok(error?.includes("could not be read as an Excel workbook"));
  // SheetJS reads plain text renamed to .xlsx as a one-column sheet — that must
  // surface the layout error, never a crash or a silent empty mapping
  const renamed = await parseApplicationsFile(
    new File(["just some prose, no table here"], "apps.xlsx"),
  );
  assert.ok(renamed.error?.includes("no sheet has a 'product' column"));
});

test("an oversized file is rejected before parsing", async () => {
  const big = new File([new Uint8Array(10 * 1024 * 1024 + 1)], "big.xlsx");
  const { error } = await parseApplicationsFile(big);
  assert.ok(error?.includes("over 10 MB"));
});

// --- appRowFor / pickApplicationRow ----------------------------------------------------

test("appRowFor: a present-but-blank row screens rules-only (null)", async () => {
  const { mapping } = await parseApplicationsFile(
    sheetFile([["product", "brand_name"], ["a", ""]]),
  );
  assert.equal(appRowFor(mapping, "a"), null);
  assert.equal(appRowFor(mapping, "missing"), null);
});

test("appRowFor: matching ignores capitalization", async () => {
  const { mapping } = await parseApplicationsFile(
    sheetFile([["product", "brand_name"], ["OldTom", "X"]]),
  );
  assert.ok(appRowFor(mapping, "OLDTOM"));
});

// --- review-fix regressions -----------------------------------------------------

test("products named after Object.prototype members don't hit the prototype chain", async () => {
  // no application file at all: 'constructor' must NOT read as a present row
  assert.equal(appRowFor({}, "constructor"), null);
  assert.equal(pickApplicationRow({}, "constructor").row, null);
  // first row named 'Constructor' must not be falsely reported as a duplicate
  const { mapping, warnings } = await parseApplicationsFile(
    sheetFile([["product", "brand_name"], ["Constructor", "X"]]),
  );
  assert.equal(warnings.length, 0);
  assert.ok(appRowFor(mapping, "constructor"));
});

test("a product named __proto__ is stored as a real row, not prototype pollution", async () => {
  const { mapping } = await parseApplicationsFile(
    sheetFile([
      ["product", "brand_name"],
      ["__proto__", "Evil"],
      ["oldtom", "Old Tom"],
    ]),
  );
  // both rows present and counted; the __proto__ row is a real, matchable product
  assert.deepEqual(Object.keys(mapping).sort(), ["__proto__", "oldtom"]);
  assert.equal(appRowFor(mapping, "__proto__")?.brand_name, "Evil");
  // and no prototype was polluted: a normal key inherits nothing
  assert.equal(appRowFor(mapping, "oldtom")?.brand_name, "Old Tom");
});

test("pickApplicationRow: stem match, single-row fallback, ambiguity -> null", async () => {
  const { mapping } = await parseApplicationsFile(
    sheetFile([["product", "brand_name"], ["a", "X"], ["b", "Y"]]),
  );
  assert.equal(pickApplicationRow(mapping, "A").row?.brand_name, "X");
  assert.equal(pickApplicationRow(mapping, "zzz").row, null);
  const single = (
    await parseApplicationsFile(sheetFile([["product", "brand_name"], ["only", "Z"]]))
  ).mapping;
  assert.equal(pickApplicationRow(single, "nomatch").row?.brand_name, "Z");
});

// --- the downloadable template ------------------------------------------------------

test("the template round-trips through the parser: examples load, no warnings", async () => {
  const bytes = await buildApplicationsTemplate();
  const { mapping, error, warnings } = await parseApplicationsFile(
    new File([bytes], "ttb-application-template.xlsx"),
  );
  assert.equal(error, null);
  assert.deepEqual(warnings, []); // the instructions sheet must not trip the multi-sheet warning
  assert.deepEqual(Object.keys(mapping).sort(), ["oldtom", "riverbend"]);
  assert.equal(mapping.oldtom.brand_name, "OLD TOM RESERVE");
  assert.equal(mapping.riverbend.country_of_origin, "Product of France");
  // every verifier column is present and filled in at least one example row
  const rows: AppRow[] = Object.values(mapping);
  for (const fld of Object.keys(rows[0]) as (keyof AppRow)[]) {
    assert.ok(rows.some((r) => r[fld].trim() !== ""), `template never fills ${fld}`);
  }
});
