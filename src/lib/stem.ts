/**
 * Product-stem grouping — a 1:1 port of app.py's `_stem` / `_group_uploads`
 * (the same `_Front`/`_Other` convention as scripts/smoke_test.py). Keep the
 * behavior in lockstep with app.py: these two implementations are the same
 * contract in two languages.
 */

/** Product stem of a filename: extension dropped and a TRAILING side marker
 *  stripped — `_Front`, `-other`, ` back`, `_Label` (optionally followed by a
 *  copy number). Anchored at the end, so words inside a product name (e.g.
 *  `back_forty_ipa`) are never eaten. Shared by upload grouping and
 *  application-data matching. */
export function stem(filename: string): string {
  const dot = filename.lastIndexOf(".");
  const base = dot > 0 ? filename.slice(0, dot) : filename;
  const stripped = base.replace(
    /[ _\-]+(front|other|back|label)([ _\-]*\d+|\s*\(\d+\))?$/i,
    "",
  );
  return stripped || base;
}

export interface Product<T extends { name: string }> {
  label: string;
  files: T[];
}

/** Group files into products. With grouping on, files sharing a name stem are
 *  read together as one product (front + back screened as one label instead of
 *  the front false-failing the warning that lives on the back). With grouping
 *  off, each file is its own product — but the label is still the stem, so
 *  application-data matching keeps working. Upload order is preserved. */
export function groupUploads<T extends { name: string }>(
  files: T[],
  groupPairs: boolean,
): Product<T>[] {
  if (!groupPairs) {
    return files.map((f) => ({ label: stem(f.name), files: [f] }));
  }
  const groups = new Map<string, T[]>();
  for (const f of files) {
    const key = stem(f.name);
    const group = groups.get(key);
    if (group) {
      group.push(f);
    } else {
      groups.set(key, [f]);
    }
  }
  return [...groups.entries()].map(([label, group]) => ({ label, files: group }));
}
