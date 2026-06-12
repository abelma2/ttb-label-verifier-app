import type { AdditionalStatement, Extraction } from "@/lib/types";

function Kv({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 py-1 text-sm sm:flex-row sm:gap-3">
      <span className="shrink-0 text-slate-500 sm:w-56">{k}</span>
      <span className="text-slate-800">{children}</span>
    </div>
  );
}

function Expander({ summary, children }: { summary: string; children: React.ReactNode }) {
  return (
    <details className="group rounded-xl border border-slate-200 bg-white">
      <summary className="cursor-pointer select-none rounded-xl px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50 group-open:border-b group-open:border-slate-100">
        {summary}
      </summary>
      <div className="px-4 py-3">{children}</div>
    </details>
  );
}

/** Evidence-only extraction fields + additional statements: shown for the
 *  reviewer, never auto-checked. */
const EVIDENCE_FIELDS = [
  "fanciful_name",
  "statement_of_composition",
  "sulfite_declaration",
] as const;

export function OtherLabelDetails({
  extracted,
  additionalStatements,
  imageQualityNotes,
}: {
  extracted: Extraction;
  additionalStatements: AdditionalStatement[];
  imageQualityNotes: string | null;
}) {
  const rows: { k: string; v: string }[] = [];
  for (const name of EVIDENCE_FIELDS) {
    const obj = extracted[name];
    if (obj.present || obj.value) {
      rows.push({
        k: name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()),
        v: `${obj.value || "(present, unreadable)"} (${obj.confidence || "—"} confidence)`,
      });
    }
  }
  for (const s of additionalStatements) {
    rows.push({ k: s.kind ? `Other statement [${s.kind}]` : "Other statement", v: s.value });
  }
  if (imageQualityNotes) {
    rows.push({ k: "Image quality note", v: imageQualityNotes });
  }
  return (
    <Expander summary="Other label details (for reference — not auto-checked)">
      {rows.length > 0 ? (
        rows.map((r, i) => (
          <Kv key={i} k={r.k}>
            {r.v}
          </Kv>
        ))
      ) : (
        <p className="text-sm text-slate-500">
          No evidence-only fields or additional statements were found on this label.
        </p>
      )}
    </Expander>
  );
}
