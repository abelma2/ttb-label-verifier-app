import BatchClient from "@/components/BatchClient";
import Tabs from "@/components/Tabs";
import VerifierClient from "@/components/VerifierClient";

export default function Home() {
  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 sm:px-6">
      <header className="border-b border-slate-200 py-8">
        <p className="text-xs font-semibold uppercase tracking-widest text-blue-700">
          TTB / 27 CFR compliance screening
        </p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
          Alcohol Label Verifier
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
          Upload a label (beer, wine, or distilled spirits) and the verifier reads it with a
          vision model, then deterministically checks it against the federal labeling rules —
          and, if you provide them, against the values from the application. Each field comes
          back as <span className="font-medium text-emerald-700">pass</span>,{" "}
          <span className="font-medium text-amber-700">needs review</span>, or{" "}
          <span className="font-medium text-red-700">fail</span>.
        </p>
      </header>

      <main className="flex-1 py-8">
        <Tabs
          tabs={[
            { id: "single", label: "Single label", content: <VerifierClient /> },
            { id: "batch", label: "Batch screening", content: <BatchClient /> },
          ]}
        />
      </main>

      <footer className="border-t border-slate-200 py-6">
        <p className="text-xs leading-5 text-slate-500">
          A screening aid for label review — not a final legal determination. The vision model
          transcribes what it sees; all compliance judgments are made by deterministic rules
          grounded in the TTB Beverage Alcohol Manuals.
        </p>
      </footer>
    </div>
  );
}
