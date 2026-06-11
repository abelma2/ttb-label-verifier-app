import type { Status } from "@/lib/types";

const STYLES: Record<Status, { label: string; className: string; icon: React.ReactNode }> = {
  pass: {
    label: "Pass",
    className: "bg-pass-soft text-emerald-800 ring-emerald-600/20",
    icon: (
      <svg viewBox="0 0 16 16" aria-hidden="true" className="h-3.5 w-3.5 fill-emerald-600">
        <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-6.5 6.5a.75.75 0 0 1-1.06 0l-3-3a.75.75 0 1 1 1.06-1.06l2.47 2.47 5.97-5.97a.75.75 0 0 1 1.06 0Z" />
      </svg>
    ),
  },
  needs_review: {
    label: "Needs review",
    className: "bg-review-soft text-amber-800 ring-amber-600/20",
    icon: (
      <svg viewBox="0 0 16 16" aria-hidden="true" className="h-3.5 w-3.5 fill-amber-600">
        <path d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM7.25 4.75a.75.75 0 0 1 1.5 0v3.5a.75.75 0 0 1-1.5 0v-3.5ZM8 12a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z" />
      </svg>
    ),
  },
  fail: {
    label: "Fail",
    className: "bg-fail-soft text-red-800 ring-red-600/20",
    icon: (
      <svg viewBox="0 0 16 16" aria-hidden="true" className="h-3.5 w-3.5 fill-red-600">
        <path d="M5.28 4.22a.75.75 0 0 0-1.06 1.06L6.94 8l-2.72 2.72a.75.75 0 1 0 1.06 1.06L8 9.06l2.72 2.72a.75.75 0 1 0 1.06-1.06L9.06 8l2.72-2.72a.75.75 0 0 0-1.06-1.06L8 6.94 5.28 4.22Z" />
      </svg>
    ),
  },
};

export default function StatusPill({ status }: { status: Status }) {
  const s = STYLES[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${s.className}`}
    >
      {s.icon}
      {s.label}
    </span>
  );
}
