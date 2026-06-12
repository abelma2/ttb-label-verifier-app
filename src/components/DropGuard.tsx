"use client";

import { useEffect } from "react";

/** Window-level guard against missed drags: a file dropped anywhere no element
 *  claims would otherwise trigger the browser default — navigating the tab to
 *  the dropped file, unloading the app and losing every upload, typed value,
 *  and result in memory. The dropzones' own handlers run first (they're inner
 *  targets), so this only swallows drops that landed off-target. */
export default function DropGuard() {
  useEffect(() => {
    // dragover must always be cancelled, or the browser never delivers drop
    const allowDrop = (e: DragEvent) => e.preventDefault();
    const swallowDrop = (e: DragEvent) => {
      // text dragged into a form field keeps its native insert-on-drop (the
      // insert is the drop's default action, so preventDefault here would
      // silently kill it) — but a FILE dropped on a field would still
      // navigate, so file drags are swallowed even on editable targets
      const t = e.target;
      const editable =
        t instanceof HTMLInputElement ||
        t instanceof HTMLTextAreaElement ||
        (t instanceof HTMLElement && t.isContentEditable);
      if (editable && !e.dataTransfer?.types.includes("Files")) return;
      e.preventDefault();
    };
    window.addEventListener("dragover", allowDrop);
    window.addEventListener("drop", swallowDrop);
    return () => {
      window.removeEventListener("dragover", allowDrop);
      window.removeEventListener("drop", swallowDrop);
    };
  }, []);
  return null;
}
