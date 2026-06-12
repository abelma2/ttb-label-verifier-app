"use client";

import type { ApplicationData } from "@/lib/types";

interface FieldDef {
  key: keyof ApplicationData;
  label: string;
  placeholder: string;
  advanced?: boolean;
}

const FIELDS: FieldDef[] = [
  { key: "brand_name", label: "Brand name", placeholder: "e.g. Old Tom Reserve" },
  {
    key: "class_type",
    label: "Class / type designation",
    placeholder: "e.g. Kentucky Straight Bourbon Whiskey",
  },
  { key: "alcohol_content", label: "Alcohol content", placeholder: "e.g. 45% Alc./Vol." },
  { key: "net_contents", label: "Net contents", placeholder: "e.g. 750 mL" },
  {
    key: "name_and_address",
    label: "Name & address",
    placeholder: "e.g. Old Tom Distillery, Bardstown, KY",
  },
  {
    key: "country_of_origin",
    label: "Country of origin (imports only)",
    placeholder: "e.g. Scotland — leave blank for domestic",
  },
  {
    key: "fanciful_name",
    label: "Fanciful name",
    placeholder: "e.g. Stormchaser White",
    advanced: true,
  },
  {
    key: "statement_of_composition",
    label: "Statement of composition",
    placeholder: "e.g. Rum with natural flavors added",
    advanced: true,
  },
];

interface ApplicationFormProps {
  values: ApplicationData;
  disabled: boolean;
  onChange: (values: ApplicationData) => void;
}

export default function ApplicationForm({ values, disabled, onChange }: ApplicationFormProps) {
  const renderField = ({ key, label, placeholder }: FieldDef) => (
    <div key={key}>
      <label htmlFor={`app-${key}`} className="mb-1.5 block text-sm font-medium text-slate-700">
        {label}
      </label>
      <input
        id={`app-${key}`}
        type="text"
        value={values[key] ?? ""}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange({ ...values, [key]: e.target.value })}
        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-500 focus:outline focus:outline-2 focus:-outline-offset-1 focus:outline-blue-600 disabled:bg-slate-50 disabled:opacity-60"
      />
    </div>
  );

  return (
    <div>
      <div className="grid gap-4 sm:grid-cols-2">
        {FIELDS.filter((f) => !f.advanced).map(renderField)}
      </div>
      <details className="mt-4 group">
        <summary className="cursor-pointer select-none text-sm font-medium text-blue-700 hover:text-blue-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600">
          More application fields
          <span className="ml-1 text-xs font-normal text-slate-500">
            (fanciful name, statement of composition — accepted as alternate matches for brand
            and class)
          </span>
        </summary>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {FIELDS.filter((f) => f.advanced).map(renderField)}
        </div>
      </details>
    </div>
  );
}
