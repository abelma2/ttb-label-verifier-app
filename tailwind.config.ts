import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Status colors used across pills, banners, and borders.
        pass: { DEFAULT: "#059669", soft: "#ecfdf5" }, // emerald-600 / emerald-50
        review: { DEFAULT: "#d97706", soft: "#fffbeb" }, // amber-600 / amber-50
        fail: { DEFAULT: "#dc2626", soft: "#fef2f2" }, // red-600 / red-50
      },
    },
  },
  plugins: [],
};

export default config;
