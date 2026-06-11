// Tailwind v3 (pure-JS) on purpose: v4's native engine (lightningcss/oxide)
// ships no 32-bit Windows binaries, and the local dev machine runs 32-bit
// Node. v3 has zero native dependencies and identical output for this UI.
const config = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};

export default config;
