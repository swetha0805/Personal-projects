/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0d0f12",
          800: "#15181d",
          700: "#1d2127",
          600: "#272c34",
        },
        shell: "#f2f3f5",
        accent: "#c8f169",
        positive: "#e5484d",
        negative: "#30a46c",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      borderRadius: {
        card: "1.75rem",
      },
      boxShadow: {
        card: "0 1px 2px rgba(16,24,40,.04), 0 8px 24px -12px rgba(16,24,40,.10)",
      },
    },
  },
  plugins: [],
};
