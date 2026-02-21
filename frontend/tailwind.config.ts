import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg:       "#080812",
        surface:  "#0e0e1c",
        card:     "#13132a",
        border:   "#1c1c38",
        accent:   "#00d4ff",
        success:  "#00e676",
        warning:  "#ffc107",
        danger:   "#ff1744",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "fade-in":  "fadeIn 0.4s ease-out",
        "slide-up": "slideUp 0.35s ease-out",
        "pulse-dot": "pulseDot 2s ease-in-out infinite",
      },
      keyframes: {
        fadeIn:   { from: { opacity: "0" },                      to: { opacity: "1" } },
        slideUp:  { from: { opacity: "0", transform: "translateY(8px)" }, to: { opacity: "1", transform: "translateY(0)" } },
        pulseDot: { "0%, 100%": { opacity: "1" },                "50%":        { opacity: "0.3" } },
      },
    },
  },
  plugins: [],
};

export default config;
