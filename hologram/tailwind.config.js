/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'stark-cyan': '#00E5FF',
        'alert-red': '#FF3D00',
        'neutral-orange': '#FF8C00',
      },
      backgroundImage: {
        'glass': 'rgba(255, 255, 255, 0.25)',
      },
    },
  },
  plugins: [],
}
