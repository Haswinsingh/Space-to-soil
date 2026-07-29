/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#040814',
          card: 'rgba(15, 23, 42, 0.45)',
          border: 'rgba(255, 255, 255, 0.08)',
          text: '#f8fafc',
          muted: '#94a3b8'
        },
        quantum: {
          cyan: '#00f2fe',
          emerald: '#10b981',
          blue: '#3b82f6',
          violet: '#8b5cf6',
          yellow: '#fbbf24',
          rose: '#f43f5e'
        }
      },
      fontFamily: {
        sans: ['Inter', 'Outfit', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'quantum-glow': '0 0 25px rgba(0, 242, 254, 0.25)',
        'quantum-glow-green': '0 0 25px rgba(16, 185, 129, 0.25)',
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}
