import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/index.css'
import App from './app/App.tsx'

// Suppress React warnings from UI library internals
const originalWarn = console.warn
console.warn = (...args) => {
  // Suppress known harmless React warnings from shadcn/ui and motion/react
  const message = args.join(' ')
  if (message.includes('Warning: Function components cannot be given refs') ||
      message.includes('Warning: React does not recognize the')) {
    return // Suppress these warnings
  }
  originalWarn.apply(console, args)
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);