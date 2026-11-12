import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/auth': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/send-emergency-email': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/notification-history': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
    },
  },
});
