import path from 'path';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Viteの設定を定義
export default defineConfig({
    // Reactプラグインを有効化
    plugins: [react()],

    // パスエイリアスの設定
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },

    // ビルド設定
    build: {
        outDir: 'dist',
        sourcemap: true,
        minify: 'terser',
        target: 'es2018',
        rollupOptions: {
            output: {
                manualChunks: {
                    'react-vendor': ['react', 'react-dom'],
                    'ui-components': [
                        '@radix-ui/react-accordion',
                        '@radix-ui/react-dialog',
                        '@radix-ui/react-dropdown-menu',
                        '@radix-ui/react-select',
                        '@radix-ui/react-tabs'
                    ],
                    'form-handling': [
                        'react-hook-form',
                        'zod',
                        '@hookform/resolvers'
                    ],
                    'data-visualization': [
                        'chart.js',
                        'react-chartjs-2',
                        'recharts'
                    ],
                    'utilities': [
                        'date-fns',
                        'axios',
                        'clsx',
                        'tailwind-merge'
                    ]
                }
            }
        },
        chunkSizeWarningLimit: 800,
        assetsInlineLimit: 4096,
        cssCodeSplit: true,
        reportCompressedSize: true,
    },

    // 開発サーバーの設定
    server: {
        port: 3000,
        host: true,
        hmr: {
            overlay: true
        },
        cors: true
    },

    // 依存関係の最適化設定
    optimizeDeps: {
        exclude: ['lucide-react'],
        include: [
            'react',
            'react-dom',
            'react-router-dom'
        ]
    },

    // プレビューサーバーの設定
    preview: {
        port: 3000,
        host: true
    }
});