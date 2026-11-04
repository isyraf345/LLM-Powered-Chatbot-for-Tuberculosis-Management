import { defineConfig } from '@playwright/test'

const port = Number(process.env.TEST_FRONTEND_PORT ?? 5173)

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  workers: 1,
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium', screenshot: 'only-on-failure' },
  webServer: { command: `npm run dev -- --port ${port}`, url: `http://127.0.0.1:${port}`, reuseExistingServer: false },
})
