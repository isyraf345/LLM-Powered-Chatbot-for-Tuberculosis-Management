import { test, expect } from '@playwright/test'

const answer = {
  answer: 'This is a **test answer**.', retriever: 'single_nq', generation_model: 'test-model',
  request_id: 'request-123', trace_id: 'trace-123',
  passages: [{ text: 'A test passage.', source: 'Test reference.pdf', page: 4, score: 0.72 }],
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/retrievers', route => route.fulfill({ json: {
    default: 'single_nq', retrievers: [{ id: 'single_nq', name: 'Single NQ' }, { id: 'multiset', name: 'Multiset' }],
  } }))
})

test('sends selected retriever, shows answer, sources and trace; resets the session', async ({ page }, testInfo) => {
  let body: unknown
  await page.route('**/api/chat', async route => {
    body = route.request().postDataJSON()
    await route.fulfill({ json: { ...answer, retriever: 'multiset' } })
  })
  await page.goto('/')
  await page.getByLabel('Retrieval model').selectOption('multiset')
  await page.screenshot({ path: testInfo.outputPath('desktop.png'), fullPage: true })
  await page.getByLabel('Your question about tuberculosis').fill('What is TB?')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.getByText('This is a', { exact: false })).toBeVisible()
  expect(body).toEqual({ question: 'What is TB?', retriever: 'multiset' })
  await page.locator('.sources summary').click()
  await expect(page.getByText('A test passage.')).toBeVisible()
  await page.getByText('Request details', { exact: true }).click()
  await expect(page.getByText('trace-123', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'New conversation' }).click()
  await expect(page.getByRole('heading', { name: /Understand TB/ })).toBeVisible()
  await expect(page.getByText('A test passage.')).toHaveCount(0)
})

test('preserves the failed question and retries without duplicating it', async ({ page }) => {
  let calls = 0
  await page.route('**/api/chat', route => {
    calls++
    return calls === 1
      ? route.fulfill({ status: 503, headers: { 'X-Request-ID': 'failed-request' }, json: { detail: 'busy' } })
      : route.fulfill({ json: answer })
  })
  await page.goto('/')
  await page.getByLabel('Your question about tuberculosis').fill('How does TB spread?')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.getByRole('alert')).toContainText('busy or unavailable')
  await expect(page.getByRole('alert')).toContainText('failed-request')
  await page.getByRole('button', { name: 'Retry question' }).click()
  await expect(page.locator('.answer')).toContainText('test answer')
  await expect(page.locator('.question')).toHaveCount(1)
  expect(calls).toBe(2)
})

test('prevents duplicate submits while loading and blocks blank questions', async ({ page }) => {
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  let calls = 0
  await page.route('**/api/chat', async route => { calls++; await gate; await route.fulfill({ json: answer }) })
  await page.goto('/')
  const input = page.getByLabel('Your question about tuberculosis')
  await input.fill('   ')
  await expect(page.getByRole('button', { name: 'Send question' })).toBeDisabled()
  await input.fill('What is TB?')
  await input.press('Enter')
  await expect(page.getByRole('status')).toContainText('Finding passages')
  await input.fill('Second question')
  await input.press('Enter')
  await expect(page.getByRole('button', { name: 'Working' })).toBeDisabled()
  await expect(page.getByLabel('Retrieval model')).toBeDisabled()
  release()
  await expect(page.locator('.answer')).toBeVisible()
  expect(calls).toBe(1)
})

test('API connection failure can be recovered', async ({ page }) => {
  await page.route('**/api/retrievers', route => route.abort())
  await page.goto('/')
  await expect(page.getByRole('alert')).toContainText('Cannot reach the API')
  await page.unroute('**/api/retrievers')
  await page.route('**/api/retrievers', route => route.fulfill({ json: { default: 'single_nq', retrievers: [{ id: 'single_nq', name: 'Single NQ' }] } }))
  await page.getByRole('button', { name: 'Reconnect' }).click()
  await expect(page.getByLabel('Retrieval model')).toBeEnabled()
})

test('mobile layout fits viewport and model HTML cannot execute', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.route('**/api/chat', route => route.fulfill({ json: { ...answer, answer: 'Safe text <img src="bad" onerror="alert(1)">\n\n[bad link](javascript:alert(1))' } }))
  await page.goto('/')
  await page.getByLabel('Your question about tuberculosis').fill('What is TB?')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.locator('.answer')).toContainText('Safe text')
  await expect(page.locator('.answer img')).toHaveCount(0)
  await expect(page.locator('.answer a[href^="javascript:"]')).toHaveCount(0)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('mobile.png'), fullPage: true })
})
