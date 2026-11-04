import { test, expect } from '@playwright/test'
import type { Route } from '@playwright/test'

const answer = {
  answer: 'This is a **test answer**.', retriever: 'single_nq', generation_model: 'test-model',
  request_id: 'request-123', trace_id: 'trace-123',
  passages: [{ text: 'A test passage.', source: 'Test reference.pdf', page: 4, score: 0.72 }],
}

function saved(route: Route, result = answer) {
  const body = route.request().postDataJSON()
  return { id: new URL(route.request().url()).pathname.split('/').at(-1), ...body,
    status: 'completed', answer: result, error: null, request_id: 'request-123', feedback: null }
}

test.beforeEach(async ({ page }) => {
  const conversations = new Map<string, { id: string; title: string; created_at: string; updated_at: string; messages: ReturnType<typeof saved>[] }>()
  await page.route('**/api/session', route => route.fulfill({ json: { expires_at: '2026-12-05T00:00:00Z' } }))
  await page.route('**/api/conversations**', async route => {
    const parts = new URL(route.request().url()).pathname.split('/').filter(Boolean)
    const id = parts[2]
    const method = route.request().method()
    if (!id) {
      if (method === 'DELETE') conversations.clear()
      return route.fulfill({ json: method === 'GET' ? [...conversations.values()] : { deleted: true } })
    }
    if (parts.length === 3) {
      if (method === 'DELETE') { conversations.delete(id); return route.fulfill({ json: { deleted: true } }) }
      if (method === 'PUT' && !conversations.has(id)) conversations.set(id, { id, title: route.request().postDataJSON().title, created_at: '2026-11-05', updated_at: '2026-11-05', messages: [] })
      const item = conversations.get(id)
      if (!item) return route.fulfill({ status: 404, json: {} })
      if (method === 'PATCH') item.title = route.request().postDataJSON().title
      return route.fulfill({ json: item })
    }
    const item = conversations.get(id)!
    if (parts[5] === 'feedback') {
      const message = item.messages.find(message => message.id === parts[4])!
      message.feedback = route.request().postDataJSON()
      return route.fulfill({ json: message.feedback })
    }
    const message = saved(route)
    const previous = item.messages.findIndex(message => message.id === parts[4])
    if (previous < 0) item.messages.push(message)
    else item.messages[previous] = message
    return route.fulfill({ json: message })
  })
  await page.route('**/api/retrievers', route => route.fulfill({ json: {
    default: 'single_nq', retrievers: [{ id: 'single_nq', name: 'Single NQ' }, { id: 'multiset', name: 'Multiset' }],
  } }))
})

test('sends selected retriever, shows answer, sources and trace; resets the session', async ({ page }, testInfo) => {
  let body: unknown
  await page.route('**/api/conversations/*/messages/*', async route => {
    body = route.request().postDataJSON()
    await route.fulfill({ json: saved(route, { ...answer, retriever: 'multiset' }) })
  })
  await page.goto('/')
  await page.getByLabel('Retrieval model').selectOption('multiset')
  await page.screenshot({ path: testInfo.outputPath('desktop.png'), fullPage: true })
  await page.getByLabel('Your question about tuberculosis').fill('What is TB?')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.getByText('This is a', { exact: false })).toBeVisible()
  expect(body).toEqual({ question: 'What is TB?', retriever: 'multiset', attempt: 0 })
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
  await page.route('**/api/conversations/*/messages/*', route => {
    calls++
    return calls === 1
      ? route.fulfill({ status: 503, headers: { 'X-Request-ID': 'failed-request' }, json: { detail: 'busy' } })
      : route.fulfill({ json: saved(route) })
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
  await page.route('**/api/conversations/*/messages/*', async route => { calls++; await gate; await route.fulfill({ json: saved(route) }) })
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
  await page.route('**/api/conversations/*/messages/*', route => route.fulfill({ json: saved(route, { ...answer, answer: 'Safe text <img src="bad" onerror="alert(1)">\n\n[bad link](javascript:alert(1))' }) }))
  await page.goto('/')
  await page.getByLabel('Your question about tuberculosis').fill('What is TB?')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.locator('.answer')).toContainText('Safe text')
  await expect(page.locator('.answer img')).toHaveCount(0)
  await expect(page.locator('.answer a[href^="javascript:"]')).toHaveCount(0)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('mobile.png'), fullPage: true })
})


test('history survives reload, supports feedback, rename and deletion', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('Your question about tuberculosis').fill('My saved question')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.locator('.answer')).toContainText('test answer')
  await page.getByText('Was this answer helpful?', { exact: true }).click()
  await page.getByRole('button', { name: 'Not helpful', exact: true }).click()
  await page.getByLabel('Comment (optional)').fill('Please explain the evidence.')
  await page.getByRole('button', { name: 'Save feedback' }).click()
  await expect(page.getByText('Feedback saved.', { exact: true })).toBeVisible()
  await page.reload()
  await page.getByRole('navigation', { name: 'Saved conversations' }).getByRole('button', { name: 'My saved question', exact: true }).click()
  await expect(page.locator('.answer')).toContainText('test answer')
  await page.getByText('Was this answer helpful?', { exact: true }).click()
  await expect(page.getByLabel('Comment (optional)')).toHaveValue('Please explain the evidence.')
  page.once('dialog', dialog => dialog.accept('Renamed notes'))
  await page.getByRole('button', { name: 'Rename', exact: true }).click()
  await expect(page.getByRole('navigation').getByRole('button', { name: 'Renamed notes' })).toBeVisible()
  page.once('dialog', dialog => dialog.accept())
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(page.getByRole('navigation').getByRole('button', { name: 'Renamed notes' })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: /Understand TB/ })).toBeVisible()
})

test('clear all history removes saved conversations', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('Your question about tuberculosis').fill('A saved chat')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.locator('.answer')).toBeVisible()
  page.once('dialog', dialog => dialog.accept())
  await page.getByRole('button', { name: 'Clear all history' }).click()
  await expect(page.getByText('No saved conversations yet.')).toBeVisible()
})
