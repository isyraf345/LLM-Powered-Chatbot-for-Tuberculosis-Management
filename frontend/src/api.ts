export type Retriever = { id: string; name: string }
export type Passage = { text: string; source: string; page: number | null; score: number }
export type Answer = {
  answer: string
  retriever: string
  generation_model: string
  passages: Passage[]
  request_id: string
  trace_id: string | null
}

export class ApiError extends Error {
  constructor(message: string, public requestId: string | null = null) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, { ...init, signal: AbortSignal.timeout(300_000) })
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === 'TimeoutError'
    throw new ApiError(timedOut
      ? 'The request timed out. The server may still be processing it; wait before retrying.'
      : 'Cannot reach the API. Check that the backend is running, then try again.')
  }
  if (!response.ok) {
    const messages: Record<number, string> = {
      422: 'Check your question: use between 1 and 4,000 characters.',
      502: 'The answer provider could not be reached. Try again shortly.',
      503: 'The service is busy or unavailable. Wait a moment, then retry.',
      504: 'Answer generation timed out. Try again shortly.',
    }
    throw new ApiError(messages[response.status] ?? 'The request failed. Please try again.', response.headers.get('X-Request-ID'))
  }
  try { return await response.json() as T }
  catch { throw new ApiError('The API returned an unreadable response. Check the backend connection.') }
}

export const getRetrievers = () => request<{ default: string; retrievers: Retriever[] }>('/retrievers')
export const ask = (question: string, retriever: string) => request<Answer>('/chat', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, retriever }),
})
