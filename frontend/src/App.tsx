import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ApiError, ask, getRetrievers } from './api'
import type { Answer, Retriever } from './api'

type Turn = { id: number; question: string; retriever: string; answer?: Answer; error?: ApiError }
const suggestions = ['What is tuberculosis?', 'How does TB spread?', 'How is tuberculosis treated?']

function Sources({ answer }: { answer: Answer }) {
  return <details className="sources">
    <summary>Retrieved passages <span>{answer.passages.length}</span></summary>
    <p className="muted">These are search results, not verified citations. The context limit may exclude some passages from the answer.</p>
    {answer.passages.map((passage, index) => <article className="passage" key={index}>
      <div className="passage-heading"><b>{index + 1}. {passage.source}</b><span>Page {passage.page ?? 'unknown'}</span></div>
      <p>{passage.text}</p>
      <small>Retrieval similarity: {passage.score.toFixed(3)} · Not a confidence score</small>
    </article>)}
  </details>
}

export default function App() {
  const [retrievers, setRetrievers] = useState<Retriever[]>([])
  const [retriever, setRetriever] = useState('single_nq')
  const [connectionError, setConnectionError] = useState('')
  const [connecting, setConnecting] = useState(true)
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const sending = useRef(false)
  const nextId = useRef(0)
  const bottom = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)

  async function connect() {
    setConnecting(true)
    setConnectionError('')
    try {
      const data = await getRetrievers()
      setRetrievers(data.retrievers)
      setRetriever(data.default)
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : 'Cannot connect to the API.')
    } finally { setConnecting(false) }
  }
  useEffect(() => { void connect() }, [])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [turns, busy])

  async function send(text: string, selected: string, retryId?: number) {
    if (sending.current || !text.trim() || text.length > 4000 || !retrievers.length) return
    sending.current = true
    setBusy(true)
    const id = retryId ?? nextId.current++
    const turn: Turn = { id, question: text, retriever: selected }
    setTurns(previous => retryId === undefined ? [...previous, turn] : previous.map(item => item.id === id ? turn : item))
    if (retryId === undefined) setQuestion('')
    try {
      const answer = await ask(text, selected)
      setTurns(previous => previous.map(item => item.id === id ? { ...item, answer } : item))
    } catch (error) {
      const failure = error instanceof ApiError ? error : new ApiError('Something went wrong. Please retry.')
      setTurns(previous => previous.map(item => item.id === id ? { ...item, error: failure } : item))
    } finally {
      sending.current = false
      setBusy(false)
      input.current?.focus()
    }
  }

  return <div className="shell">
    <aside className="sidebar">
      <a className="brand" href="#main"><span className="brand-mark" aria-hidden="true">+</span><span>TB Companion<small>INFORMATION ASSISTANT</small></span></a>
      <button className="new-chat" disabled={busy || turns.length === 0} onClick={() => { setTurns([]); setQuestion(''); input.current?.focus() }}>＋ New conversation</button>
      <div className="sidebar-section"><label htmlFor="retriever">Retrieval model</label>
        <select id="retriever" value={retriever} disabled={busy || connecting || !retrievers.length} onChange={event => setRetriever(event.target.value)}>
          {retrievers.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
        <p>Choose how supporting passages are found.</p>
      </div>
      <div className="session-note"><span className="eyebrow">THIS SESSION</span><p>Messages stay in this tab and clear on refresh. Each question is answered independently.</p></div>
      <div className="sidebar-footer">Learn. Ask. Understand.<br /><small>An educational research project.</small></div>
    </aside>

    <main id="main">
      <header className="topbar"><span>Tuberculosis knowledge assistant</span><span className="badge">Research preview</span></header>
      <div className="conversation">
        {connectionError && <div className="error" role="alert">{connectionError} <button onClick={() => void connect()} disabled={connecting}>Reconnect</button></div>}
        {connecting && <p role="status" className="muted">Connecting to the API…</p>}
        {turns.length === 0 && <section className="welcome">
          <span className="eyebrow">A LITTLE CLARITY STARTS WITH A QUESTION</span>
          <h1>Understand TB.<br /><em>One question at a time.</em></h1>
          <p>Explore tuberculosis information with an AI assistant and inspect the passages retrieved from its reference documents.</p>
          <div className="suggestions">{suggestions.map((text, index) => <button key={text} disabled={busy || !retrievers.length} onClick={() => { setQuestion(text); input.current?.focus() }}><span>0{index + 1}</span>{text}<b aria-hidden="true">↗</b></button>)}</div>
        </section>}
        <div className="turns" role="log" aria-live="polite" aria-label="Conversation">
          {turns.map(turn => <section className="turn" key={turn.id}>
            <div className="question"><span className="eyebrow">YOU</span><p>{turn.question}</p></div>
            <div className="response"><span className="eyebrow">TB COMPANION</span>
              {!turn.answer && !turn.error && <p role="status" className="loading">Finding passages and preparing an answer… <small>The first request may take longer while the model loads.</small></p>}
              {turn.error && <div className="error" role="alert"><p>{turn.error.message}</p>{turn.error.requestId && <small>Request: {turn.error.requestId}</small>}<button disabled={busy} onClick={() => void send(turn.question, turn.retriever, turn.id)}>Retry question</button></div>}
              {turn.answer && <>
                <div className="answer"><ReactMarkdown skipHtml components={{ img: () => null, a: ({ children, href }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a> }}>{turn.answer.answer}</ReactMarkdown></div>
                <Sources answer={turn.answer} />
                <details className="request-details"><summary>Request details</summary><dl><dt>Retriever</dt><dd>{turn.answer.retriever}</dd><dt>Model</dt><dd>{turn.answer.generation_model}</dd><dt>Request ID</dt><dd>{turn.answer.request_id}</dd><dt>Trace ID</dt><dd>{turn.answer.trace_id ?? 'Tracing disabled'}</dd></dl></details>
              </>}
            </div>
          </section>)}
        </div>
        <div ref={bottom} />
      </div>
      <footer className="composer-area"><form onSubmit={event => { event.preventDefault(); void send(question, retriever) }}>
        <label className="sr-only" htmlFor="question">Your question about tuberculosis</label>
        <textarea id="question" ref={input} placeholder="What would you like to know about TB?" value={question} maxLength={4000} rows={2} onChange={event => setQuestion(event.target.value)} onKeyDown={event => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(question, retriever) }
        }} />
        <div className="composer-bottom"><small>{question.length.toLocaleString()} / 4,000 · Shift + Enter for a new line</small><button className="send" type="submit" disabled={busy || !retrievers.length || !question.trim()}>{busy ? 'Working…' : 'Send question ↑'}</button></div>
      </form><p className="disclaimer">Each question is answered independently. Messages clear on refresh.<br />For learning, not diagnosis. AI answers can be incorrect. Consult a qualified healthcare professional for medical advice.</p></footer>
    </main>
  </div>
}
