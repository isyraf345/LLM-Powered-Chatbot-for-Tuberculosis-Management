import { useState } from 'react'
import { saveFeedback } from './api'
import type { SavedMessage } from './api'

export default function Feedback({ cid, messageId, initial }: { cid: string; messageId: string; initial?: SavedMessage['feedback'] }) {
  const [rating, setRating] = useState<-1 | 1 | null>(initial?.rating ?? null)
  const [comment, setComment] = useState(initial?.comment ?? '')
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')
  return <details className="feedback"><summary>Was this answer helpful?</summary>
    <div className="feedback-options">
      <button type="button" aria-pressed={rating === 1} disabled={saving} onClick={() => { setRating(1); setNotice('') }}>Helpful</button>
      <button type="button" aria-pressed={rating === -1} disabled={saving} onClick={() => { setRating(-1); setNotice('') }}>Not helpful</button>
    </div>
    <label>Comment (optional)<textarea value={comment} maxLength={2000} disabled={saving} onChange={e => { setComment(e.target.value); setNotice('') }} /></label>
    <button type="button" disabled={saving || rating === null} onClick={async () => {
      if (rating === null) return
      setSaving(true)
      try { await saveFeedback(cid, messageId, rating, comment); setNotice('Feedback saved.') }
      catch { setNotice('Could not save feedback. Please try again.') }
      finally { setSaving(false) }
    }}>{saving ? 'Saving…' : 'Save feedback'}</button>
    <span role="status">{notice}</span>
  </details>
}
