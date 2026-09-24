import { useState, useRef, useEffect } from 'react'
import { queryHealRAG, triggerIngest } from '../api.js'
import styles from './Chat.module.css'

export default function Chat() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hello! I am HealRAG — a self-healing AI assistant. Ask me anything about your indexed documents.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastMeta, setLastMeta] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setLoading(true)
    try {
      const data = await queryHealRAG(q)
      setLastMeta(data)
      setMessages(prev => [...prev, { role: 'assistant', text: data.answer, meta: data }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: `Error: ${err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }

  const handleIngest = async () => {
    setLoading(true)
    try {
      const res = await triggerIngest()
      setMessages(prev => [...prev, { role: 'assistant', text: `✅ Ingestion complete. Indexed ${res.chunks_indexed || 0} chunks from ${res.documents_loaded || 0} documents.` }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: `Ingestion error: ${err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <span className={styles.toolbarTitle}>💬 Chat</span>
        <button className={styles.ingestBtn} onClick={handleIngest} disabled={loading}>
          📥 Ingest Documents
        </button>
      </div>

      <div className={styles.messages}>
        {messages.map((msg, i) => (
          <div key={i} className={`${styles.msg} ${styles[msg.role]}`}>
            <div className={styles.bubble}>{msg.text}</div>
            {msg.meta && (
              <div className={styles.metaRow}>
                {msg.meta.sources?.length > 0 && (
                  <div className={styles.sources}>
                    <span className={styles.metaLabel}>📄 Sources:</span>
                    {msg.meta.sources.map((s, j) => (
                      <span key={j} className={styles.sourceTag}>{s.source} (p.{s.page})</span>
                    ))}
                  </div>
                )}
                <div className={styles.scores}>
                  <span className={styles.score}>🔍 Retrieval: {(msg.meta.retrieval?.relevance_score * 100 || 0).toFixed(0)}%</span>
                  <span className={styles.score}>🎯 Grounding: {(msg.meta.grounding?.score * 100 || 0).toFixed(0)}%</span>
                  {msg.meta.healing?.triggered && (
                    <span className={styles.healBadge}>🔧 Healed (×{msg.meta.healing.retry_count})</span>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className={`${styles.msg} ${styles.assistant}`}>
            <div className={`${styles.bubble} ${styles.typing}`}>
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className={styles.inputRow}>
        <textarea
          className={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask a question... (Enter to send)"
          rows={2}
          disabled={loading}
        />
        <button className={styles.sendBtn} onClick={send} disabled={loading || !input.trim()}>
          {loading ? '...' : '→'}
        </button>
      </div>
    </div>
  )
}
