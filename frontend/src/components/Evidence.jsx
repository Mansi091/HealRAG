import { useState, useEffect } from 'react'
import { getEvidence } from '../api.js'
import styles from './Evidence.module.css'

const STATUS_COLOR = { success: '#4ade80', active: '#60a5fa', failed: '#f87171', 'bounded_stop': '#f59e0b' }

export default function Evidence() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchEvents = async () => {
    setLoading(true)
    try {
      const data = await getEvidence(50)
      setEvents(data.events || [])
    } catch (e) { setEvents([]) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchEvents() }, [])

  if (loading) return <div className={styles.loading}>Loading evidence journal...</div>

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>Healing Evidence Journal</h2>
        <div className={styles.count}>{events.length} events</div>
      </div>
      <button className={styles.refreshBtn} onClick={fetchEvents}>🔄 Refresh</button>

      {events.length === 0 ? (
        <div className={styles.empty}>No healing events recorded yet. Events appear when self-healing is triggered.</div>
      ) : (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Query</th>
                <th>Failure Type</th>
                <th>Action</th>
                <th>Attempt</th>
                <th>Result</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {[...events].reverse().map((e, i) => (
                <tr key={i}>
                  <td className={styles.ts}>{e.timestamp?.slice(0, 19).replace('T', ' ')}</td>
                  <td className={styles.query} title={e.query}>{e.query?.slice(0, 30)}{e.query?.length > 30 ? '...' : ''}</td>
                  <td><span className={styles.tag}>{e.failure_type}</span></td>
                  <td>{e.action}</td>
                  <td className={styles.center}>#{e.attempt}</td>
                  <td className={styles.result}>{e.result?.slice(0, 25)}</td>
                  <td><span className={styles.statusDot} style={{ color: STATUS_COLOR[e.status] || '#94a3b8' }}>● {e.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
