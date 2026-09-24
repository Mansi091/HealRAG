import { useState, useEffect } from 'react'
import { getEvaluation, runEvaluation } from '../api.js'
import styles from './Evaluation.module.css'

const METRICS = [
  { key: 'context_precision', label: 'Context Precision', icon: '🔍' },
  { key: 'context_recall', label: 'Context Recall', icon: '📚' },
  { key: 'faithfulness', label: 'Faithfulness', icon: '🎯' },
  { key: 'answer_relevancy', label: 'Answer Relevancy', icon: '💡' },
]

export default function Evaluation() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try { setData(await getEvaluation()) }
    catch (e) { setData({ error: e.message }) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchData() }, [])

  const handleRun = async () => {
    setRunning(true)
    try { await runEvaluation(); await fetchData() }
    catch (e) { console.error(e) }
    finally { setRunning(false) }
  }

  if (loading) return <div className={styles.loading}>Loading evaluation data...</div>

  const baseline = data?.baseline || {}
  const regression = data?.regression_report || {}
  const isDegraded = regression.regression_detected

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>Evaluation Metrics</h2>
        <div className={`${styles.badge} ${isDegraded ? styles.degraded : styles.healthy}`}>
          {isDegraded ? '⚠️ DEGRADED' : '✅ HEALTHY'}
        </div>
      </div>

      <div className={styles.metricsGrid}>
        {METRICS.map(({ key, label, icon }) => {
          const val = baseline[key]
          const pct = val != null ? Math.round(val * 100) : null
          const degraded = regression.degraded_metrics?.[key]
          return (
            <div key={key} className={`${styles.metricCard} ${degraded ? styles.metricDegraded : ''}`}>
              <div className={styles.metricIcon}>{icon}</div>
              <div className={styles.metricLabel}>{label}</div>
              <div className={styles.metricValue}>{pct != null ? `${pct}%` : 'N/A'}</div>
              {degraded && (
                <div className={styles.dropBadge}>↓ {(degraded.drop * 100).toFixed(1)}% drop</div>
              )}
              {pct != null && (
                <div className={styles.bar}>
                  <div className={styles.fill} style={{ width: `${pct}%` }} />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {regression.summary && (
        <div className={`${styles.summaryBox} ${isDegraded ? styles.summaryDegraded : styles.summaryOk}`}>
          {regression.summary}
        </div>
      )}

      <div className={styles.actions}>
        <button className={styles.runBtn} onClick={handleRun} disabled={running}>
          {running ? 'Running Evaluation...' : '▶ Run RAGAS Evaluation'}
        </button>
        <button className={styles.refreshBtn} onClick={fetchData}>🔄 Refresh</button>
      </div>

      {!data?.baseline && (
        <div className={styles.noBaseline}>
          No baseline established yet. Run an evaluation to create one.
        </div>
      )}
    </div>
  )
}
