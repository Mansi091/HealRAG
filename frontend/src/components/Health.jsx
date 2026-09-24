import { useState, useEffect } from 'react'
import { getHealth, triggerRecovery } from '../api.js'
import styles from './Health.module.css'

export default function Health() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [recovering, setRecovering] = useState(false)
  const [recoveryResult, setRecoveryResult] = useState(null)

  const fetchHealth = async () => {
    setLoading(true)
    try { setHealth(await getHealth()) }
    catch (e) { setHealth({ status: 'ERROR', summary: e.message }) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchHealth() }, [])

  const handleRecover = async () => {
    setRecovering(true)
    try { setRecoveryResult(await triggerRecovery()) }
    catch (e) { setRecoveryResult({ error: e.message }) }
    finally { setRecovering(false); fetchHealth() }
  }

  if (loading) return <div className={styles.loading}>Loading system health...</div>

  const isHealthy = health?.status === 'HEALTHY'

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>System Health</h2>
        <div className={`${styles.badge} ${isHealthy ? styles.healthy : styles.degraded}`}>
          {isHealthy ? '✅ HEALTHY' : '⚠️ DEGRADED'}
        </div>
      </div>

      <div className={styles.summary}>{health?.summary}</div>

      <div className={styles.grid}>
        {Object.entries(health?.diagnostics || {}).map(([key, val]) => (
          <div key={key} className={styles.card}>
            <div className={styles.cardKey}>{key.replace(/_/g, ' ')}</div>
            <div className={`${styles.cardVal} ${val === false ? styles.fail : val === true ? styles.pass : ''}`}>
              {typeof val === 'boolean' ? (val ? '✅ OK' : '❌ FAIL') : String(val)}
            </div>
          </div>
        ))}
      </div>

      {!isHealthy && (
        <button className={styles.recoverBtn} onClick={handleRecover} disabled={recovering}>
          {recovering ? 'Running Recovery...' : '🔧 Trigger Recovery'}
        </button>
      )}

      {recoveryResult && (
        <div className={styles.recoveryBox}>
          <div className={styles.recoveryTitle}>Recovery Result</div>
          <pre className={styles.recoveryPre}>{JSON.stringify(recoveryResult, null, 2)}</pre>
        </div>
      )}

      <button className={styles.refreshBtn} onClick={fetchHealth}>🔄 Refresh</button>
    </div>
  )
}
