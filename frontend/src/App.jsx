import { useState } from 'react'
import Chat from './components/Chat.jsx'
import Health from './components/Health.jsx'
import Evidence from './components/Evidence.jsx'
import Evaluation from './components/Evaluation.jsx'
import styles from './App.module.css'

const TABS = ['Chat', 'Health', 'Evaluation', 'Evidence']

export default function App() {
  const [activeTab, setActiveTab] = useState('Chat')

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.shield}>🛡️</span>
          <span className={styles.title}>HealRAG</span>
          <span className={styles.subtitle}>Self-Healing RAG System</span>
        </div>
        <nav className={styles.nav}>
          {TABS.map(tab => (
            <button
              key={tab}
              className={`${styles.navBtn} ${activeTab === tab ? styles.active : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </nav>
      </header>
      <main className={styles.main}>
        {activeTab === 'Chat' && <Chat />}
        {activeTab === 'Health' && <Health />}
        {activeTab === 'Evaluation' && <Evaluation />}
        {activeTab === 'Evidence' && <Evidence />}
      </main>
    </div>
  )
}
