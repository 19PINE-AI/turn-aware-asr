import React, { useEffect, useState } from 'react'
import { asset } from '../../lib/useJson.js'

// module-level singleton so only one probe clip plays at a time
let globalAudio = null
let currentUrl = null
const listeners = new Set()
const notify = () => listeners.forEach(fn => fn(currentUrl))

function playUrl(url) {
  if (!globalAudio) {
    globalAudio = new Audio()
    globalAudio.addEventListener('ended', () => { currentUrl = null; notify() })
  }
  if (currentUrl === url) {
    globalAudio.pause()
    currentUrl = null
    notify()
    return
  }
  globalAudio.src = asset(url)
  globalAudio.play()
  currentUrl = url
  notify()
}

export default function MiniPlay({ url, title = 'play audio' }) {
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    const fn = (cur) => setPlaying(cur === url)
    listeners.add(fn)
    return () => listeners.delete(fn)
  }, [url])
  return (
    <button className={`miniplay${playing ? ' on' : ''}`} title={title}
      onClick={() => playUrl(url)}>
      {playing ? '❚❚' : '▶'}
    </button>
  )
}
