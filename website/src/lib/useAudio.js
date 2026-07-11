import { useEffect, useRef, useState, useCallback } from 'react'
import { asset } from './useJson.js'

/** One shared <audio> element per hook instance; smooth playhead via rAF. */
export function useAudio(url) {
  const ref = useRef(null)
  const raf = useRef(0)
  const [playing, setPlaying] = useState(false)
  const [t, setT] = useState(0)
  const [dur, setDur] = useState(0)

  useEffect(() => {
    const a = new Audio()
    ref.current = a
    a.preload = 'metadata'
    const onMeta = () => setDur(a.duration || 0)
    const onEnd = () => setPlaying(false)
    a.addEventListener('loadedmetadata', onMeta)
    a.addEventListener('ended', onEnd)
    return () => {
      cancelAnimationFrame(raf.current)
      a.pause()
      a.removeEventListener('loadedmetadata', onMeta)
      a.removeEventListener('ended', onEnd)
      a.src = ''
    }
  }, [])

  useEffect(() => {
    const a = ref.current
    if (!a || !url) return
    a.src = asset(url)
    a.load()
    setPlaying(false)
    setT(0)
  }, [url])

  useEffect(() => {
    if (!playing) { cancelAnimationFrame(raf.current); return }
    const tick = () => {
      setT(ref.current?.currentTime || 0)
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing])

  const toggle = useCallback(() => {
    const a = ref.current
    if (!a) return
    if (a.paused) { a.play(); setPlaying(true) } else { a.pause(); setPlaying(false) }
  }, [])

  const seek = useCallback((sec) => {
    const a = ref.current
    if (!a) return
    a.currentTime = Math.max(0, sec)
    setT(a.currentTime)
    if (a.paused) { a.play(); setPlaying(true) }
  }, [])

  const stop = useCallback(() => {
    const a = ref.current
    if (!a) return
    a.pause(); setPlaying(false)
  }, [])

  return { playing, t, dur, toggle, seek, stop }
}

export const fmtTime = (s) => {
  if (!isFinite(s)) return '0:00.0'
  const m = Math.floor(s / 60)
  const r = s - m * 60
  return `${m}:${r.toFixed(1).padStart(4, '0')}`
}
