import { useEffect, useState } from 'react'

const cache = new Map()

export function useJson(path) {
  const [data, setData] = useState(cache.get(path) || null)
  const [error, setError] = useState(null)
  useEffect(() => {
    if (cache.has(path)) { setData(cache.get(path)); return }
    let dead = false
    fetch(import.meta.env.BASE_URL + path)
      .then(r => { if (!r.ok) throw new Error(`${r.status} loading ${path}`); return r.json() })
      .then(d => { cache.set(path, d); if (!dead) setData(d) })
      .catch(e => { if (!dead) setError(e) })
    return () => { dead = true }
  }, [path])
  return { data, error }
}

export const asset = (p) => import.meta.env.BASE_URL + p
