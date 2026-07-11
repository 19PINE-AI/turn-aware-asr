// Headless smoke test: load the built site, capture console errors, exercise
// the explorer tabs, and save full-page screenshots.
import { chromium } from 'playwright'
import { createServer } from 'http'
import { readFile } from 'fs/promises'
import { extname, join } from 'path'

const DIST = new URL('../dist', import.meta.url).pathname
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.mp3': 'audio/mpeg', '.svg': 'image/svg+xml' }

const server = createServer(async (req, res) => {
  const path = req.url === '/' ? '/index.html' : decodeURIComponent(req.url.split('?')[0])
  try {
    const buf = await readFile(join(DIST, path))
    res.writeHead(200, { 'content-type': MIME[extname(path)] || 'application/octet-stream' })
    res.end(buf)
  } catch {
    res.writeHead(404); res.end('nf')
  }
})
await new Promise(r => server.listen(4198, r))

const browser = await chromium.launch({ executablePath: '/usr/bin/chromium-browser', args: ['--no-sandbox'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
const errors = []
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()) })
page.on('pageerror', e => errors.push(String(e)))

await page.goto('http://localhost:4198/', { waitUntil: 'networkidle' })
await page.waitForTimeout(1500)
const shots = process.env.SHOTS || 'shots'

await page.screenshot({ path: `${shots}/full.png`, fullPage: true })

// exercise explorer tabs
for (const [label, name] of [['Dictation probe', 'dictation'], ['Spelled entities & context', 'spelled'], ['Earnings-22 biasing', 'earnings']]) {
  await page.click(`button:has-text("${label}")`)
  await page.waitForTimeout(1200)
  await page.locator('#explorer').screenshot({ path: `${shots}/tab-${name}.png` })
}
// back to replay, select more arms
await page.click('button:has-text("Streaming replay benchmark")')
await page.waitForTimeout(800)
await page.click('button:has-text("Released unified model (rank-32)")').catch(() => {})
await page.click('button:has-text("Kyutai STT semantic-VAD (thr 0.5)")').catch(() => {})
await page.waitForTimeout(600)
await page.locator('#explorer').screenshot({ path: `${shots}/tab-replay.png` })

console.log('console errors:', errors.length ? errors : 'none')
await browser.close()
server.close()
