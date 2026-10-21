// End-to-end check for the built site. Serve _site/ (python3 -m http.server 8765 -d _site),
// open http://localhost:8765/ in Chrome, paste this into DevTools, and read the PASS/FAIL table.
// It drives both pages in same-origin iframes at desktop and phone widths.
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const results = [];
  const check = (name, ok, detail = '') => results.push({ check: name, result: ok ? 'PASS' : 'FAIL', detail: String(detail).slice(0, 160) });

  async function open(src, w, h) {
    const f = document.createElement('iframe');
    f.style.cssText = `position:fixed;left:0;top:0;width:${w}px;height:${h}px;border:0;opacity:0;pointer-events:none`;
    f.src = src;
    document.body.appendChild(f);
    await new Promise(r => f.onload = r);
    const errors = [];
    f.contentWindow.addEventListener('error', e => errors.push(e.message));
    await sleep(2500);
    return { f, w: f.contentWindow, d: f.contentDocument, errors, close: () => f.remove() };
  }

  // Computed colours come as rgb()/rgba(), or color(srgb r g b) with 0..1 channels for color-mix().
  const rgb = s => { const n = s.match(/[\d.]+/g).map(Number); return s.startsWith('color(') ? n.slice(0, 3).map(v => v * 255).concat(n.slice(3)) : n; };
  const lum = c => c.slice(0, 3).map(v => (v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)
    .reduce((a, v, i) => a + v * [0.2126, 0.7152, 0.0722][i], 0);
  function contrastFails(p) {
    // A painted ::before layer (the chapter bands) counts as the background.
    const bgOf = el => { for (; el; el = el.parentElement) {
      const before = p.w.getComputedStyle(el, '::before'), b = rgb(before.backgroundColor);
      if (before.content !== 'none' && before.position === 'absolute' && (b.length < 4 || b[3] > 0.9)) return b;
      const a = rgb(p.w.getComputedStyle(el).backgroundColor); if (a.length < 4 || a[3] > 0.9) return a; } return [255, 255, 255]; };
    const out = new Set();
    p.d.querySelectorAll('body *').forEach(el => {
      if (![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) return;
      if (el.closest('.js-plotly-plot, .hero, .sr-only, canvas')) return;
      const cs = p.w.getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity < 0.95 || !el.getClientRects().length) return;
      let fg = rgb(cs.color); const bg = bgOf(el);
      if (fg.length === 4) fg = fg.slice(0, 3).map((v, i) => v * fg[3] + bg[i] * (1 - fg[3]));
      const [lo, hi] = [lum(fg), lum(bg)].sort((a, b) => a - b), ratio = (hi + 0.05) / (lo + 0.05);
      const size = parseFloat(cs.fontSize), need = size >= 24 || (+cs.fontWeight >= 700 && size >= 18.66) ? 3 : 4.5;
      if (ratio < need) out.add(`${el.tagName}.${el.className} ${ratio.toFixed(2)}`);
    });
    return [...out];
  }
  async function drawAll(p, sel) {
    p.d.documentElement.style.scrollBehavior = 'auto';
    const els = [...p.d.querySelectorAll(sel)];
    for (const el of els) { el.scrollIntoView(); await sleep(150); }
    await sleep(1500);
    return [els.filter(el => el.querySelector('.main-svg')).length, els.length];
  }
  async function bothThemes(p, label) {
    for (const dark of [false, true]) {
      if (p.w.isDark() !== dark) p.w.toggleTheme();
      await sleep(900);  // let the 0.3s theme transitions finish
      const f = contrastFails(p);
      check(`${label}: text contrast, ${dark ? 'dark' : 'light'}`, !f.length, f.join(' | '));
    }
    if (p.w.isDark()) p.w.toggleTheme();
  }

  // Story page, desktop
  let p = await open('/', 1280, 800);
  const story = [...p.d.querySelectorAll('.chapter')];
  check('story: 8 chapters', story.length === 8, story.length);
  for (const ch of story) {
    for (const b of ch.querySelectorAll('.beat')) { b.scrollIntoView({ block: 'center' }); await sleep(250); }
    await sleep(600);
    const gd = ch.querySelector('.lead-chart');
    const last = ch.querySelectorAll('.beat').length - 1;
    check(`story: ${ch.id} lead draws and follows beats`, !!gd.querySelector('.main-svg') && ch.querySelector('.beat.on')?.dataset.k === String(last),
      `active beat ${ch.querySelector('.beat.on')?.dataset.k} of ${last}`);
  }
  const [sd, st] = await drawAll(p, '.lazy-chart');
  check('story: supporting charts draw', sd === st, `${sd}/${st}`);
  check('story: no click-only elements', !p.d.querySelectorAll('span[onclick], th[onclick], div[onclick]:not([role])').length);
  check('story: no horizontal overflow (1280px)', p.d.documentElement.scrollWidth <= 1280, p.d.documentElement.scrollWidth);
  await bothThemes(p, 'story');
  check('story: no script errors', !p.errors.length, p.errors.join(' | '));
  p.close();

  // Story page, phone: the active beat moves the chart marker
  p = await open('/', 390, 760);
  check('story: no horizontal overflow (390px)', p.d.documentElement.scrollWidth <= 390, p.d.documentElement.scrollWidth);
  const beat = p.d.querySelectorAll('#growth .beat')[1];
  p.w.scrollTo(0, beat.getBoundingClientRect().top + p.w.scrollY - 560);
  await sleep(2000);
  const marker = (p.d.getElementById('lead-growth').layout?.shapes || []).map(s => s.x0).join(',');
  check('story: phone beat 2 moves the growth marker', p.d.querySelector('#growth .beat.on')?.dataset.k === '1' && marker && !marker.startsWith('Spring 2005'), marker);
  p.close();

  // Explore page
  p = await open('/explore/?course=COE%20420', 1280, 800);
  check('explore: ?course= opens that course', /COE 420/.test(p.d.querySelector('#dep-result .cx-title')?.textContent || ''));
  check('explore: lookup comes first', p.d.querySelector('main > section')?.id === 'browse');
  const [ed, et] = await drawAll(p, '.lazy-chart');
  check('explore: all charts draw', ed === et && et >= 62, `${ed}/${et}`);
  check('explore: no click-only elements', !p.d.querySelectorAll('span[onclick], th[onclick], div[onclick]:not([role])').length);
  await bothThemes(p, 'explore');
  check('explore: no script errors', !p.errors.length, p.errors.join(' | '));
  p.close();

  p = await open('/explore/?q=Calculus', 1280, 800);
  check('explore: ?q= free text finds a course', !!p.d.querySelector('#dep-result .cx-title') || p.d.getElementById('dep-suggestions').style.display === 'block');
  p.close();

  p = await open('/explore/#chart-saturday_decline', 1280, 800);
  check('explore: #chart- deep link draws that chart', !!p.d.querySelector('#chart-saturday_decline .main-svg'));
  p.close();

  const share = await (await fetch('/share/growth/index.html')).text();
  check('share: stub redirects to the chart in Explore', share.includes('/explore/#chart-growth'));

  console.table(results);
  const failed = results.filter(r => r.result === 'FAIL').length;
  return `${results.length - failed}/${results.length} checks passed`;
})();
