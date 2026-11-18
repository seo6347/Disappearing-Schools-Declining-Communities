/* Entry point only. All state, selectors and listeners belong to the supplied root. */
function initMethod(root) {
  "use strict";
  if (!root || root.id !== "tab-method") throw new Error("initMethod requires #tab-method");
  const instanceKey = Symbol.for("method-tab.instance");
  if (root[instanceKey]) return root[instanceKey];
  const doc = root.ownerDocument;
  const win = doc.defaultView;
  const dialog = root.querySelector(".method-dialog");
  const scroller = root.querySelector(".method-scroller");
  const mount = root.querySelector(".method-svg-mount");
  const status = root.querySelector(".method-load-status");
  const steps = Array.from(root.querySelectorAll("[data-step-id]"));
  const qaEnabled = new URLSearchParams(win.location.search).get("qa") === "1";
  const abort = new AbortController();
  const motion = win.matchMedia("(prefers-reduced-motion: reduce)");
  const shortLandscape = win.matchMedia("(max-width: 767px) and (max-height: 520px)");
  let destroyed = false;
  let readyPromise;
  let svg;
  let nodes;
  let timeline;
  let trigger;
  let resizeObserver;
  let ranges = [];
  let state;
  let qa;
  let frame = 0;
  let refreshFrame = 0;
  let savedBody;
  let opener;
  let activeIndex = 0;
  let activeProgress = 0;
  let fallback = false;
  let svgIdentity;
  let debugState = {};

  const clamp = (n) => Math.max(0, Math.min(1, n));
  const ramp = (n, start, end) => clamp((n - start) / (end - start));
  const listen = (target, type, callback, options = {}) => target.addEventListener(type, callback, { ...options, signal: abort.signal });
  function message(text) { status.textContent = text; status.hidden = !text; }
  function opacity(node, value) { node.style.opacity = String(value); }

  async function prepare() {
    const base = new URL(root.dataset.methodAssets || "./", doc.baseURI);
    const response = await win.fetch(new URL("img/scene-01.svg", base), { signal: abort.signal });
    if (!response.ok) throw new Error("SVG HTTP " + response.status);
    const parsed = new win.DOMParser().parseFromString(await response.text(), "image/svg+xml");
    if (parsed.querySelector("parsererror") || parsed.documentElement.localName !== "svg") throw new Error("Invalid scene SVG");
    if (destroyed) return;
    // This is the only SVG source and the only insertion. Reopening never replaces it.
    svg = doc.importNode(parsed.documentElement, true);
    mount.replaceChildren(svg);
    svgIdentity = svg;
    mount.setAttribute("aria-busy", "false");
    const find = (selector) => root.querySelector(selector);
    const reveal = Array.from(svg.querySelectorAll("[data-method-reveal]"));
    reveal.forEach((node) => {
      const path = svg.querySelector(node.getAttribute("href"));
      const length = path.getTotalLength();
      node.dataset.methodLength = String(length);
      node.setAttribute("stroke-dasharray", String(length));
    });
    const outline = find("[data-method-outline-mask]");
    const outlineLength = find("#method-catchment-shape").getTotalLength();
    outline.removeAttribute("pathLength");
    outline.setAttribute("stroke-dasharray", String(outlineLength));
    nodes = {
      fill: find("[data-method-fill]"), outline, outlineLength, reveal,
      school: find('[data-method-layer="school"]'),
      catchmentLegend: find('[data-method-layer="catchment-legend"]'),
      censusLegend: find('[data-method-layer="census-legend"]'),
      mobileCatchmentLegend: find('[data-method-mobile-legend="catchment"]'),
      mobileCensusLegend: find('[data-method-mobile-legend="census"]'),
      mobileJointLegend: find('[data-method-mobile-legend="joint"]'),
      mobilePointLegend: find('[data-method-mobile-legend="point"]'),
      censusBoundaries: find('#method-census-boundaries'),
      chart: find(".method-age-chart"), first: find('[data-method-age="first"]'),
      second: find('[data-method-age="second"]'), result: find(".method-age-result")
    };
    nodes.extra = Object.fromEntries(Array.from(svg.querySelectorAll('[data-method-new]')).map(node => [node.dataset.methodNew, node]));
    nodes.lines = Array.from(svg.querySelectorAll('#method-allocation-lines path, #method-joint-allocation > g[marker-end] path'));
    nodes.lines.forEach(path => { path.dataset.length = path.getTotalLength(); path.style.strokeDasharray = path.dataset.length; });
    const dataResponse = await win.fetch(new URL('data/method_example_fixture.json', base), { signal: abort.signal });
    if (!dataResponse.ok) throw new Error('Fixture HTTP ' + dataResponse.status);
    const fixture = await dataResponse.json();
    const secondResponse = await win.fetch(new URL('img/scene-02.svg', base), { signal: abort.signal });
    if (!secondResponse.ok) throw new Error('Scene 2 SVG HTTP ' + secondResponse.status);
    const secondDocument = new win.DOMParser().parseFromString(await secondResponse.text(), 'image/svg+xml');
    if (secondDocument.querySelector('parsererror')) throw new Error('Invalid Scene 2 SVG');
    nodes.secondSvg = doc.importNode(secondDocument.documentElement, true);
    const secondMount = root.querySelector('.method-svg-mount-second');
    secondMount.replaceChildren(nodes.secondSvg); secondMount.setAttribute('aria-busy', 'false');
    nodes.secondLayers = Object.fromEntries(Array.from(nodes.secondSvg.querySelectorAll('[data-s2]')).map(node => [node.dataset.s2, node]));
    nodes.secondEdges = Array.from(nodes.secondSvg.querySelectorAll('[data-s2="edges"] path'));
    // A separate reveal mask keeps each edge's dashed pattern intact.
    const ns = 'http://www.w3.org/2000/svg';
    const defs = doc.createElementNS(ns, 'defs'); nodes.secondSvg.prepend(defs);
    nodes.secondEdges.forEach((path, index) => {
      const mask = doc.createElementNS(ns, 'mask'); mask.id = 'method-s2-edge-reveal-' + index;
      mask.setAttribute('maskUnits', 'userSpaceOnUse'); mask.setAttribute('x', '0'); mask.setAttribute('y', '0'); mask.setAttribute('width', '560'); mask.setAttribute('height', '610');
      const draw = path.cloneNode(false); draw.removeAttribute('data-pair'); draw.setAttribute('stroke', 'white'); draw.setAttribute('stroke-width', '10'); draw.setAttribute('fill', 'none');
      draw.dataset.length = path.getTotalLength(); draw.style.strokeDasharray = draw.dataset.length;
      mask.append(draw); defs.append(mask); path.setAttribute('mask', 'url(#' + mask.id + ')');
      path.methodReveal = draw;
    });
    const jointExample = fixture.independent_diagram_examples.joint;
    root.querySelector('[data-method-diagram-summary]').textContent = ['A', 'B'].map((school, index) => school + '초 ' + jointExample.initial[index] + '명 → ' + jointExample.illustrative_adjusted[index] + '명').join(' / ');
    await prepareLaterScenes(base, fixture);
    createTimeline();
    renderAt(0);
  }

  function createTimeline() {
    timeline?.kill();
    state = { fill: 0, outline: 0, school: 0, legend: 0, census: 0, censusLegend: 0, chart: 0, first: 0, second: 0, result: 0, hold: 0 };
    if (!win.gsap || !win.ScrollTrigger) {
      fallback = true;
      root.classList.add("method-static");
      message("스크롤 애니메이션을 불러오지 못해 정적 설명으로 표시합니다.");
      return;
    }
    win.gsap.registerPlugin(win.ScrollTrigger);
    timeline = win.gsap.timeline({ paused: true, defaults: { ease: "none" } });
    timeline.addLabel("1-1", 0)
      .to(state, { outline: 1, duration: .32 }, 0)
      .to(state, { fill: 1, duration: .3 }, .03)
      .to(state, { school: 1, legend: 1, duration: .2 }, .14)
      .addLabel("1-2", 1)
      .to(state, { census: 1, duration: .48 }, 1)
      .to(state, { censusLegend: 1, duration: .2 }, 1.12)
      .addLabel("1-3", 2)
      .to(state, { chart: 1, first: 1, duration: .18 }, 2.02)
      .to(state, { second: 1, duration: .18 }, 2.18)
      .to(state, { result: 1, duration: .16 }, 2.34)
      .to(state, { hold: 1, duration: .5 }, 2.5);
    steps.slice(3).forEach((step, index) => timeline.addLabel(step.dataset.stepId, index + 3));
    timeline.to(state, { hold: 2, duration: steps.length - 3 }, 3);
    fallback = false;
    root.classList.remove("method-static");
    message("");
  }

  function renderAt(position) {
    if (!nodes) return;
    if (timeline) timeline.time(Math.max(0, Math.min(steps.length, position)), true);
    else Object.keys(state).forEach((key) => { state[key] = 1; });
    const reduce = motion.matches;
    opacity(nodes.fill, reduce ? ramp(position, 0, .12) : state.fill);
    nodes.outline.setAttribute("stroke-dashoffset", String(nodes.outlineLength * (1 - (reduce ? state.fill : state.outline))));
    opacity(nodes.school, state.school);
    opacity(nodes.catchmentLegend, state.legend);
    opacity(nodes.censusLegend, state.censusLegend);
    opacity(nodes.mobileCatchmentLegend, state.legend);
    opacity(nodes.mobileCensusLegend, state.censusLegend);
    nodes.reveal.forEach((node, index) => {
      const amount = reduce ? (state.census > 0 ? 1 : 0) : ramp(state.census, index * .065, .6 + index * .065);
      node.setAttribute("stroke-dashoffset", String(Number(node.dataset.methodLength) * (1 - amount)));
      node.style.opacity = reduce ? String(state.census) : "1";
    });
    opacity(nodes.chart, state.chart);
    opacity(nodes.first, state.first);
    opacity(nodes.second, state.second);
    opacity(nodes.result, state.result);
    nodes.chart.setAttribute("aria-hidden", state.chart < .05 ? "true" : "false");
    const enter = n => ramp(position, n - 1, n - .5);
    const a4 = enter(4), a5 = enter(5), a6 = enter(6), a7 = enter(7), a8 = enter(8);
    // Retain the original boundaries in the DOM, but remove them as the joint zone appears.
    opacity(nodes.censusBoundaries, 1 - a7);
    nodes.mobilePointLegend.hidden = a5 === 0;
    opacity(nodes.mobilePointLegend, a5);
    opacity(nodes.extra['point-legend'], a5);
    opacity(nodes.extra.k, a4 * (1 - a5));
    opacity(nodes.extra.points, a5);
    opacity(nodes.extra.links, a5 * (1 - a6));
    opacity(nodes.extra.outside, a6 * (1 - a7));
    opacity(nodes.extra.joint, a7);
    nodes.mobileJointLegend.hidden = a7 === 0;
    opacity(nodes.mobileJointLegend, a7);
    opacity(nodes.extra.split, a7);
    opacity(nodes.extra.calibrated, a8);
    svg.querySelectorAll('[data-method-before]').forEach(node => opacity(node, 1 - a8 * .5));
    nodes.lines.forEach((path, i) => {
      const progress = i < 5 ? ramp(a5, .22 + i * .09, .6 + i * .09) : ramp(a7, .3, 1);
      path.style.strokeDashoffset = String(Number(path.dataset.length) * (1 - (reduce && progress > 0 ? 1 : progress)));
    });
    if (win.matchMedia('(max-width: 767px)').matches) {
      svg.setAttribute('viewBox', (75 * (1 - a7)) + ' 0 ' + (410 + 150 * a7) + ' 550');
    }
    if (nodes.secondLayers) {
      const layer = nodes.secondLayers;
      const arrive = n => ramp(position, 7 + n, 7.5 + n);
      const filter = arrive(1), offices = arrive(2), graph = arrive(3), distances = arrive(4), prune = arrive(5), candidates = arrive(6);
      opacity(layer.filter, filter * (1 - offices)); opacity(layer.map, offices);
      nodes.secondSvg.querySelectorAll('[data-s2-number]').forEach((node, i) => opacity(node, ramp(filter, i * .22, .4 + i * .22)));
      opacity(layer['other-office'], 1 - graph); opacity(layer['office-legend'], 1 - graph);
      opacity(layer['cross-office'], (1 - graph) * ramp(offices, .15, .5));
      opacity(layer['cross-mark'], ramp(offices, .5, .8));
      const fill = [244,188,193].map((v, i) => Math.round(v + ([243,240,230][i] - v) * graph));
      nodes.secondSvg.querySelector('#method-s2-office-a').setAttribute('fill', 'rgb(' + fill.join(',') + ')');
      opacity(layer.edges, graph); opacity(layer.distances, distances); opacity(layer.candidates, candidates);
      nodes.secondEdges.forEach((edge, i) => {
        const amount = ramp(graph, i * .07, .55 + i * .07);
        edge.methodReveal.style.strokeDashoffset = String(Number(edge.methodReveal.dataset.length) * (1 - amount));
        opacity(edge, ['AC', 'AD'].includes(edge.dataset.pair) ? 1 - prune : 1);
      });
      nodes.secondSvg.querySelectorAll('[data-s2="distances"] [data-pair]').forEach(node => opacity(node, ['AC', 'AD'].includes(node.dataset.pair) ? 1 - prune : 1));
      nodes.secondSvg.querySelectorAll('[data-s2-candidate]').forEach((node, i) => opacity(node, ramp(candidates, i * .23, .5 + i * .23)));
      root.querySelectorAll('[data-method-candidate]').forEach((node, i) => opacity(node, ramp(candidates, i * .23, .5 + i * .23)));
    }
    renderLaterScenes(position);
    debugState = { ...state, sameSvg: svg === svgIdentity, svgCount: mount.querySelectorAll("svg").length };
  }

  async function prepareLaterScenes(base, fixture) {
    const schools = Object.fromEntries(fixture.schools.map(s => [s.id, s]));
    const rows = fixture.stage4_commute_rows;
    const weight = rows.reduce((sum, row) => sum + row.student_weight, 0);
    const burden = Object.fromEntries(['B', 'D'].map(k => [k, rows.reduce((sum, row) => sum + row.student_weight * Math.max(row['after_' + k + '_min'] - row.before_min, 0), 0) / weight]));
    if (fixture.schools.some(s => s.capacity !== s.rooms * 28) || ['B','D'].some(k => Math.abs(burden[k] - fixture.stage4_mean_additional_min['retained_' + k]) > 1e-8)) throw new Error('Inconsistent capacity or weighted commute fixture');
    const national = fixture.national_actual_results;
    const format = value => Number(value.toFixed(2)).toLocaleString('ko-KR');
    const bindings = {
      'abc-capacity': 'C초 ' + schools.C.capacity + '명 < ABC ' + ['A','B','C'].reduce((n,k) => n + schools[k].students, 0) + '명',
      'bcd-total': weight + '명',
      'burden-B': format(burden.B) + '분', 'burden-D': format(burden.D) + '분',
      'burden-comparison': 'B초 존치 ' + format(burden.B) + '분 / D초 존치 ' + format(burden.D) + '분',
      'national-clusters': national.final_clusters.toLocaleString('ko-KR') + '개 군집',
      'national-summary': national.final_clusters + '개 군집 · 통합중심학교 ' + national.retained_schools + '개교 · 폐교 대상 ' + national.closed_schools.toLocaleString('ko-KR') + '개교'
    };
    fixture.candidates.forEach(c => { bindings['score-' + c.id] = c.access_score_min + '분'; });
    ['B','C','D'].forEach(k => { bindings['capacity-' + k] = k + '초 ' + schools[k].capacity + '명 ' + (schools[k].capacity >= weight ? '✓' : '×'); });
    nodes.laterScenes = [];
    for (const scene of [3,4,5]) {
      const response = await win.fetch(new URL('img/scene-0' + scene + '.svg', base), { signal: abort.signal });
      if (!response.ok) throw new Error('Scene ' + scene + ' SVG HTTP ' + response.status);
      const parsed = new win.DOMParser().parseFromString(await response.text(), 'image/svg+xml');
      if (parsed.querySelector('parsererror')) throw new Error('Invalid scene SVG ' + scene);
      if (destroyed) return;
      const element = doc.importNode(parsed.documentElement, true);
      const target = root.querySelector('[data-svg-scene="' + scene + '"]');
      target.replaceChildren(element); target.setAttribute('aria-busy', 'false');
      const drawing = { element, layers: Array.from(element.querySelectorAll('[data-appear]')), draws: Array.from(element.querySelectorAll('[data-draw]')), bars: Array.from(element.querySelectorAll('[data-bar]')), dims: Array.from(element.querySelectorAll('[data-dim]')), scene4Stages: Array.from(element.querySelectorAll('[data-s4-from], [data-s4-until]')) };
      drawing.layers.forEach(layer => opacity(layer, 0));
      drawing.draws.forEach(path => { path.dataset.length = path.getTotalLength(); path.style.strokeDasharray = path.dataset.length; });
      nodes.laterScenes.push(drawing);
    }
    root.querySelectorAll('[data-bind]').forEach(element => { if (bindings[element.dataset.bind] !== undefined) element.textContent = bindings[element.dataset.bind]; });
    nodes.national = root.querySelector('.method-national');
    const map = nodes.national.querySelector('img');
    map.src = new URL('img/final-school-map.png', base).href;
    map.addEventListener('load', requestRefresh, { once: true, signal: abort.signal });
    map.addEventListener('error', () => message('전국 지도 파일을 불러오지 못했습니다.'), { once: true, signal: abort.signal });
  }

  function renderLaterScenes(position) {
    if (!nodes.laterScenes) return;
    const startOf = id => steps.findIndex(step => step.dataset.stepId === id);
    const appear = id => ramp(position, startOf(id), startOf(id) + .5);
    nodes.laterScenes.forEach(drawing => {
      drawing.layers.forEach(element => {
        const start = startOf(element.dataset.appear) + Number(element.dataset.appearOffset || 0);
        let amount = ramp(position, start, start + Number(element.dataset.appearDuration || .5));
        if (element.dataset.until) amount *= 1 - appear(element.dataset.until);
        if (element.dataset.return) amount = Math.max(amount, appear(element.dataset.return));
        opacity(element, amount);
      });
      drawing.draws.forEach(path => {
        const start = startOf(path.dataset.draw) + Number(path.dataset.drawOffset || 0);
        const amount = ramp(position, start, start + Number(path.dataset.drawDuration || .5));
        path.style.strokeDashoffset = String(Number(path.dataset.length) * (1 - amount));
      });
      drawing.bars.forEach(bar => bar.setAttribute('width', Number(bar.dataset.width) * appear(bar.dataset.bar)));
      drawing.dims.forEach(element => opacity(element, 1 - .65 * appear(element.dataset.dim)));
      drawing.scene4Stages.forEach(element => {
        const start = startOf('4-3');
        let amount = 1;
        if (element.dataset.s4From !== undefined) {
          const from = start + Number(element.dataset.s4From);
          amount *= ramp(position, from, from + .15);
        }
        if (element.dataset.s4Until !== undefined) {
          const until = start + Number(element.dataset.s4Until);
          amount *= 1 - ramp(position, until, until + .15);
        }
        opacity(element, amount);
      });
      const scene4Map = drawing.element.querySelector('[data-s4-map]');
      if (scene4Map) {
        const amount = appear('4-3');
        scene4Map.setAttribute('transform', 'translate(' + (15 * amount) + ' ' + (35 * amount) + ') scale(' + (1 - .14 * amount) + ')');
      }
    });
    const national = appear('5-5');
    opacity(nodes.laterScenes[2].element, 1 - national);
    opacity(root.querySelector('[data-example-final]'), 1 - national);
    opacity(nodes.national, national);
    nodes.national.setAttribute('aria-hidden', String(national === 0));
    nodes.laterScenes[2].element.setAttribute('aria-hidden', String(national === 1));
  }

  function measure() {
    const top = scroller.getBoundingClientRect().top;
    const y = scroller.scrollTop;
    const mobile = win.matchMedia("(max-width: 767px)").matches;
    // Size the desktop story to the open dialog. Mobile keeps its existing svh layout.
    if (mobile) dialog.style.removeProperty('--method-modal-unit');
    else dialog.style.setProperty('--method-modal-unit', (scroller.clientHeight / 100) + 'px');
    steps.forEach(step => {
      const copy = step.querySelector('.method-copy');
      if (mobile) copy.style.removeProperty('--method-copy-top');
      else {
        const preferredTop = Math.max(88, Math.min(130, scroller.clientHeight * .14));
        // Long explanations can scroll past the top so their final notes remain reachable.
        copy.style.setProperty('--method-copy-top', Math.min(preferredTop, scroller.clientHeight - copy.offsetHeight - 24) + 'px');
      }
    });
    nodes.laterScenes?.forEach(drawing => {
      const element = drawing.element;
      const bounds = element.getBoundingClientRect();
      const scale = Math.max(.01, Math.min(bounds.width / 700, bounds.height / 610));
      element.querySelectorAll('text').forEach(text => {
        if (!text.dataset.originalFont) text.dataset.originalFont = text.getAttribute('font-size') || text.parentElement.closest('[font-size]')?.getAttribute('font-size') || '23';
        text.setAttribute('font-size', Math.max(Number(text.dataset.originalFont), (mobile ? 16 : 18) / scale));
      });
      element.querySelectorAll('[data-s4-fit-width]').forEach(text => {
        text.removeAttribute('textLength');
        const available = Number(text.dataset.s4FitWidth);
        if (text.getComputedTextLength() > available) {
          text.setAttribute('textLength', available);
          text.setAttribute('lengthAdjust', 'spacingAndGlyphs');
        }
      });
      element.querySelectorAll('[data-s4-highlight]').forEach(highlight => {
        const text = element.querySelector('#' + highlight.dataset.s4Highlight);
        highlight.setAttribute('width', text.getComputedTextLength());
      });
    });
    if (nodes.secondSvg) {
      nodes.secondSvg.setAttribute('viewBox', mobile ? '40 85 480 500' : '0 0 560 610');
      const secondBox = nodes.secondSvg.getBoundingClientRect();
      const secondScale = Math.max(.01, Math.min(secondBox.width / (mobile ? 480 : 560), secondBox.height / (mobile ? 500 : 610)));
      nodes.secondSvg.querySelectorAll('text').forEach(text => {
        if (!text.dataset.baseFont) text.dataset.baseFont = text.getAttribute('font-size') || text.parentElement.closest('[font-size]')?.getAttribute('font-size') || '20';
        text.setAttribute('font-size', Math.max(Number(text.dataset.baseFont), (mobile ? 16 : 18) / secondScale));
      });
    }
    svg.setAttribute("viewBox", mobile ? "75 0 410 550" : "0 0 560 610");
    const box = svg.getBoundingClientRect();
    const scale = Math.min(box.width / (mobile ? 410 : 560), box.height / (mobile ? 550 : 610));
    const schoolScale = mobile ? Math.max(1, 25 / (34 * scale)) : 1;
    nodes.school.setAttribute("transform", "translate(226 244) scale(" + schoolScale + ") translate(-226 -244)");
    const schoolName = nodes.school.querySelectorAll("text")[1];
    schoolName.setAttribute("font-size", String(Math.max(20, (mobile ? 16 : 20) / (scale * schoolScale))));
    const figure = root.querySelector(".method-figure");
    const stickyHeight = mobile && !shortLandscape.matches ? figure.getBoundingClientRect().height : 0;
    const storyTop = root.querySelector(".method-story").getBoundingClientRect().top - top + y;
    // Browsers quantize scrollTop. Integer boundaries keep a QA jump to 0%
    // from landing a fraction of a pixel in the previous step.
    const starts = steps.map((step, index) => {
      const ownFigure = step.closest('.method-story').querySelector('.method-figure');
      const ownStickyHeight = mobile && !shortLandscape.matches ? ownFigure.getBoundingClientRect().height : 0;
      return Math.round(index === 0 ? storyTop : step.getBoundingClientRect().top - top + y - ownStickyHeight);
    });
    ranges = steps.map((step, index) => ({
      id: step.dataset.stepId, start: starts[index],
      // A two-pixel hold between steps makes both 100% and the next 0% addressable.
      end: index < steps.length - 1 && steps[index + 1].dataset.stepId.split('-')[0] === step.dataset.stepId.split('-')[0] ? starts[index + 1] - 2 : Math.round(step.getBoundingClientRect().bottom - top + y - (mobile && !shortLandscape.matches ? step.closest('.method-story').querySelector('.method-figure').getBoundingClientRect().height : 0))
    }));
  }

  function update() {
    frame = 0;
    if (!dialog.open || !ranges.length || !nodes) return;
    const y = scroller.scrollTop;
    let index = 0;
    for (let i = 1; i < ranges.length; i += 1) if (y >= ranges[i].start - .1) index = i;
    const range = ranges[index];
    const progress = clamp((y - range.start) / Math.max(1, range.end - range.start));
    let currentScene = '1';
    let inSceneTitle = y < ranges[0].start - .1;
    root.querySelectorAll('[data-scene-title]').forEach(title => {
      const top = title.getBoundingClientRect().top - scroller.getBoundingClientRect().top + y;
      if (y >= top) {
        currentScene = title.dataset.sceneTitle;
        const firstRange = ranges.find(item => item.id === currentScene + '-1');
        inSceneTitle = y < firstRange.start;
      }
    });
    root.querySelector('.method-scene-indicator').textContent = 'SCENE ' + currentScene.padStart(2, '0');
    activeIndex = index;
    activeProgress = progress;
    renderAt(y < ranges[0].start ? 0 : index + progress);
    steps.forEach((step, i) => {
      step.dataset.active = String(i === index && y >= ranges[0].start);
      if (i === index && y >= ranges[0].start) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    });
    if (qa) {
      const titleScreen = inSceneTitle;
      qa.querySelector("[data-qa-id]").textContent = titleScreen ? 'TITLE ' + currentScene.padStart(2, '0') : "#" + range.id;
      qa.querySelector("[data-qa-progress]").textContent = Math.round(progress * 100) + "%";
      qa.querySelector("meter").value = progress;
      qa.querySelector("select").value = range.id;
      qa.querySelector("[data-qa-phase]").textContent = titleScreen ? "제목" : progress < .499 ? "변화 구간" : "유지 구간";
      qa.querySelector('[data-qa-action="prev"]').disabled = index === 0;
      qa.querySelector('[data-qa-action="next"]').disabled = index === ranges.length - 1;
      qa.dataset.stepId = titleScreen ? "title" : range.id;
      qa.dataset.progress = progress.toFixed(5);
    }
  }
  function requestUpdate() { if (!frame) frame = win.requestAnimationFrame(update); }

  function refresh() {
    if (!dialog.open || !nodes || destroyed) return;
    measure();
    trigger?.refresh();
    update();
  }
  function requestRefresh() {
    if (refreshFrame) win.cancelAnimationFrame(refreshFrame);
    refreshFrame = win.requestAnimationFrame(() => { refreshFrame = 0; refresh(); });
  }
  function startScroll() {
    measure();
    if (!fallback && !trigger) {
      // Only this trigger is owned by this instance. No global defaults, pin or snap.
      trigger = win.ScrollTrigger.create({
        id: "method-scene-01", scroller, trigger: root.querySelector(".method-story"),
        start: () => ranges[0].start, end: () => ranges[ranges.length - 1].end,
        onUpdate: update, onRefresh: update
      });
    }
    trigger?.enable(false, false);
    refresh();
  }

  function jump(id, progress = 0) {
    if (!dialog.open || !ranges.length) return;
    const range = ranges.find((item) => item.id === id);
    if (!range) return;
    scroller.scrollTop = range.start + clamp(progress) * (range.end - range.start);
    // ScrollTrigger and QA both render from the real scrollTop; no separate QA state.
    trigger?.update();
    update();
  }

  function createQA() {
    if (!qaEnabled) return;
    qa = doc.createElement("aside");
    qa.className = "method-qa";
    qa.setAttribute("aria-label", "장면 QA 패널");
    qa.innerHTML = '<div class="method-qa-title"><strong>QA <output data-qa-id>TITLE</output> · <output data-qa-progress>0%</output></strong><button type="button" class="method-qa-collapse" aria-label="QA 패널 접기" aria-expanded="true">−</button></div><div class="method-qa-detail"><span class="method-qa-phase" data-qa-phase>제목</span><meter min="0" max="1" value="0" aria-label="현재 단계 진행률"></meter><div class="method-qa-row"><select aria-label="이동할 단계"><option value="1-1">#1-1 통학구역</option><option value="1-2">#1-2 집계구</option><option value="1-3">#1-3 학령인구</option></select><button type="button" data-qa-action="go">이동</button></div><div class="method-qa-row"><button type="button" data-qa-action="prev">이전</button><button type="button" data-qa-action="next">다음</button></div><div class="method-qa-row"><button type="button" data-qa-progress-target="0">0%</button><button type="button" data-qa-progress-target="0.5">50%</button><button type="button" data-qa-progress-target="1">100%</button></div></div>';
    dialog.append(qa);
    const select = qa.querySelector('select');
    select.replaceChildren(...steps.map(step => { const option = doc.createElement('option'); option.value = step.dataset.stepId; option.textContent = '#' + step.dataset.stepId + ' ' + step.querySelector('.method-step-number').textContent.split('/')[1].trim(); return option; }));
    listen(qa, "click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.classList.contains("method-qa-collapse")) {
        const collapsed = qa.dataset.collapsed !== "true";
        qa.dataset.collapsed = String(collapsed);
        button.textContent = collapsed ? "+" : "−";
        button.setAttribute("aria-expanded", String(!collapsed));
        button.setAttribute("aria-label", collapsed ? "QA 패널 펼치기" : "QA 패널 접기");
      } else if (button.dataset.qaProgressTarget !== undefined) jump(ranges[activeIndex].id, Number(button.dataset.qaProgressTarget));
      else if (button.dataset.qaAction === "go") jump(qa.querySelector("select").value);
      else if (button.dataset.qaAction === "prev") jump(ranges[Math.max(0, activeIndex - 1)].id);
      else if (button.dataset.qaAction === "next") jump(ranges[Math.min(ranges.length - 1, activeIndex + 1)].id);
    });
  }

  function lockBackground() {
    const body = doc.body;
    const properties = ["position", "top", "left", "width", "overflow"];
    savedBody = { x: win.scrollX, y: win.scrollY, hadStyle: body.hasAttribute("style"), styles: properties.map((key) => [key, body.style.getPropertyValue(key), body.style.getPropertyPriority(key)]) };
    body.style.position = "fixed";
    body.style.top = -savedBody.y + "px";
    body.style.left = -savedBody.x + "px";
    body.style.width = "100%";
    body.style.overflow = "hidden";
  }
  function restoreBackground() {
    if (!savedBody) return;
    const saved = savedBody;
    savedBody = null;
    saved.styles.forEach(([key, value, priority]) => value ? doc.body.style.setProperty(key, value, priority) : doc.body.style.removeProperty(key));
    if (!saved.hadStyle && !doc.body.style.length) doc.body.removeAttribute("style");
    win.scrollTo({ left: saved.x, top: saved.y, behavior: "instant" });
    if (opener?.isConnected) opener.focus({ preventScroll: true });
  }
  async function open(source) {
    if (destroyed) throw new Error("Method instance was destroyed");
    if (dialog.open) return;
    opener = source instanceof win.HTMLElement ? source : doc.activeElement;
    lockBackground();
    try { dialog.showModal(); } catch (error) { restoreBackground(); throw error; }
    scroller.scrollTop = 0;
    root.querySelector(".method-close").focus({ preventScroll: true });
    if (!readyPromise) readyPromise = prepare().catch((error) => {
      if (!destroyed) { mount.setAttribute("aria-busy", "false"); message("도식을 불러오지 못했습니다. HTTP 서버 실행과 img/scene-01.svg 경로를 확인해주세요."); }
      readyPromise = null;
      if (qaEnabled) console.error("[method-tab]", error);
      return false;
    });
    await readyPromise;
    if (!destroyed && dialog.open && nodes) startScroll();
  }
  function close() { if (dialog.open) dialog.close(); }
  function destroy() {
    if (destroyed) return;
    close();
    restoreBackground();
    destroyed = true;
    abort.abort();
    resizeObserver?.disconnect();
    trigger?.kill();
    timeline?.kill();
    motion.removeEventListener("change", requestRefresh);
    win.cancelAnimationFrame(frame);
    win.cancelAnimationFrame(refreshFrame);
    qa?.remove();
    mount.replaceChildren();
    root.querySelectorAll('.method-svg-mount-second, .method-later-mount').forEach(element => element.replaceChildren());
    delete root[instanceKey];
  }
  listen(root.querySelector(".method-close"), "click", close);
  listen(dialog, "cancel", (event) => { event.preventDefault(); close(); });
  listen(dialog, "close", () => { if (!dialog.open) { trigger?.disable(false); restoreBackground(); } });
  listen(scroller, "scroll", requestUpdate, { passive: true });
  listen(win, "resize", requestRefresh, { passive: true });
  motion.addEventListener("change", requestRefresh);
  if (win.ResizeObserver) {
    resizeObserver = new win.ResizeObserver(requestRefresh);
    resizeObserver.observe(scroller);
    steps.forEach((step) => resizeObserver.observe(step));
  }
  doc.fonts?.ready.then(() => { if (!destroyed) requestRefresh(); });
  createQA();
  const controller = { open, close, refresh, destroy };
  if (qaEnabled) controller.inspect = () => ({
    open: dialog.open, stepId: ranges[activeIndex]?.id, progress: activeProgress,
    scrollTop: scroller.scrollTop, ranges: ranges.map((range) => ({ ...range })),
    svg: { ...debugState }, reducedMotion: motion.matches, fallback,
    triggerCount: trigger ? 1 : 0
  });
  root[instanceKey] = controller;
  return controller;
}
