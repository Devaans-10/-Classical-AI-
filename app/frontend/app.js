/**
 * AID201A Classical AI Document QA - Frontend Application Module
 * High-craft technical research console interface
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    docName: 'handbook.md',
    facts: [],
    rules: [],
    sections: [],
    schema: null,
    activeTraceData: null,
    chatHistory: [],
    searchSectionTerm: '',
  };

  // DOM Elements - Header
  const headerDocName = document.getElementById('headerDocName');
  const headerFactCount = document.getElementById('headerFactCount');
  const heroDocTag = document.getElementById('heroDocTag');

  // DOM Elements - Left Panel
  const summaryFacts = document.getElementById('summaryFacts');
  const summaryRules = document.getElementById('summaryRules');
  const summarySections = document.getElementById('summarySections');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadProgress = document.getElementById('uploadProgress');
  const progressFill = document.getElementById('progressFill');
  const btnReloadSample = document.getElementById('btnReloadSample');

  const segmentedTabs = document.querySelectorAll('.segmented-tab');
  const segmentedGlider = document.getElementById('segmentedGlider');
  const sectionSearchInput = document.getElementById('sectionSearchInput');
  const sectionList = document.getElementById('sectionList');
  const factsFilter = document.getElementById('factsFilter');
  const factsTableBody = document.getElementById('factsTableBody');
  const rulesList = document.getElementById('rulesList');
  const schemaInfo = document.getElementById('schemaInfo');

  // DOM Elements - Centre Panel (Chat)
  const chatHeroEmpty = document.getElementById('chatHeroEmpty');
  const chatPillsRow = document.getElementById('chatPillsRow');
  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const questionInput = document.getElementById('questionInput');
  const btnClearChat = document.getElementById('btnClearChat');
  const suggestCards = document.querySelectorAll('.suggest-card');
  const pillChips = document.querySelectorAll('.pill-chip');

  // DOM Elements - Right Panel (Trace)
  const emptyTrace = document.getElementById('emptyTrace');
  const activeTrace = document.getElementById('activeTrace');
  const traceHeaderAction = document.getElementById('traceHeaderAction');
  const btnCopyTrace = document.getElementById('btnCopyTrace');
  const traceQuestion = document.getElementById('traceQuestion');
  const traceStatusBadge = document.getElementById('traceStatusBadge');
  const traceTemplateTag = document.getElementById('traceTemplateTag');
  const traceTimeTag = document.getElementById('traceTimeTag');
  const traceConfTag = document.getElementById('traceConfTag');
  const traceStepCountBadge = document.getElementById('traceStepCountBadge');
  const traceStepsTimeline = document.getElementById('traceStepsTimeline');
  const citationsList = document.getElementById('citationsList');

  // DOM Elements - Evaluation Modal
  const evalModal = document.getElementById('evalModal');
  const btnOpenEval = document.getElementById('btnOpenEval');
  const btnMobileEval = document.getElementById('btnMobileEval');
  const btnCloseEvalModal = document.getElementById('btnCloseEvalModal');
  const btnCloseEvalBtn = document.getElementById('btnCloseEvalBtn');
  const btnRerunEval = document.getElementById('btnRerunEval');
  const evalAccuracy = document.getElementById('evalAccuracy');
  const evalCorrectRatio = document.getElementById('evalCorrectRatio');
  const evalPrecisionRecall = document.getElementById('evalPrecisionRecall');
  const evalF1 = document.getElementById('evalF1');
  const evalInferenceAcc = document.getElementById('evalInferenceAcc');
  const evalAbstention = document.getElementById('evalAbstention');
  const ablAccGainText = document.getElementById('ablAccGainText');
  const ablInfGainText = document.getElementById('ablInfGainText');
  const ablWithAccBar = document.getElementById('ablWithAccBar');
  const ablWithoutAccBar = document.getElementById('ablWithoutAccBar');
  const ablWithInfBar = document.getElementById('ablWithInfBar');
  const ablWithoutInfBar = document.getElementById('ablWithoutInfBar');
  const evalTableBody = document.getElementById('evalTableBody');

  // DOM Elements - Mobile Navigation
  const mobileTabBtns = document.querySelectorAll('.mobile-tab-btn');

  // ------------------------------------------------------------------------ //
  // Boot & Initialization                                                    //
  // ------------------------------------------------------------------------ //
  async function initApp() {
    await fetchStats();
    await fetchSections();
    await fetchFacts();
    await fetchRules();
    await fetchSchema();
    setupMobilePanels();
  }

  // ------------------------------------------------------------------------ //
  // Number Count-Up Animation Utility                                        //
  // ------------------------------------------------------------------------ //
  function animateCount(element, target, duration = 800) {
    if (!element) return;
    const start = 0;
    const startTime = performance.now();
    function update(time) {
      const elapsed = time - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const current = Math.floor(start + (target - start) * ease);
      element.textContent = current;
      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.textContent = target;
      }
    }
    requestAnimationFrame(update);
  }

  // ------------------------------------------------------------------------ //
  // API Fetchers                                                             //
  // ------------------------------------------------------------------------ //
  async function fetchStats() {
    try {
      const res = await fetch('/api/kb/stats');
      const data = await res.json();
      state.docName = data.doc_name;

      if (headerDocName) headerDocName.textContent = data.doc_name;
      if (headerFactCount) headerFactCount.textContent = `${data.facts} facts`;
      if (heroDocTag) heroDocTag.textContent = data.doc_name;

      animateCount(summaryFacts, data.facts);
      animateCount(summaryRules, data.rules);
      animateCount(summarySections, data.sections);
    } catch (e) {
      console.error('Failed to load KB stats:', e);
    }
  }

  async function fetchSections() {
    try {
      const res = await fetch('/api/kb/sections');
      const data = await res.json();
      state.sections = data.sections || [];
      renderSections();
    } catch (e) {
      console.error('Failed to load sections:', e);
    }
  }

  async function fetchFacts(query = '') {
    try {
      const res = await fetch(`/api/kb/facts?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      state.facts = data.facts || [];
      renderFacts();
    } catch (e) {
      console.error('Failed to load facts:', e);
    }
  }

  async function fetchRules() {
    try {
      const res = await fetch('/api/kb/rules');
      const data = await res.json();
      state.rules = data.rules || [];
      renderRules();
    } catch (e) {
      console.error('Failed to load rules:', e);
    }
  }

  async function fetchSchema() {
    try {
      const res = await fetch('/api/kb/schema');
      state.schema = await res.json();
      renderSchema();
    } catch (e) {
      console.error('Failed to load schema:', e);
    }
  }

  // ------------------------------------------------------------------------ //
  // Section Content Formatter (Transforms raw markdown to clean KV rows)     //
  // ------------------------------------------------------------------------ //
  function formatSectionContent(rawLines) {
    if (!rawLines) return '';
    const lines = typeof rawLines === 'string' ? rawLines.split('\n') : rawLines;
    const kvPairs = [];
    const clauses = [];
    const tableRows = [];
    const generalLines = [];

    const kvRegex = /^\s*(?:\*\*)?([^:*]{2,50})(?:\*\*)?\s*:\s*(.+)$/;
    const clauseRegex = /^\s*(?:\*\*)?Clause\s+([\d.]+)(?:\*\*)?\s*[:]\s*(.+)$/i;
    const tableRowRegex = /^\s*\|(.+)\|\s*$/;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // Table separator row
      if (/^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(trimmed)) {
        continue;
      }

      // Check Table Row
      const tableMatch = trimmed.match(tableRowRegex);
      if (tableMatch) {
        const cells = tableMatch[1].split('|').map((c) => c.trim()).filter(Boolean);
        if (cells.length > 0) {
          tableRows.push(cells);
        }
        continue;
      }

      // Check Clause
      const clauseMatch = trimmed.match(clauseRegex);
      if (clauseMatch) {
        clauses.push({
          id: clauseMatch[1],
          text: clauseMatch[2].replace(/\*\*/g, '').trim(),
        });
        continue;
      }

      // Check Key/Value
      const kvMatch = trimmed.match(kvRegex);
      if (kvMatch) {
        kvPairs.push({
          key: kvMatch[1].replace(/\*\*/g, '').trim(),
          val: kvMatch[2].replace(/\*\*/g, '').trim(),
        });
        continue;
      }

      // Ordinary line
      generalLines.push(trimmed.replace(/\*\*/g, ''));
    }

    let html = '';

    // Render Table if present
    if (tableRows.length >= 2) {
      const header = tableRows[0];
      const rows = tableRows.slice(1);
      html += `
        <div style="overflow-x:auto; margin:6px 0 10px 0;">
          <table class="triples-table" style="font-size:0.68rem;">
            <thead>
              <tr>${header.map((h) => `<th>${escapeHtml(h)}</th>`).join('')}</tr>
            </thead>
            <tbody>
              ${rows.map((r) => `<tr>${r.map((c) => `<td>${escapeHtml(c)}</td>`).join('')}</tr>`).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    // Render Key/Value Pairs
    if (kvPairs.length > 0) {
      html += `
        <table class="kv-rows-table">
          <tbody>
            ${kvPairs.map((kv) => `
              <tr class="kv-row">
                <td class="kv-key">${escapeHtml(kv.key)}</td>
                <td class="kv-val">${escapeHtml(kv.val)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }

    // Render Clauses
    if (clauses.length > 0) {
      html += clauses.map((c) => `
        <div class="clause-block">
          <div class="clause-tag">Clause ${escapeHtml(c.id)}</div>
          <div>${escapeHtml(c.text)}</div>
        </div>
      `).join('');
    }

    // Render General text if nothing else matched
    if (html === '' && generalLines.length > 0) {
      html = `<div style="font-size:0.72rem; color:var(--text-muted); line-height:1.4;">${generalLines.map(escapeHtml).join('<br>')}</div>`;
    }

    return html;
  }

  function getSectionBorderClass(heading) {
    const h = heading.toLowerCase();
    if (h.includes('general') || h.includes('contact')) return 'border-info';
    if (h.includes('attendance') || h.includes('hostel') || h.includes('library')) return 'border-policy';
    if (h.includes('fee')) return 'border-fees';
    if (h.includes('exam') || h.includes('grading')) return 'border-exam';
    if (h.includes('table') || h.includes('date')) return 'border-schedule';
    return 'border-info';
  }

  // ------------------------------------------------------------------------ //
  // Render Left Panel Components                                             //
  // ------------------------------------------------------------------------ //
  function renderSections() {
    sectionList.innerHTML = '';
    const term = state.searchSectionTerm.toLowerCase().trim();

    const filtered = state.sections.filter((sec) => {
      if (!term) return true;
      return sec.heading.toLowerCase().includes(term) || sec.full_text.toLowerCase().includes(term);
    });

    if (filtered.length === 0) {
      sectionList.innerHTML = '<div style="text-align:center; padding:16px; color:var(--text-dim); font-size:0.75rem;">No matching sections found.</div>';
      return;
    }

    filtered.forEach((sec, idx) => {
      const card = document.createElement('div');
      const borderClass = getSectionBorderClass(sec.heading);
      const isInitiallyOpen = idx === 0 || filtered.length === 1;

      card.className = `section-card ${borderClass} ${isInitiallyOpen ? 'open' : ''}`;
      card.innerHTML = `
        <div class="section-card-header">
          <div class="section-card-title">
            <svg class="section-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
            <span>${escapeHtml(sec.heading)}</span>
          </div>
          <span class="section-meta-chip">${sec.line_count} lines</span>
        </div>
        <div class="section-card-body">
          ${formatSectionContent(sec.full_text)}
        </div>
      `;

      card.querySelector('.section-card-header').addEventListener('click', () => {
        card.classList.toggle('open');
      });

      sectionList.appendChild(card);
    });
  }

  function renderFacts() {
    factsTableBody.innerHTML = '';
    if (state.facts.length === 0) {
      factsTableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:16px; color:var(--text-dim);">No matching triples found.</td></tr>';
      return;
    }

    state.facts.slice(0, 120).forEach((f) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong style="color:var(--text-primary);">${escapeHtml(f.subject)}</strong></td>
        <td><span style="color:var(--accent-violet);">${escapeHtml(f.predicate)}</span></td>
        <td><span style="color:var(--accent-cyan);">${escapeHtml(f.object)}</span></td>
        <td><span style="color:var(--text-dim); font-size:0.65rem;">${escapeHtml(f.citation)}</span></td>
      `;
      factsTableBody.appendChild(tr);
    });
  }

  function renderRules() {
    rulesList.innerHTML = '';
    state.rules.forEach((r) => {
      const card = document.createElement('div');
      card.className = 'rule-card';

      const bodyStr = r.body.map((b) => `(${b.subject}, ${b.predicate}, ${b.object})`).join(' ∧ ');
      const headStr = `(${r.head.subject}, ${r.head.predicate}, ${r.head.object})`;

      card.innerHTML = `
        <div class="rule-card-header">
          <span class="rule-title">${escapeHtml(r.name)}</span>
          <label class="rule-switch">
            <input type="checkbox" ${r.enabled ? 'checked' : ''} data-rule="${escapeHtml(r.name)}">
            <span class="rule-slider"></span>
          </label>
        </div>
        <div class="rule-code-preview">
          ${escapeHtml(bodyStr)}<br>➔ <strong style="color:var(--accent-emerald);">${escapeHtml(headStr)}</strong>
        </div>
        <div class="rule-source-tag">
          Provenance: ${escapeHtml(r.provenance ? `${r.provenance.section} ${r.provenance.clause || ''}` : 'rules.json')}
        </div>
      `;

      const toggleInput = card.querySelector('input[type="checkbox"]');
      toggleInput.addEventListener('change', async (e) => {
        const enabled = e.target.checked;
        await toggleRule(r.name, enabled);
      });

      rulesList.appendChild(card);
    });
  }

  async function toggleRule(ruleName, enabled) {
    try {
      await fetch('/api/kb/rules/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rule_name: ruleName, enabled }),
      });
      await fetchRules();
    } catch (e) {
      console.error('Failed to toggle rule:', e);
    }
  }

  function renderSchema() {
    if (!state.schema) return;
    schemaInfo.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:10px; font-size:0.74rem;">
        <div style="background:var(--surface-card); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); padding:10px 12px;">
          <div style="font-weight:700; color:var(--text-primary); margin-bottom:4px;">Knowledge Representation</div>
          <p style="color:var(--text-muted); line-height:1.45;">Strict classical AI schema storing assertions as relational triples annotated with exact document provenance.</p>
        </div>
        <div style="background:var(--surface-card); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); padding:10px 12px; font-family:var(--font-mono); font-size:0.68rem; line-height:1.6; color:var(--text-secondary);">
          <div><strong style="color:var(--accent-cyan);">Triple:</strong> (subject, predicate, object)</div>
          <div><strong style="color:var(--accent-violet);">Provenance:</strong> (section, clause, page)</div>
          <div><strong style="color:var(--accent-emerald);">Horn Clause:</strong> head :- body conditions</div>
          <div><strong style="color:var(--accent-amber);">Comparison:</strong> cmp:ge, cmp:lt, cmp:le, cmp:gt</div>
        </div>
        <div style="background:var(--surface-card); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); padding:10px 12px;">
          <div style="font-weight:700; color:var(--text-primary); margin-bottom:6px;">Authored Query Templates (${state.schema.components.QueryTemplates.length})</div>
          <div style="display:flex; flex-wrap:wrap; gap:4px;">
            ${state.schema.components.QueryTemplates.map((t) => `<span class="mono-badge" style="background:var(--surface-2); padding:3px 6px; border-radius:4px; border:1px solid var(--border-subtle);">${escapeHtml(t)}</span>`).join('')}
          </div>
        </div>
      </div>
    `;
  }

  // ------------------------------------------------------------------------ //
  // Segmented Tabs Glider Slider                                             //
  // ------------------------------------------------------------------------ //
  segmentedTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      segmentedTabs.forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));

      tab.classList.add('active');
      segmentedGlider.style.transform = `translateX(${index * 100}%)`;

      const tabId = tab.getAttribute('data-tab');
      const targetPane = document.getElementById(`tab${tabId.charAt(0).toUpperCase() + tabId.slice(1)}`);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // Section search filter
  sectionSearchInput.addEventListener('input', (e) => {
    state.searchSectionTerm = e.target.value;
    renderSections();
  });

  // Facts triples filter
  factsFilter.addEventListener('input', (e) => {
    fetchFacts(e.target.value);
  });

  // ------------------------------------------------------------------------ //
  // File Upload Handlers with Progress Animation                             //
  // ------------------------------------------------------------------------ //
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      uploadFile(fileInput.files[0]);
    }
  });

  async function uploadFile(file) {
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!['.md', '.txt', '.docx', '.pdf'].includes(ext)) {
      alert(`Unsupported file format '${ext}'. Please upload .pdf, .docx, .md, or .txt`);
      return;
    }

    uploadProgress.classList.add('active');
    progressFill.style.width = '35%';

    const formData = new FormData();
    formData.append('file', file);

    try {
      progressFill.style.width = '70%';
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      progressFill.style.width = '100%';
      const data = await res.json();

      setTimeout(() => {
        uploadProgress.classList.remove('active');
        progressFill.style.width = '0%';
      }, 450);

      // Refresh KB
      await initApp();

      // Add system message in chat
      addSystemNotification(`Document '${data.doc_name}' parsed successfully: ${data.summary}`);
    } catch (err) {
      uploadProgress.classList.remove('active');
      progressFill.style.width = '0%';
      showToast('Upload error: ' + err.message, 'error');
    }
  }

  btnReloadSample.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/load-sample', { method: 'POST' });
      const data = await res.json();
      await initApp();
      addSystemNotification(`Reset to default course handbook: ${data.summary}`);
    } catch (e) {
      alert('Failed to reset sample handbook');
    }
  });

  // ------------------------------------------------------------------------ //
  // Chat Question Answering & Messaging                                      //
  // ------------------------------------------------------------------------ //
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = questionInput.value.trim();
    if (!q) return;
    questionInput.value = '';
    await askQuestion(q);
  });

  suggestCards.forEach((card) => {
    card.addEventListener('click', async () => {
      const q = card.getAttribute('data-q');
      if (q) {
        await askQuestion(q);
      }
    });
  });

  pillChips.forEach((chip) => {
    chip.addEventListener('click', async () => {
      const q = chip.getAttribute('data-q');
      if (q) {
        await askQuestion(q);
      }
    });
  });

  btnClearChat.addEventListener('click', () => {
    chatMessages.innerHTML = '';
    state.chatHistory = [];
    if (chatHeroEmpty) {
      chatHeroEmpty.style.display = 'flex';
      chatMessages.appendChild(chatHeroEmpty);
    }
    if (chatPillsRow) chatPillsRow.style.display = 'none';
  });

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener('click', toggleTheme);
  }

  async function askQuestion(question) {
    // Hide Hero and Show Pills Row
    if (chatHeroEmpty) chatHeroEmpty.style.display = 'none';
    if (chatPillsRow) chatPillsRow.style.display = 'flex';

    // 1. Render User Question Bubble
    addUserBubble(question);

    // 2. Render Assistant Placeholder
    const placeholderEl = addAssistantPlaceholder();

    try {
      const res = await fetch('/api/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const ans = await res.json();
      state.activeTraceData = ans;

      // 3. Render Assistant Answer Card
      updateAssistantCard(placeholderEl, ans);

      // 4. Update Right Reasoning Trace Panel
      renderTracePanel(ans);
    } catch (err) {
      placeholderEl.innerHTML = `
        <div class="answer-card-v2">
          <div class="verdict-strip strip-abstain">
            <span class="verdict-tag tag-abstain">Error Processing Query</span>
          </div>
          <div class="answer-body">
            <div class="answer-text-large" style="color:var(--accent-rose); font-size:0.95rem;">${escapeHtml(err.message)}</div>
          </div>
        </div>
      `;
    }
  }

  function addUserBubble(text) {
    const item = document.createElement('div');
    item.className = 'chat-item user';
    item.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(item);
    scrollChatBottom();
  }

  function addAssistantPlaceholder() {
    const item = document.createElement('div');
    item.className = 'chat-item assistant';
    item.innerHTML = `
      <div class="answer-card-v2" style="opacity:0.75;">
        <div class="verdict-strip strip-derived">
          <span class="verdict-tag tag-derived">Evaluating Horn Rules...</span>
          <span class="verdict-meta">Backward chaining in progress</span>
        </div>
        <div class="answer-body">
          <div style="font-size:0.85rem; color:var(--text-muted); font-family:var(--font-mono);">Unifying question bindings with Knowledge Base triples...</div>
        </div>
      </div>
    `;
    chatMessages.appendChild(item);
    scrollChatBottom();
    return item;
  }

  function updateAssistantCard(el, ans) {
    const isAnswered = ans.status === 'answered';
    const isDerived = ans.trace && ans.trace.some((t) => t.type === 'derived' || t.type === 'rule');

    let stripClass = 'strip-abstain';
    let tagClass = 'tag-abstain';
    let verdictLabel = 'Not in Document';

    if (isAnswered) {
      if (isDerived) {
        stripClass = 'strip-derived';
        tagClass = 'tag-derived';
        verdictLabel = 'Derived via Horn Chaining';
      } else {
        stripClass = 'strip-answered';
        tagClass = 'tag-answered';
        verdictLabel = 'Answered from Direct Fact';
      }
    }

    const citationsHtml = (ans.citations || [])
      .map((c) => `
        <span class="citation-chip-v2">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path></svg>
          <span>${escapeHtml(c.short || c.section)}</span>
        </span>
      `)
      .join('');

    el.innerHTML = `
      <div class="answer-card-v2">
        <div class="verdict-strip ${stripClass}">
          <span class="verdict-tag ${tagClass}">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              ${isAnswered ? '<polyline points="20 6 9 17 4 12"></polyline>' : '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line>'}
            </svg>
            <span>${verdictLabel}</span>
          </span>
          <span class="verdict-meta">${Math.round(ans.confidence * 100)}% conf · ${ans.response_time_ms.toFixed(1)} ms</span>
        </div>

        <div class="answer-body">
          <div class="answer-text-large">${escapeHtml(ans.answer)}</div>
          ${citationsHtml ? `<div class="citations-chips-wrap">${citationsHtml}</div>` : ''}
        </div>

        <div class="answer-footer-row">
          <span class="template-badge-mono">template: ${escapeHtml(ans.template || 'direct')}</span>
          <div class="answer-actions">
            <button class="btn-copy-action" title="Copy answer">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            </button>
            <button class="btn-trace-action">
              <span>Inspect Reasoning Trace</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
          </div>
        </div>
      </div>
    `;

    // Action: Copy Answer
    const btnCopy = el.querySelector('.btn-copy-action');
    btnCopy.addEventListener('click', () => {
      navigator.clipboard.writeText(ans.answer);
      btnCopy.style.color = 'var(--accent-emerald)';
      setTimeout(() => { btnCopy.style.color = ''; }, 1200);
    });

    // Action: Inspect Trace
    const btnInspect = el.querySelector('.btn-trace-action');
    btnInspect.addEventListener('click', () => {
      renderTracePanel(ans);
      if (window.innerWidth <= 900) {
        switchMobilePanel('panelRight');
      }
    });

    scrollChatBottom();
  }

  function addSystemNotification(text) {
    const item = document.createElement('div');
    item.className = 'chat-item';
    item.innerHTML = `
      <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:var(--radius-sm); padding:10px 14px; text-align:center; font-size:0.75rem; color:var(--text-secondary);">
        ${escapeHtml(text)}
      </div>
    `;
    chatMessages.appendChild(item);
    scrollChatBottom();
  }

  function scrollChatBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ------------------------------------------------------------------------ //
  // Right Panel: Reasoning Trace & Citations Renderer                        //
  // ------------------------------------------------------------------------ //
  function renderTracePanel(ans) {
    if (!ans) return;

    emptyTrace.style.display = 'none';
    activeTrace.style.display = 'flex';
    if (traceHeaderAction) traceHeaderAction.style.display = 'block';

    traceQuestion.textContent = ans.question;

    const isAnswered = ans.status === 'answered';
    traceStatusBadge.className = `badge ${isAnswered ? 'badge-answered' : 'badge-abstain'}`;
    traceStatusBadge.textContent = isAnswered ? 'Answered' : 'Not in Document';

    traceTemplateTag.textContent = `template: ${ans.template || 'direct'}`;
    traceTimeTag.textContent = `${ans.response_time_ms.toFixed(1)} ms`;
    traceConfTag.textContent = `${Math.round(ans.confidence * 100)}% conf`;

    const steps = ans.trace || [];
    traceStepCountBadge.textContent = `${steps.length} steps`;

    // Render Steps Timeline
    traceStepsTimeline.innerHTML = '';
    steps.forEach((step) => {
      const card = document.createElement('div');
      card.className = 'timeline-card';

      const type = step.type || 'lookup';
      const dotClass = `dot-${type}`;
      const typeClass = `type-${type}`;

      card.innerHTML = `
        <div class="timeline-dot ${dotClass}"></div>
        <div class="timeline-inner-card">
          <div class="timeline-step-type-tag ${typeClass}">Step ${step.step_no} · ${type.toUpperCase()}</div>
          <div class="timeline-step-desc-text">${escapeHtml(step.description)}</div>
          ${step.detail ? `<div class="timeline-step-detail-box">${escapeHtml(step.detail)}</div>` : ''}
        </div>
      `;
      traceStepsTimeline.appendChild(card);
    });

    // Render Citations
    citationsList.innerHTML = '';
    const citations = ans.citations || [];
    if (citations.length === 0) {
      citationsList.innerHTML = '<div style="font-size:0.72rem; color:var(--text-dim); padding:6px 0;">No external document citations (abstained or question context).</div>';
    } else {
      citations.forEach((c) => {
        const tile = document.createElement('div');
        tile.className = 'citation-tile';
        tile.innerHTML = `
          <div class="citation-tile-source">${escapeHtml(c.short || c.section)}</div>
          ${c.triple ? `<div class="citation-tile-triple">${escapeHtml(c.triple)}</div>` : ''}
        `;
        citationsList.appendChild(tile);
      });
    }

    // Copy Trace Button Handler
    btnCopyTrace.onclick = () => {
      const traceText = `Q: ${ans.question}\nAnswer: ${ans.answer} (${ans.status})\nTemplate: ${ans.template}\nTime: ${ans.response_time_ms}ms\n\nTrace Steps:\n` +
        steps.map((s) => `[Step ${s.step_no} - ${s.type.toUpperCase()}] ${s.description} ${s.detail ? `(${s.detail})` : ''}`).join('\n') +
        `\n\nCitations:\n` + citations.map((c) => `- ${c.short || c.section}`).join('\n');
      navigator.clipboard.writeText(traceText);
      btnCopyTrace.querySelector('span').textContent = 'Copied!';
      setTimeout(() => { btnCopyTrace.querySelector('span').textContent = 'Copy Trace'; }, 1500);
    };
  }

  // ------------------------------------------------------------------------ //
  // Evaluation Benchmark & Ablation Study Modal                              //
  // ------------------------------------------------------------------------ //
  btnOpenEval.addEventListener('click', openEvalModal);
  btnMobileEval.addEventListener('click', openEvalModal);
  btnCloseEvalModal.addEventListener('click', closeEvalModal);
  btnCloseEvalBtn.addEventListener('click', closeEvalModal);
  btnRerunEval.addEventListener('click', runEvaluationSuite);

  async function openEvalModal() {
    evalModal.style.display = 'flex';
    await runEvaluationSuite();
  }

  function closeEvalModal() {
    evalModal.style.display = 'none';
  }

  async function runEvaluationSuite() {
    btnRerunEval.disabled = true;
    btnRerunEval.innerHTML = '<span>Evaluating 35 Gold Cases...</span>';

    try {
      const res = await fetch('/api/eval');
      const data = await res.json();
      const withRules = data.with_rules;
      const withoutRules = data.without_rules;
      const delta = data.delta;

      evalAccuracy.textContent = `${(withRules.accuracy * 100).toFixed(1)}%`;
      evalCorrectRatio.textContent = `${withRules.correct} / ${withRules.total} test cases correct`;
      evalPrecisionRecall.textContent = `${withRules.precision.toFixed(2)} / ${withRules.recall.toFixed(2)}`;
      evalF1.textContent = `F1 Score: ${withRules.f1.toFixed(2)}`;

      const infAcc = withRules.category_breakdown?.inference?.accuracy || 1.0;
      evalInferenceAcc.textContent = `${(infAcc * 100).toFixed(1)}%`;
      evalAbstention.textContent = `${(withRules.abstention_rate * 100).toFixed(1)}%`;

      // Ablation Progress Bars
      const accGain = (delta.accuracy_gain * 100).toFixed(1);
      ablAccGainText.textContent = `+${accGain}% Net Gain`;
      ablWithAccBar.style.width = `${withRules.accuracy * 100}%`;
      ablWithoutAccBar.style.width = `${withoutRules.accuracy * 100}%`;

      const infDiff = ((delta.inference_accuracy_with_rules - delta.inference_accuracy_without_rules) * 100).toFixed(1);
      ablInfGainText.textContent = `+${infDiff}% Critical Gain`;
      ablWithInfBar.style.width = `${delta.inference_accuracy_with_rules * 100}%`;
      ablWithoutInfBar.style.width = `${delta.inference_accuracy_without_rules * 100}%`;

      // Render Table
      evalTableBody.innerHTML = '';
      withRules.results.forEach((r) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong style="font-family:var(--font-mono);">${escapeHtml(r.id)}</strong></td>
          <td><span class="mono-badge" style="background:var(--surface-2); padding:2px 5px; border-radius:3px;">${escapeHtml(r.category)}</span></td>
          <td>${escapeHtml(r.question)}</td>
          <td style="color:var(--accent-cyan); font-family:var(--font-mono);">${escapeHtml(r.expected)}</td>
          <td style="font-weight:600;">${escapeHtml(r.actual)}</td>
          <td><span class="${r.correct ? 'verdict-pass' : 'verdict-fail'}">${r.correct ? 'PASS' : 'FAIL'}</span></td>
          <td style="font-family:var(--font-mono); color:var(--text-dim); font-size:0.68rem;">${r.response_time_ms} ms</td>
        `;
        evalTableBody.appendChild(tr);
      });
    } catch (e) {
      console.error('Failed to run evaluation suite:', e);
      alert('Error running evaluation suite');
    } finally {
      btnRerunEval.disabled = false;
      btnRerunEval.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="1 4 1 10 7 10"></polyline>
          <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
        </svg>
        <span>Re-run Evaluation Benchmark</span>
      `;
    }
  }

  // ------------------------------------------------------------------------ //
  // Mobile Responsiveness & Bottom Navigation                                //
  // ------------------------------------------------------------------------ //
  function setupMobilePanels() {
    if (window.innerWidth <= 900) {
      switchMobilePanel('panelCenter');
    }
  }

  window.addEventListener('resize', () => {
    if (window.innerWidth > 900) {
      document.querySelectorAll('.panel').forEach((p) => p.classList.remove('mobile-active'));
    } else {
      const activeBtn = document.querySelector('.mobile-tab-btn.active');
      if (activeBtn && activeBtn.getAttribute('data-panel')) {
        switchMobilePanel(activeBtn.getAttribute('data-panel'));
      }
    }
  });

  mobileTabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetPanelId = btn.getAttribute('data-panel');
      if (targetPanelId) {
        mobileTabBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        switchMobilePanel(targetPanelId);
      }
    });
  });

  function switchMobilePanel(panelId) {
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('mobile-active'));
    const target = document.getElementById(panelId);
    if (target) {
      target.classList.add('mobile-active');
    }
  }

  // Utility
  function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Initial Run
  initApp();
});
