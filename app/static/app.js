const draftBtn = document.getElementById("draftBtn");
const copyBtn = document.getElementById("copyBtn");
const ticketEl = document.getElementById("ticket");
const toneEl = document.getElementById("tone");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const loadingEl = document.getElementById("loading");
const tagsEl = document.getElementById("tags");
const draftOutputEl = document.getElementById("draftOutput");
const notesEl = document.getElementById("notes");
const sourcesEl = document.getElementById("sources");

async function generateDraft() {
  const ticket_text = ticketEl.value.trim();
  errorEl.textContent = "";

  if (!ticket_text) {
    errorEl.textContent = "Paste a ticket first.";
    return;
  }

  resultEl.hidden = true;
  loadingEl.hidden = false;
  draftBtn.disabled = true;

  try {
    const res = await fetch("/api/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticket_text, tone: toneEl.value }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    loadingEl.hidden = true;
    draftBtn.disabled = false;
  }
}

function renderResult(data) {
  tagsEl.innerHTML = "";
  [
    { label: data.category, cls: "" },
    { label: `urgency: ${data.urgency}`, cls: `urgency-${data.urgency}` },
    { label: `sentiment: ${data.sentiment}`, cls: "" },
  ].forEach(({ label, cls }) => {
    const span = document.createElement("span");
    span.className = `tag ${cls}`;
    span.textContent = label;
    tagsEl.appendChild(span);
  });

  draftOutputEl.value = data.draft_reply;

  notesEl.textContent = data.notes_for_agent
    ? `Note for agent: ${data.notes_for_agent}`
    : "";

  sourcesEl.innerHTML = "";
  data.retrieved_sources.forEach((s) => {
    const li = document.createElement("li");
    const used = data.sources_used.includes(s.id) ? "✓ cited" : "retrieved, not cited";
    li.textContent = `[${s.id}] ${s.title} — relevance ${s.score} — ${used}`;
    sourcesEl.appendChild(li);
  });

  resultEl.hidden = false;
}

draftBtn.addEventListener("click", generateDraft);
copyBtn.addEventListener("click", () => {
  draftOutputEl.select();
  document.execCommand("copy");
  copyBtn.textContent = "Copied!";
  setTimeout(() => (copyBtn.textContent = "Copy to clipboard"), 1500);
});
