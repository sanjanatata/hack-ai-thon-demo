const BASE = import.meta.env.VITE_API_URL || "/api";

export async function fetchProperties() {
  const r = await fetch(`${BASE}/properties`);
  return r.json();
}

export async function fetchProperty(id) {
  const r = await fetch(`${BASE}/properties/${id}`);
  return r.json();
}

export async function generateQuestions(propertyId, reviewText, starRating) {
  const r = await fetch(`${BASE}/properties/${propertyId}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_text: reviewText, rating: starRating }),
  });
  return r.json();
}

export async function submitAnswer(propertyId, gapTopic, gapType, answer, reviewText) {
  const r = await fetch(`${BASE}/properties/${propertyId}/answers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gap_topic: gapTopic, gap_type: gapType, answer, review_text: reviewText }),
  });
  return r.json();
}

export async function logSkip(data) {
  await fetch(`${BASE}/skip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function transcribeAudio(blob, filename = "audio.webm") {
  const fd = new FormData();
  fd.append("file", blob, filename);
  const r = await fetch(`${BASE}/transcribe`, { method: "POST", body: fd });
  if (!r.ok) {
    let detail = "";
    try {
      const j = await r.json();
      detail = j?.detail ? `: ${j.detail}` : "";
    } catch {
      // ignore
    }
    throw new Error(`Transcription failed (${r.status})${detail}`);
  }
  return r.json();
}
