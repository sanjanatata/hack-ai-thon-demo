const BASE = "/api";

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
    body: JSON.stringify({ review_text: reviewText, star_rating: starRating }),
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
