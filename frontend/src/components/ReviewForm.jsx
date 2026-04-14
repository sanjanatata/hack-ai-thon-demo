import { useState } from "react";
import "./ReviewForm.css";

const STAR_LABELS = ["Terrible", "Poor", "Average", "Good", "Excellent"];

export default function ReviewForm({ onSubmit }) {
  const [stars, setStars] = useState(4);
  const [hovered, setHovered] = useState(0);
  const [text, setText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    await onSubmit(text, stars);
    setSubmitting(false);
  }

  const displayStars = hovered || stars;

  return (
    <div className="review-form-wrap">
      <div className="form-header">
        <h1>Leave a Review</h1>
        <p>Share your experience — we'll ask 1–2 quick follow-ups to fill key info gaps.</p>
      </div>

      <form className="review-form" onSubmit={handleSubmit}>
        {/* Star rating */}
        <div className="star-section">
          <label className="field-label">Overall rating</label>
          <div className="stars-row" role="group" aria-label="Star rating">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                type="button"
                className={`star-btn ${n <= displayStars ? "filled" : ""}`}
                onClick={() => setStars(n)}
                onMouseEnter={() => setHovered(n)}
                onMouseLeave={() => setHovered(0)}
                aria-label={`${n} star${n !== 1 ? "s" : ""}`}
              >
                ★
              </button>
            ))}
          </div>
          <div className="star-label">{STAR_LABELS[displayStars - 1]}</div>
        </div>

        {/* Review text */}
        <div className="text-section">
          <label className="field-label" htmlFor="review-text">
            Your review <span className="optional">(optional — but helps personalize follow-up questions)</span>
          </label>
          <textarea
            id="review-text"
            className="review-textarea"
            placeholder="Tell future guests what to expect…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
          />
        </div>

        <button
          type="submit"
          className="submit-btn"
          disabled={submitting}
        >
          {submitting ? "Analyzing…" : "Submit Review"}
        </button>
      </form>
    </div>
  );
}
