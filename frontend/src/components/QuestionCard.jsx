import { useState, useRef, useEffect } from "react";
import "./QuestionCard.css";

const STAR_VALS = ["1", "2", "3", "4", "5"];

export default function QuestionCard({ question, onAnswer, onSkip }) {
  const [selectedOption, setSelectedOption] = useState(null);
  const [freeText, setFreeText] = useState("");
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [useText, setUseText] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    setSelectedOption(null);
    setFreeText("");
    setTranscript("");
    setListening(false);
    setUseText(false);
  }, [question]);

  const isClosed =
    question.answer_format === "binary" || question.answer_format === "rating_scale";
  const isRating = question.answer_format === "rating_scale";
  const isShortText = question.answer_format === "short_text";

  // Voice input
  function startListening() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setUseText(true);
      return;
    }
    const rec = new SpeechRecognition();
    rec.continuous = false;
    rec.interimResults = true;
    rec.lang = "en-US";
    recognitionRef.current = rec;

    rec.onstart = () => setListening(true);
    rec.onend = () => setListening(false);
    rec.onresult = (e) => {
      const t = Array.from(e.results)
        .map((r) => r[0].transcript)
        .join("");
      setTranscript(t);
      setFreeText(t);
    };
    rec.onerror = () => {
      setListening(false);
      setUseText(true);
    };
    rec.start();
  }

  function stopListening() {
    recognitionRef.current?.stop();
    setListening(false);
  }

  function handleSubmit() {
    if (isClosed && selectedOption) {
      onAnswer(selectedOption);
    } else if ((isShortText || useText) && freeText.trim()) {
      onAnswer(freeText.trim());
    }
  }

  const canSubmit = isClosed
    ? !!selectedOption
    : !!(freeText.trim() || transcript.trim());

  return (
    <div className="question-card">
      <div className="question-meta">
        <span className={`gap-badge gap-${question.gap_type}`}>
          {gapLabel(question.gap_type)}
        </span>
        <span className="topic-badge">{topicLabel(question.gap_topic)}</span>
      </div>

      <h2 className="question-text">{question.question_text}</h2>

      {/* Closed question: tap buttons or stars */}
      {isClosed && question.options && (
        <div className={isRating ? "star-options" : "option-buttons"}>
          {isRating ? (
            <div className="rating-row">
              {STAR_VALS.map((v) => (
                <button
                  key={v}
                  className={`star-opt ${selectedOption === v ? "selected" : ""}`}
                  onClick={() => setSelectedOption(v)}
                  aria-label={`${v} star`}
                >
                  ★
                </button>
              ))}
              {selectedOption && (
                <span className="rating-label">{selectedOption} / 5</span>
              )}
            </div>
          ) : (
            question.options.map((opt) => (
              <button
                key={opt}
                className={`opt-btn ${selectedOption === opt ? "selected" : ""}`}
                onClick={() => setSelectedOption(opt)}
              >
                {opt}
              </button>
            ))
          )}
        </div>
      )}

      {/* Open question: voice first, text fallback */}
      {isShortText && !useText && (
        <div className="voice-section">
          {!listening ? (
            <button className="mic-btn" onClick={startListening}>
              🎙 Speak your answer
            </button>
          ) : (
            <button className="mic-btn listening" onClick={stopListening}>
              ⏹ Stop recording
            </button>
          )}
          {transcript && (
            <div className="transcript-preview">
              <p>{transcript}</p>
              <button className="clear-transcript" onClick={() => { setTranscript(""); setFreeText(""); }}>
                Clear
              </button>
            </div>
          )}
          <button className="text-fallback" onClick={() => setUseText(true)}>
            Type instead
          </button>
        </div>
      )}

      {isShortText && useText && (
        <div className="text-input-section">
          <textarea
            className="free-text-input"
            placeholder="Share your thoughts (2–3 sentences max)…"
            value={freeText}
            onChange={(e) => setFreeText(e.target.value)}
            rows={3}
            maxLength={300}
          />
          <button className="voice-fallback" onClick={() => { setUseText(false); setFreeText(""); }}>
            Use voice instead
          </button>
        </div>
      )}

      {/* Actions */}
      <div className="card-actions">
        <button className="skip-btn" onClick={onSkip}>
          Skip
        </button>
        {canSubmit && (
          <button className="answer-btn" onClick={handleSubmit}>
            Submit answer →
          </button>
        )}
      </div>

      <div className="question-source">
        {question.source === "llm" ? "✦ AI-generated question" : "Template question"}
      </div>
    </div>
  );
}

function gapLabel(type) {
  const labels = {
    policy_contradiction: "⚠ Policy contradiction",
    zero_fill: "● 0% filled",
    sparse_fill: "◐ Sparse data",
    score_drift: "↘ Score drift",
    staleness: "⏱ Stale data",
  };
  return labels[type] || type;
}

function topicLabel(topic) {
  const labels = {
    pets: "Pets",
    affordability: "Value for money",
    service_checkin: "Service & Check-in",
    location_transportation: "Location",
    ambiance_decor: "Room quality",
    amenities_food: "Amenities & Food",
    cleanliness: "Cleanliness",
    accessibility: "Accessibility",
  };
  return labels[topic] || topic;
}
