import { useState, useRef, useEffect } from "react";
import "./QuestionCard.css";
import { transcribeAudio } from "../api";

const STAR_VALS = ["1", "2", "3", "4", "5"];

export default function QuestionCard({ question, onAnswer, onSkip }) {
  const [selectedOption, setSelectedOption] = useState(null);
  /** Optional elaboration for star ratings (LLM often asks open-ended follow-ups in the same sentence). */
  const [ratingDetail, setRatingDetail] = useState("");
  const [freeText, setFreeText] = useState("");
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [useText, setUseText] = useState(false);
  const recognitionRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const [transcribing, setTranscribing] = useState(false);

  useEffect(() => {
    setSelectedOption(null);
    setRatingDetail("");
    setFreeText("");
    setTranscript("");
    setListening(false);
    setUseText(false);
    setTranscribing(false);
  }, [question]);

  const isClosed =
    question.answer_format === "binary" || question.answer_format === "rating_scale";
  const isRating = question.answer_format === "rating_scale";
  const isShortText = question.answer_format === "short_text";

  // Voice input (MediaRecorder -> backend OpenAI STT)
  async function startListening() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorderRef.current = rec;

      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstart = () => setListening(true);
      rec.onstop = async () => {
        setListening(false);
        // stop tracks
        try {
          streamRef.current?.getTracks()?.forEach((t) => t.stop());
        } catch {
          // ignore
        }
        streamRef.current = null;

        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (!blob.size) return;
        setTranscribing(true);
        try {
          const { text } = await transcribeAudio(blob, "answer.webm");
          const t = (text || "").trim();
          setTranscript(t);
          setFreeText(t);
        } catch (e) {
          console.error(e);
          setUseText(true);
        } finally {
          setTranscribing(false);
        }
      };

      rec.start();
    } catch (e) {
      console.error(e);
      setUseText(true);
    }
  }

  function stopListening() {
    try {
      recorderRef.current?.stop();
    } catch {
      // ignore
    }
    setListening(false);
  }

  function handleSubmit() {
    if (isClosed && selectedOption) {
      if (isRating && ratingDetail.trim()) {
        onAnswer(`${selectedOption}/5 — ${ratingDetail.trim()}`);
      } else {
        onAnswer(selectedOption);
      }
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
            <div className="rating-block">
              <div className="rating-row">
                {STAR_VALS.map((v) => (
                  <button
                    key={v}
                    type="button"
                    className={`star-opt ${selectedOption && parseInt(v) <= parseInt(selectedOption) ? "selected" : ""}`}
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
              <label className="rating-detail-label" htmlFor="rating-detail">
                Add a few words (optional)
              </label>
              <textarea
                id="rating-detail"
                className="rating-detail-input"
                rows={3}
                maxLength={500}
                placeholder="e.g. what stood out, or anything future guests should know…"
                value={ratingDetail}
                onChange={(e) => setRatingDetail(e.target.value)}
                aria-describedby="rating-detail-hint"
              />
              <p id="rating-detail-hint" className="rating-detail-hint">
                Stars are enough to submit; add detail here when the question asks for it.
              </p>
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
            <button className="mic-btn" onClick={startListening} disabled={transcribing}>
              {transcribing ? "Transcribing…" : "Speak your answer"}
            </button>
          ) : (
            <button className="mic-btn listening" onClick={stopListening}>
              Stop recording
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
