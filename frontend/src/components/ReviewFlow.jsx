import { useState } from "react";
import { generateQuestions, submitAnswer, logSkip } from "../api";
import ReviewForm from "./ReviewForm";
import QuestionCard from "./QuestionCard";
import ImpactPanel from "./ImpactPanel";
import QuestionQueue from "./QuestionQueue";
import "./ReviewFlow.css";

const STAGES = {
  FORM: "form",
  LOADING: "loading",
  QUESTION: "question",
  IMPACT: "impact",
};

export default function ReviewFlow({ propertyId }) {
  const [stage, setStage] = useState(STAGES.FORM);
  const [reviewData, setReviewData] = useState({ text: "", stars: 4 });
  const [questionData, setQuestionData] = useState(null); // full API response
  const [currentQIndex, setCurrentQIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [impactData, setImpactData] = useState([]);
  const [showQueue, setShowQueue] = useState(false);

  async function handleReviewSubmit(text, stars) {
    setReviewData({ text, stars });
    setStage(STAGES.LOADING);
    try {
      const data = await generateQuestions(propertyId, text, stars);
      setQuestionData(data);
      setCurrentQIndex(0);
      setAnswers([]);
      setImpactData([]);
      if (data.questions && data.questions.length > 0) {
        setStage(STAGES.QUESTION);
      } else {
        setStage(STAGES.IMPACT);
      }
    } catch (e) {
      console.error(e);
      setStage(STAGES.FORM);
    }
  }

  async function handleAnswer(answer) {
    const q = questionData.questions[currentQIndex];
    try {
      const result = await submitAnswer(
        propertyId,
        q.gap_topic,
        q.gap_type,
        answer,
        reviewData.text
      );
      setAnswers((prev) => [...prev, { question: q, answer }]);
      setImpactData((prev) => [...prev, result.impact]);
    } catch (e) {
      console.error(e);
    }

    const nextIndex = currentQIndex + 1;
    if (nextIndex < questionData.questions.length) {
      setCurrentQIndex(nextIndex);
    } else {
      setStage(STAGES.IMPACT);
    }
  }

  async function handleSkip() {
    const q = questionData.questions[currentQIndex];
    logSkip({
      question_id: q.gap_topic,
      property_id: propertyId,
      reviewer_archetype: questionData.archetype,
      skip_timestamp: new Date().toISOString(),
    });

    const nextIndex = currentQIndex + 1;
    if (nextIndex < questionData.questions.length) {
      setCurrentQIndex(nextIndex);
    } else {
      setStage(STAGES.IMPACT);
    }
  }

  function handleReset() {
    setStage(STAGES.FORM);
    setReviewData({ text: "", stars: 4 });
    setQuestionData(null);
    setCurrentQIndex(0);
    setAnswers([]);
    setImpactData([]);
    setShowQueue(false);
  }

  const totalQuestions = questionData?.questions?.length ?? 0;

  return (
    <div className="review-flow">
      {stage === STAGES.FORM && (
        <ReviewForm onSubmit={handleReviewSubmit} />
      )}

      {stage === STAGES.LOADING && (
        <div className="loading-screen">
          <div className="spinner" />
          <p>Analyzing your review for data gaps…</p>
        </div>
      )}

      {stage === STAGES.QUESTION && questionData && (
        <div className="question-stage">
          {/* Progress bar */}
          <div className="progress-bar-wrap">
            <div className="progress-label">
              Question {currentQIndex + 1} of {totalQuestions}
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${((currentQIndex + 1) / totalQuestions) * 100}%` }}
              />
            </div>
          </div>

          <QuestionCard
            question={questionData.questions[currentQIndex]}
            onAnswer={handleAnswer}
            onSkip={handleSkip}
          />

          {/* Judge panel toggle */}
          <button
            className="toggle-queue-btn"
            onClick={() => setShowQueue((v) => !v)}
          >
            {showQueue ? "Hide" : "Show"} gap ranking queue (judges)
          </button>

          {showQueue && questionData.gap_queue && (
            <QuestionQueue
              queue={questionData.gap_queue}
              archetype={questionData.archetype}
              coveredTopics={questionData.already_covered_topics}
            />
          )}
        </div>
      )}

      {stage === STAGES.IMPACT && (
        <ImpactPanel
          impacts={impactData}
          answers={answers}
          archetype={questionData?.archetype}
          gapQueue={questionData?.gap_queue}
          onReset={handleReset}
        />
      )}
    </div>
  );
}
