import "./ImpactPanel.css";

const TOPIC_ICON = {
  pets: "🐾",
  affordability: "💰",
  service_checkin: "🛎",
  location_transportation: "📍",
  ambiance_decor: "🛏",
  amenities_food: "🍳",
  cleanliness: "✨",
  accessibility: "♿",
};

export default function ImpactPanel({ impacts, answers, archetype, gapQueue, onReset }) {
  const hasAnswers = impacts && impacts.length > 0;

  return (
    <div className="impact-panel">
      <div className="impact-header">
        <div className="impact-check">✓</div>
        <h2>Thank you — you just helped future guests</h2>
        <p className="impact-sub">
          Here's what changed in the property's info card because of your answers.
        </p>
      </div>

      {!hasAnswers && (
        <div className="no-answers">
          <p>No questions answered — that's okay! Your review was still submitted.</p>
        </div>
      )}

      {/* Before / After cards */}
      {hasAnswers && (
        <div className="before-after-section">
          {impacts.map((impact, i) => (
            <div key={i} className="before-after-card">
              <div className="bac-topic">
                <span className="bac-icon">{TOPIC_ICON[impact.topic] || "•"}</span>
                <span className="bac-topic-name">{topicLabel(impact.topic)}</span>
                <span className={`gap-badge-sm gap-${impact.gap_type}`}>
                  {gapLabel(impact.gap_type)}
                </span>
              </div>
              <div className="bac-columns">
                <div className="bac-col before">
                  <div className="bac-col-label">Before your answer</div>
                  <div className="bac-value missing">{impact.before_label}</div>
                </div>
                <div className="bac-arrow">→</div>
                <div className="bac-col after">
                  <div className="bac-col-label">After your answer</div>
                  <div className="bac-value filled">{impact.after_label}</div>
                </div>
              </div>
              <div className="bac-lift">{impact.lift_message}</div>
              {impact.insight_snippet && (
                <div className="bac-insight">
                  💡 <em>{impact.insight_snippet}</em>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Gap queue (judges panel) */}
      {gapQueue && gapQueue.length > 0 && (
        <div className="gap-queue-section">
          <h3>Full gap ranking queue</h3>
          <p className="queue-sub">How the system scored and ranked each data gap for your review</p>
          <table className="queue-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Topic</th>
                <th>Gap type</th>
                <th>Score</th>
                <th>Friction</th>
                <th>Final rank</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {gapQueue.map((g) => (
                <tr key={g.rank} className={g.status === "asked" ? "row-asked" : g.skipped ? "row-skipped" : ""}>
                  <td>{g.rank}</td>
                  <td>{topicLabel(g.topic)}</td>
                  <td>
                    <span className={`gap-badge-sm gap-${g.gap_type}`}>
                      {gapLabel(g.gap_type)}
                    </span>
                  </td>
                  <td>{g.gap_score?.toFixed(2)}</td>
                  <td>{g.friction_cost}</td>
                  <td><strong>{g.final_rank?.toFixed(3)}</strong></td>
                  <td>
                    {g.status === "asked" ? (
                      <span className="status-asked">→ Asked</span>
                    ) : g.skipped ? (
                      <span className="status-skipped">Covered in review</span>
                    ) : (
                      <span className="status-queued">Queued</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {archetype && (
        <div className="archetype-row">
          <span>Detected traveler type: </span>
          <strong>{archetypeLabel(archetype)}</strong>
        </div>
      )}

      <button className="reset-btn" onClick={onReset}>
        Leave another review
      </button>
    </div>
  );
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

function gapLabel(type) {
  const labels = {
    policy_contradiction: "Policy contradiction",
    zero_fill: "0% filled",
    sparse_fill: "Sparse data",
    score_drift: "Score drift",
    staleness: "Stale",
  };
  return labels[type] || type;
}

function archetypeLabel(a) {
  const labels = {
    family: "Family traveler",
    couple: "Couple / Romantic",
    business: "Business traveler",
    pet_owner: "Pet owner",
    budget: "Budget traveler",
    accessibility: "Accessibility-focused",
    general: "General traveler",
  };
  return labels[a] || a;
}
