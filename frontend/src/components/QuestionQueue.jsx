import "./QuestionQueue.css";

export default function QuestionQueue({ queue, archetype, coveredTopics }) {
  return (
    <div className="question-queue">
      <div className="qq-header">
        <span className="qq-title">Gap Ranking Queue</span>
        <span className="qq-sub">How the system chose these questions</span>
      </div>

      {archetype && (
        <div className="qq-archetype">
          Reviewer archetype: <strong>{archetypeLabel(archetype)}</strong>
        </div>
      )}

      {coveredTopics && coveredTopics.length > 0 && (
        <div className="qq-covered">
          Already in review (skipped): {coveredTopics.map(topicLabel).join(", ")}
        </div>
      )}

      <table className="qq-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Topic</th>
            <th>Gap type</th>
            <th>Score</th>
            <th>÷ Friction</th>
            <th>= Rank</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {queue.map((g) => (
            <tr
              key={g.rank}
              className={
                g.status === "asked"
                  ? "row-asked"
                  : g.skipped
                  ? "row-skipped"
                  : ""
              }
            >
              <td>{g.rank}</td>
              <td>{topicLabel(g.topic)}</td>
              <td>
                <span className={`badge gap-${g.gap_type}`}>
                  {gapLabel(g.gap_type)}
                </span>
              </td>
              <td>{g.gap_score?.toFixed(2)}</td>
              <td>{g.friction_cost}</td>
              <td>
                <strong>{g.final_rank?.toFixed(3)}</strong>
              </td>
              <td>
                {g.status === "asked" ? (
                  <span className="tag-asked">→ Asked</span>
                ) : g.skipped ? (
                  <span className="tag-covered">Covered</span>
                ) : (
                  <span className="tag-queued">Queued</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="qq-formula">
        Final rank = <em>gap_score</em> ÷ <em>friction_cost</em> &nbsp;|&nbsp;
        gap_score = staleness × 0.4 + guest_demand × 0.4 + reviewer_fit × 0.2
      </div>
    </div>
  );
}

function topicLabel(topic) {
  const m = {
    pets: "Pets",
    affordability: "Value for money",
    service_checkin: "Service & Check-in",
    location_transportation: "Location",
    ambiance_decor: "Room quality",
    amenities_food: "Amenities & Food",
    cleanliness: "Cleanliness",
    accessibility: "Accessibility",
  };
  return m[topic] || topic;
}

function gapLabel(type) {
  const m = {
    policy_contradiction: "Policy contradiction",
    zero_fill: "0% filled",
    sparse_fill: "Sparse data",
    score_drift: "Score drift",
    staleness: "Stale",
  };
  return m[type] || type;
}

function archetypeLabel(a) {
  const m = {
    family: "Family traveler",
    couple: "Couple / Romantic",
    business: "Business traveler",
    pet_owner: "Pet owner",
    budget: "Budget traveler",
    accessibility: "Accessibility-focused",
    general: "General traveler",
  };
  return m[a] || a;
}
