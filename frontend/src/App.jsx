import { useState, useEffect } from "react";
import { fetchProperties } from "./api";
import ReviewFlow from "./components/ReviewFlow";
import "./App.css";

const MONTEREY_ID = "fa014137b3ea9af6a90c0a86a1d099e46f7e56d6eb33db1ad1ec4bdac68c3caa";

export default function App() {
  const [properties, setProperties] = useState([]);
  const [selectedId, setSelectedId] = useState(MONTEREY_ID);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProperties()
      .then((d) => setProperties(d.properties || []))
      .catch(() => setProperties([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            <span className="brand-icon">✦</span>
            <span className="brand-name">ReviewIQ</span>
            <span className="brand-tag">Smart Gap-Filler</span>
          </div>
          <div className="property-select-wrap">
            <label htmlFor="prop-select">Property</label>
            {loading ? (
              <span className="loading-text">Loading…</span>
            ) : (
              <select
                id="prop-select"
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
              >
                {properties.map((p) => (
                  <option key={p.property_id} value={p.property_id}>
                    {p.city}, {p.country}
                    {p.star_rating ? ` (${p.star_rating}★)` : ""} — {p.total_reviews} reviews
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
      </header>

      <main className="app-main">
        {selectedId && (
          <ReviewFlow
            key={selectedId}
            propertyId={selectedId}
          />
        )}
      </main>
    </div>
  );
}
