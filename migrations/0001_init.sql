CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  text_result TEXT NOT NULL,
  confidence REAL NOT NULL,
  model TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_created_at
ON results(created_at);
