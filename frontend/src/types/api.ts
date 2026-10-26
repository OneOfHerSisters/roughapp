export interface DecisionMetrics {
  decision: string | number
  L: number
  U: number
  boundary: number
  alpha: number
  rho: number
}

export interface InconsistencyReport {
  inconsistent_class_count: number
  inconsistent_total_rows: number
  accuracy: number
  quality: number
  per_decision: DecisionMetrics[]
  examples: Record<string, unknown>[][]
}

export interface DatasetProvenance {
  kind: 'upload' | 'derived' | 'experiment_output'
  source_dataset_id: number | null
}

export interface Dataset {
  id: number
  name: string
  rows: number
  cols: number
  decision_col: string
  created_at: string
  storage_path: string
  provenance: DatasetProvenance | null
  gamma_actual: number | null
  alpha_actual?: number | null
  depth_actual?: number | null
  avg_depth_actual?: number | null
  nodes_actual?: number | null
}

export type Strategy = 'mcd' | 'rough-approximation' | 'gd'

export interface ExperimentRequest {
  dataset_id: number
  strategies?: Strategy[]
  random_state?: number
  tie_policy?: string
  qr_tie_policy?: string
}

export interface ExperimentResult {
  strategy: Strategy
  gamma_after: number | null
  alpha_after: number | null
  rows_after: number
  rows_removed: number
  depth_after: number | null
  avg_depth_after: number | null
  nodes_after: number | null
  n_gd_codes: number | null
  n_gd_conflict_codes: number | null
  time_ms: number
  details: Record<string, unknown> | null
  has_result: boolean
}

export interface Experiment {
  id: number
  dataset_id: number
  dataset_name: string
  random_state: number | null
  tie_policy: string | null
  qr_tie_policy: string | null
  created_at: string
  gamma_before: number | null
  alpha_before: number | null
  rows_before: number | null
  depth_before: number | null
  avg_depth_before: number | null
  nodes_before: number | null
  results: ExperimentResult[]
}
