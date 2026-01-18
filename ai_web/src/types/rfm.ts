export type RFMSegmentationMethod = "quantiles" | "kmeans";

export interface RFMWeights {
  r: number;
  f: number;
  m: number;
}

export interface RFMOverview {
  total_users: number;
  total_orders: number;
  total_revenue: number;
}

export interface RFMSegmentSummary {
  segment: string;
  count: number;
  share_pct: number;
  revenue: number;
  revenue_share_pct: number;
  avg_recency_days: number;
  avg_frequency: number;
  avg_monetary: number;
  avg_r_score: number;
  avg_f_score: number;
  avg_m_score: number;
  avg_rfm_score: number;
  color: string;
}

export interface RFMMatrixAxis {
  id: number;
  label: string;
}

export interface RFMMatrixThresholds {
  scale: number;
  low_end: number;
  high_start: number;
  rule: string;
}

export interface RFMMatrixCell extends RFMSegmentSummary {
  r_level: number;
  f_level: number;
}

export interface RFMMatrix {
  rows: RFMMatrixAxis[];
  cols: RFMMatrixAxis[];
  thresholds: RFMMatrixThresholds;
  cells: RFMMatrixCell[];
}

export interface RFMScoreDistributions {
  r: Record<string, number>;
  f: Record<string, number>;
  m: Record<string, number>;
}

export interface RFMMoM {
  total_users: number;
  total_revenue: number;
  total_users_change_pct: number | null;
  total_revenue_change_pct: number | null;
}

export interface RFMAnalysisData {
  analysis_id: string;
  dataset_id: string;
  reference_date: string;
  time_window_days: number;
  scoring_scale: number;
  segmentation_method: RFMSegmentationMethod;
  weights: RFMWeights;
  overview: RFMOverview;
  segments: RFMSegmentSummary[];
  matrix: RFMMatrix;
  score_distributions: RFMScoreDistributions;
  mom?: RFMMoM | null;
  insight_markdown: string;
}

export interface RFMRunRequest {
  time_window_days: number;
  scoring_scale: number;
  segmentation_method: RFMSegmentationMethod;
  weights: RFMWeights;
}

export interface RFMSegmentCustomer {
  user_id: string | number;
  recency_days: number;
  frequency: number;
  monetary: number;
  r_score: number;
  f_score: number;
  m_score: number;
  rfm_score: number;
  segment: string;
}

export interface RFMSegmentDetailResponse {
  records: RFMSegmentCustomer[];
  total: number;
  page: number;
  pageSize: number;
}
