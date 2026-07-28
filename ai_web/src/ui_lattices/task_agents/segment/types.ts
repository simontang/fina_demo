export interface SegmentDefinitionVO {
  id: number;
  tenantId: string;
  threadId?: string;
  name: string;
  description: string;
  datasourceId: number;
  querySql: string;
  status: number;
  createdAt: string;
  updatedAt: string;
}

export interface SegmentDefinitionPage {
  items: SegmentDefinitionVO[];
  total: number;
  page: number;
  pageSize: number;
}

export interface SegmentDataVO {
  id: number;
  tenantId: string;
  definitionId: number;
  runId: string;
  dataJson: string;
  rowCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface SegmentDataPage {
  items: SegmentDataVO[];
  total: number;
  page: number;
  pageSize: number;
}

export type SegmentDataRow = Record<string, unknown>;
