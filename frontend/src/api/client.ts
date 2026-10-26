import axios from 'axios'

export const api = axios.create({ baseURL: '/api', withCredentials: true })

// Seed the auth header synchronously at module load so the very first request
// (before AuthProvider's effect runs) already carries the stored token.
// Avoids a race where the initial datasets query goes out unauthenticated.
const storedToken = localStorage.getItem('token')
if (storedToken) {
  api.defaults.headers.common['Authorization'] = `Bearer ${storedToken}`
}

// ── Datasets ──────────────────────────────────────────────────────────────────

export async function uploadDataset(file: File, decisionCol: string) {
  const form = new FormData()
  form.append('file', file)
  form.append('decision_col', decisionCol)
  const { data } = await api.post('/datasets', form)
  return data
}

export async function fetchDatasets() {
  const { data } = await api.get('/datasets')
  return data
}

export async function previewDataset(id: number, offset = 0, limit = 100) {
  const { data } = await api.get(`/datasets/${id}/preview`, { params: { offset, limit } })
  return data as { columns: string[]; rows: string[][]; total: number; offset: number; limit: number }
}

export async function introduceDataset(id: number, payload: Record<string, unknown>) {
  const { data } = await api.post(`/datasets/${id}/introduce`, payload)
  return data
}

// ── Experiments ───────────────────────────────────────────────────────────────

export async function downloadAuthenticated(path: string, fallbackFilename: string) {
  const response = await api.get(path, { responseType: 'blob', validateStatus: () => true })
  if (response.status >= 400) {
    let detail = `HTTP ${response.status}`
    try {
      const text = await (response.data as Blob).text()
      const body = JSON.parse(text) as { detail?: string }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // keep generic status message
    }
    throw new Error(detail)
  }
  const cd = response.headers['content-disposition'] as string | undefined
  const match = cd?.match(/filename="?([^";]+)"?/)
  const filename = match?.[1] ?? fallbackFilename
  const url = URL.createObjectURL(response.data as Blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export async function runExperiment(payload: {
  dataset_id: number
  strategies?: string[]
  random_state?: number
  tie_policy?: string
  qr_tie_policy?: string
}) {
  const { data } = await api.post('/experiments', payload)
  return data
}

export async function fetchExperiments() {
  const { data } = await api.get('/experiments')
  return data
}

export async function fetchExperiment(id: number) {
  const { data } = await api.get(`/experiments/${id}`)
  return data
}

export async function computeTreeMetrics(id: number) {
  const { data } = await api.post(`/experiments/${id}/tree-metrics`)
  return data as {
    before: { depth: number | null; avg_depth: number | null; nodes: number | null }
    strategies: Record<string, { depth: number | null; avg_depth: number | null; nodes: number | null }>
  }
}

export interface TreeStats {
  depth: number
  avg_depth: number
  nodes: number
}

export async function previewDatasetTree(datasetId: number) {
  const { data } = await api.get<{ stats: TreeStats; text: string }>(
    `/datasets/${datasetId}/tree`,
  )
  return data
}

/** @deprecated alias */
export const buildDatasetTree = previewDatasetTree

export async function fetchDatasetTreeText(datasetId: number) {
  const { data } = await api.get<{ tree_run_id: number; text: string }>(
    `/datasets/${datasetId}/tree/text`,
  )
  return data
}

export interface TreeOrigin {
  type: 'upload' | 'derived' | 'experiment_result'
  input_dataset_name: string
  input_dataset_id: number | null
  group_name?: string
  cluster_dataset_id?: number | null
  label: string
  parent_dataset_id?: number | null
  experiment_id?: number | null
  strategy?: string | null
}

export interface TreeSourceRow {
  source_key: string
  kind: 'dataset' | 'result'
  label: string
  table_name: string
  origin: TreeOrigin
  dataset_id: number | null
  experiment_id: number | null
  strategy: string | null
  rows: number | null
  gamma: number | null
  tree_built: boolean
  built_algorithms: string[]
  tree_run_id: number | null
  depth: number | null
  avg_depth: number | null
  nodes: number | null
}

export async function fetchTreeAlgorithms() {
  const { data } = await api.get<{ name: string; description: string }[]>('/trees/algorithms')
  return data
}

export async function deleteTreeRun(treeRunId: number) {
  await api.delete(`/trees/${treeRunId}`)
}

export async function fetchTreeSources() {
  const { data } = await api.get<TreeSourceRow[]>('/trees/sources')
  return data
}

export async function buildTreeFromSource(payload: {
  kind: 'dataset' | 'result'
  dataset_id?: number
  experiment_id?: number
  strategy?: string
  algorithm?: string
}) {
  const { data } = await api.post<{
    tree_run_id: number
    depth: number
    avg_depth: number
    nodes: number
    text: string
  }>('/trees/build', payload)
  return data
}

export async function fetchTreeText(treeRunId: number) {
  const { data } = await api.get<{ tree_run_id: number; text: string }>(
    `/trees/${treeRunId}/text`,
  )
  return data
}

export async function buildDatasetTreePersist(datasetId: number) {
  const { data } = await api.post<{
    tree_run_id: number
    dataset_id: number
    dataset_name: string
    depth: number
    avg_depth: number
    nodes: number
    text: string
  }>(`/datasets/${datasetId}/tree/build`)
  return data
}

export async function buildOriginalExperimentTree(expId: number) {
  const { data } = await api.post<{
    tree_run_id: number
    dataset_id: number
    dataset_name: string
    depth: number
    avg_depth: number
    nodes: number
    text: string
  }>(`/experiments/${expId}/tree/build`)
  return data
}

export async function buildResultTree(expId: number, strategy: string) {
  const { data } = await api.post<{
    tree_run_id: number
    experiment_id: number
    strategy: string
    depth: number
    avg_depth: number
    nodes: number
    text: string
  }>(`/experiments/${expId}/results/${strategy}/tree/build`)
  return data
}

export async function fetchResultTreeText(expId: number, strategy: string) {
  const { data } = await api.get<{ tree_run_id: number; text: string }>(
    `/experiments/${expId}/results/${strategy}/tree/text`,
  )
  return data
}

export async function previewResult(expId: number, strategy: string, offset = 0, limit = 100) {
  const { data } = await api.get(`/experiments/${expId}/results/${strategy}/preview`, {
    params: { offset, limit },
  })
  return data as { columns: string[]; rows: string[][]; total: number; offset: number; limit: number }
}
