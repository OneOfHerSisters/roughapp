import { useRef, useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Container, Title, Text, Table, Button, Select, TextInput,
  Checkbox, Group, Box, Stack, Loader, Modal, Paper,
} from '@mantine/core'
import { fetchDatasets, uploadDataset, introduceDataset, previewDataset } from '../api/client'
import { api } from '../api/client'
import { useAuth } from '../auth'
import type { Dataset } from '../types/api'

interface Preview {
  columns: string[]
  rows: string[][]
  total: number
  offset: number
  limit: number
}

interface DatasetStats {
  decision_col: string
  inconsistent_class_count: number
  inconsistent_total_rows: number
  duplicate_rows: number
  missing: { total_missing: number; rows_affected: number; total_rows: number; per_column: { column: string; missing: number }[] }
  accuracy: number
  quality: number
  per_decision: { decision: string | number; alpha: number; rho: number; L: number; U: number; boundary: number }[]
}

function parseCsvHeaderLine(line: string) {
  const cells: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i]
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i += 1
      } else {
        inQuotes = !inQuotes
      }
    } else if (ch === ',' && !inQuotes) {
      cells.push(current.trim())
      current = ''
    } else {
      current += ch
    }
  }
  cells.push(current.trim())
  return cells.map(c => c.replace(/^"(.*)"$/, '$1').trim())
}

export default function DatasetsPage() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetsLoading, setDatasetsLoading] = useState(true)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [introduceLoading, setIntroduceLoading] = useState(false)
  const [completeLoadingId, setCompleteLoadingId] = useState<number | null>(null)
  const [deduplicateLoadingId, setDeduplicateLoadingId] = useState<number | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState('')
  const [stats, setStats] = useState<Record<number, DatasetStats | 'loading' | 'error'>>({})
  const [introducing, setIntroducing] = useState<Record<number, boolean>>({})
  const [introForm, setIntroForm] = useState<Record<number, { method: string; gamma: string; k: string; seed: string }>>({})
  const [attrImportance, setAttrImportance] = useState<Record<number, { gamma_full: number; attributes: { attribute: string; importance: number; gamma_without: number }[] } | 'loading' | 'error'>>({})
  const [selectedAttrs, setSelectedAttrs] = useState<Record<number, string[]>>({})
  const [completeForm, setCompleteForm] = useState<Record<number, { method: string; by_class: boolean }>>({})
  const [preview, setPreview] = useState<Record<number, Preview | 'loading' | 'error'>>({})
  const [analysisOpen, setAnalysisOpen] = useState<Record<number, boolean>>({})
  const [previewId, setPreviewId] = useState<number | null>(null)
  const [pendingUpload, setPendingUpload] = useState<{ file: File; columns: string[]; decisionCol: string | null } | null>(null)

  async function loadPreview(id: number, offset = 0) {
    setPreview(s => ({ ...s, [id]: 'loading' }))
    try {
      const data = await previewDataset(id, offset)
      setPreview(s => ({ ...s, [id]: data }))
    } catch {
      setPreview(s => ({ ...s, [id]: 'error' }))
    }
  }

  function togglePreview(id: number) {
    setPreviewId(id)
    if (!preview[id]) {
      loadPreview(id)
    }
  }
  const loadDatasets = useCallback(async () => {
    setDatasetsLoading(true)
    try {
      setDatasets(await fetchDatasets())
    } catch {
      setDatasets([])
    } finally {
      setDatasetsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadDatasets()
  }, [loadDatasets])

  async function handleUpload(file: File, decisionCol: string) {
    setUploadLoading(true)
    try {
      await uploadDataset(file, decisionCol)
      setError('')
      setPendingUpload(null)
      await loadDatasets()
    } catch {
      setError('Upload failed — check that the file is a valid CSV.')
    } finally {
      setUploadLoading(false)
    }
  }

  async function handleIntroduce(id: number, body: Record<string, unknown>) {
    setIntroduceLoading(true)
    try {
      await introduceDataset(id, body)
      await loadDatasets()
      setIntroducing(s => ({ ...s, [id]: false }))
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(typeof detail === 'string' ? detail : `Inconsistency derivation failed for dataset ${id}.`)
      setIntroducing(s => ({ ...s, [id]: false }))
    } finally {
      setIntroduceLoading(false)
    }
  }

  async function handleComplete(id: number, body: Record<string, unknown>) {
    setCompleteLoadingId(id)
    try {
      await api.post(`/datasets/${id}/complete`, body)
      await loadDatasets()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? `Missing-value completion failed for dataset ${id}.`)
    } finally {
      setCompleteLoadingId(null)
    }
  }

  async function handleDeduplicate(id: number) {
    setDeduplicateLoadingId(id)
    try {
      await api.post(`/datasets/${id}/deduplicate`)
      await loadDatasets()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? 'Could not remove duplicates.')
    } finally {
      setDeduplicateLoadingId(null)
    }
  }

  async function analyze(id: number) {
    setStats(s => ({ ...s, [id]: 'loading' }))
    try {
      const { data } = await api.get(`/datasets/${id}/analyze`)
      setStats(s => ({ ...s, [id]: data }))
    } catch {
      setStats(s => ({ ...s, [id]: 'error' }))
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) void prepareUpload(file)
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) void prepareUpload(file)
  }

  async function prepareUpload(file: File) {
    try {
      const text = await file.text()
      const firstLine = text.split(/\r?\n/).find(line => line.trim().length > 0)
      if (!firstLine) {
        setError('The CSV file appears to be empty.')
        return
      }
      const columns = parseCsvHeaderLine(firstLine)
      if (columns.length === 0) {
        setError('Could not read CSV columns.')
        return
      }
      setPendingUpload({ file, columns, decisionCol: columns[columns.length - 1] ?? null })
      setError('')
    } catch {
      setError('Could not read the selected file.')
    }
  }

  function submitComplete(id: number) {
    const f = completeForm[id] ?? { method: 'fill', by_class: false }
    setError('')
    void handleComplete(id, { method: f.method, by_class: f.by_class })
  }

  async function loadAttrImportance(id: number) {
    setAttrImportance(s => ({ ...s, [id]: 'loading' }))
    try {
      const { data } = await api.get(`/datasets/${id}/attribute-importance`)
      setAttrImportance(s => ({ ...s, [id]: data }))
    } catch {
      setAttrImportance(s => ({ ...s, [id]: 'error' }))
    }
  }

  function toggleIntroduce(id: number) {
    setIntroducing(s => ({ ...s, [id]: !s[id] }))
    if (!introForm[id]) setIntroForm(s => ({ ...s, [id]: { method: 'until_quality', gamma: '0.8', k: '10', seed: '42' } }))
  }

  function handleMethodChange(id: number, method: string) {
    setIntroForm(s => ({ ...s, [id]: { ...s[id], method } }))
    if (method === 'drop_attributes' && !attrImportance[id]) {
      loadAttrImportance(id)
    }
  }

  function toggleAttr(id: number, attr: string) {
    setSelectedAttrs(s => {
      const cur = s[id] ?? []
      return { ...s, [id]: cur.includes(attr) ? cur.filter(a => a !== attr) : [...cur, attr] }
    })
  }

  function submitIntroduce(id: number) {
    const f = introForm[id]
    if (!f) return
    setError('')
    const seed = f.seed ? Number(f.seed) : undefined
    if (f.method === 'until_quality') {
      const gamma = parseFloat(f.gamma)
      if (isNaN(gamma) || gamma <= 0 || gamma >= 1) { setError('γ must be between 0 and 1.'); return }
      void handleIntroduce(id, { method: 'until_quality', target_gamma: gamma, random_state: seed })
    } else if (f.method === 'duplicate_flip') {
      const k = parseInt(f.k)
      if (isNaN(k) || k < 1) { setError('k must be ≥ 1.'); return }
      void handleIntroduce(id, { method: 'duplicate_flip', k, random_state: seed })
    } else if (f.method === 'drop_attributes') {
      const attrs = selectedAttrs[id] ?? []
      if (attrs.length === 0) { setError('Select at least one attribute to drop.'); return }
      void handleIntroduce(id, { method: 'drop_attributes', attrs })
    }
  }

  return (
    <Container size="lg" py="lg">
      <Title order={2} mb="md">Datasets</Title>

      <Box
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        p={40} mb="md" ta="center"
        style={{
          border: '2px dashed var(--mantine-color-gray-4)',
          borderRadius: 'var(--mantine-radius-sm)',
          cursor: 'pointer',
          color: 'var(--mantine-color-dimmed)',
        }}
      >
        {uploadLoading ? 'Uploading…' : 'Drop a CSV here or click to choose'}
        {user && (
          <Text size="xs" mt={6}>
            Your datasets and derived tables are saved to your account.
          </Text>
        )}
        <input ref={inputRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={handleFile} />
      </Box>

      {error && <Text c="red" size="sm" mb="sm">{error}</Text>}

      {datasetsLoading ? (
        <Loader size="sm" />
      ) : datasets.length === 0 ? (
        <Text size="sm">No datasets yet — upload one above.</Text>
      ) : (
        <Stack gap={0}>
          {datasets.map(ds => {
            const s = stats[ds.id]
            const showIntro = introducing[ds.id]
            const showAnalysis = Boolean(analysisOpen[ds.id])
            const showPreview = previewId === ds.id
            const f = introForm[ds.id] ?? { method: 'until_quality', gamma: '0.8', k: '10', seed: '42' }
            return (
              <Box key={ds.id} py="sm" style={{ borderBottom: '1px solid var(--mantine-color-gray-3)' }}>
                <Group gap="xs">
                    <Text size="sm">
                    <Text span fw={600}>{ds.name}</Text> — {ds.rows} rows, {ds.cols} cols · decision {ds.decision_col}
                    {ds.provenance?.kind === 'derived' && ds.provenance.source_dataset_id != null && ds.gamma_actual != null && (
                      <Text span c="dimmed"> (derived from #{ds.provenance.source_dataset_id}, γ={ds.gamma_actual})</Text>
                    )}
                  </Text>
                  <Button
                    variant={showAnalysis ? 'filled' : 'default'}
                    color={showAnalysis ? 'gold' : undefined}
                    onClick={() => {
                      setAnalysisOpen(prev => ({ ...prev, [ds.id]: true }))
                      void analyze(ds.id)
                    }}
                  >
                    {s === 'loading' ? 'Analyzing…' : 'Analyze'}
                  </Button>
                  <Button
                    variant="default"
                    onClick={() => toggleIntroduce(ds.id)}
                  >
                    Introduce inconsistency
                  </Button>
                  <Button
                    variant={showPreview ? 'filled' : 'default'}
                    color={showPreview ? 'gold' : undefined}
                    onClick={() => togglePreview(ds.id)}
                  >
                    View table
                  </Button>
                </Group>

                <Modal
                  opened={showIntro}
                  onClose={() => setIntroducing(prev => ({ ...prev, [ds.id]: false }))}
                  title={`Introduce inconsistency — ${ds.name}`}
                  size="lg"
                  centered
                >
                  <Box>
                    <Group gap="sm" align="flex-end" wrap="wrap">
                      <Select
                        label="Method" size="xs" w={200}
                        value={f.method}
                        onChange={v => v && handleMethodChange(ds.id, v)}
                        data={[
                          { value: 'until_quality', label: 'Until target γ' },
                          { value: 'duplicate_flip', label: 'Duplicate & flip objects' },
                          { value: 'drop_attributes', label: 'Remove conditional attributes' },
                        ]}
                        allowDeselect={false}
                      />

                      {f.method === 'until_quality' && (
                        <TextInput
                          label="γ target" size="xs" w={90}
                          type="number" step={0.05} min={0.01} max={0.99}
                          value={f.gamma}
                          onChange={e => {
                            const v = e.currentTarget.value
                            setIntroForm(s => ({ ...s, [ds.id]: { ...(s[ds.id] ?? f), gamma: v } }))
                          }}
                        />
                      )}

                      {f.method === 'duplicate_flip' && (
                        <TextInput
                          label="Rows to flip" size="xs" w={90}
                          type="number" min={1}
                          value={f.k}
                          onChange={e => {
                            const v = e.currentTarget.value
                            setIntroForm(s => ({ ...s, [ds.id]: { ...(s[ds.id] ?? f), k: v } }))
                          }}
                        />
                      )}

                      {f.method !== 'drop_attributes' && (
                        <TextInput
                          label="Seed" size="xs" w={80}
                          type="number"
                          value={f.seed}
                          onChange={e => {
                            const v = e.currentTarget.value
                            setIntroForm(s => ({ ...s, [ds.id]: { ...(s[ds.id] ?? f), seed: v } }))
                          }}
                        />
                      )}

                      {f.method !== 'drop_attributes' && (
                        <Button onClick={() => submitIntroduce(ds.id)} loading={introduceLoading}>
                          Save inconsistent table
                        </Button>
                      )}
                    </Group>

                    {f.method === 'drop_attributes' && (() => {
                      const ai = attrImportance[ds.id]
                      if (ai === 'loading') return <Text size="sm" mt="xs">Computing importance…</Text>
                      if (ai === 'error') return <Text size="sm" c="red" mt="xs">Failed to load attributes.</Text>
                      if (!ai) return null
                      const checked = selectedAttrs[ds.id] ?? []
                      return (
                        <Box mt={10}>
                          <Table fz="xs" w="auto">
                            <Table.Thead>
                              <Table.Tr>
                                <Table.Th />
                                <Table.Th>Conditional attribute</Table.Th>
                                <Table.Th ta="right">Δγ</Table.Th>
                                <Table.Th ta="right">γ without attr.</Table.Th>
                              </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                              {ai.attributes.map((a) => {
                                return (
                                  <Table.Tr key={a.attribute}>
                                    <Table.Td>
                                      <Checkbox
                                        size="xs"
                                        checked={checked.includes(a.attribute)}
                                        onChange={() => toggleAttr(ds.id, a.attribute)}
                                      />
                                    </Table.Td>
                                    <Table.Td>
                                      {a.attribute}
                                    </Table.Td>
                                    <Table.Td ta="right" c={a.importance === 0 ? 'gray.5' : undefined}>
                                      {a.importance === 0 ? '—' : a.importance.toFixed(4)}
                                    </Table.Td>
                                    <Table.Td ta="right" c="dimmed">{a.gamma_without.toFixed(4)}</Table.Td>
                                  </Table.Tr>
                                )
                              })}
                            </Table.Tbody>
                          </Table>
                          <Button
                            mt={8}
                            onClick={() => submitIntroduce(ds.id)}
                            loading={introduceLoading}
                            disabled={checked.length === 0}
                          >
                            Drop {checked.length || ''} attribute{checked.length !== 1 ? 's' : ''} & save table
                          </Button>
                        </Box>
                      )
                    })()}
                  </Box>
                </Modal>

                {analysisOpen[ds.id] && s && s !== 'loading' && s !== 'error' && (() => {
                  const cf = completeForm[ds.id] ?? { method: 'fill', by_class: false }
                  const missing = s.missing ?? { total_missing: 0, rows_affected: 0, total_rows: 0, per_column: [] }
                  return (
                  <Paper mt="sm" pl={12} p="sm" bg="gold.0">
                    <Group justify="space-between" align="flex-start" mb={6}>
                      <Text size="sm" fw={600} c="gold.8">Analysis</Text>
                      <Button variant="subtle" color="gray" onClick={() => setAnalysisOpen(prev => ({ ...prev, [ds.id]: false }))}>
                        Close
                      </Button>
                    </Group>
                    <Text size="sm">Inconsistent classes: <Text span fw={600}>{s.inconsistent_class_count}</Text> ({s.inconsistent_total_rows} rows)</Text>
                    <Group gap="sm" align="center" mt={2}>
                      <Text size="sm">Quality γ: <Text span fw={600}>{s.quality.toFixed(4)}</Text>{'  '}Accuracy α: <Text span fw={600}>{s.accuracy.toFixed(4)}</Text></Text>
                      {s.quality < 1 && (
                        <Button
                          variant="light" color="teal" size="compact-xs"
                          onClick={() => navigate(`/experiments?dataset_id=${ds.id}`)}
                        >
                          Remove inconsistency
                        </Button>
                      )}
                    </Group>
                    <Group gap="sm" align="center" mt={2}>
                      <Text size="sm">
                        Duplicate rows: <Text span fw={600}>{s.duplicate_rows}</Text>
                      </Text>
                      {s.duplicate_rows > 0 && (
                        <Button
                          variant="light" color="gold" size="compact-xs"
                          loading={deduplicateLoadingId === ds.id}
                          onClick={() => void handleDeduplicate(ds.id)}
                        >
                          Remove duplicates
                        </Button>
                      )}
                    </Group>
                    <Group gap="sm" align="flex-end" mt={2} wrap="wrap">
                      <Text size="sm">
                        Missing values: <Text span fw={600}>{missing.total_missing}</Text>
                        {missing.total_missing > 0 && (
                          <Text span c="dimmed"> {' '}in {missing.rows_affected} of {missing.total_rows} rows
                            {missing.per_column.length > 0 && (
                              <> — {missing.per_column.map(c => `${c.column} (${c.missing})`).join(', ')}</>
                            )}
                          </Text>
                        )}
                      </Text>
                      {missing.total_missing > 0 && (
                        <>
                          <Select
                            size="xs" w={210}
                            value={cf.method}
                            onChange={v => v && setCompleteForm(st => ({ ...st, [ds.id]: { ...cf, method: v } }))}
                            data={[
                              { value: 'fill', label: 'Impute (mode / mean)' },
                              { value: 'remove', label: 'Delete incomplete objects' },
                            ]}
                            allowDeselect={false}
                          />
                          {cf.method === 'fill' && (
                            <Group gap={6} mb={6} wrap="nowrap">
                              <Checkbox
                                size="xs"
                                checked={cf.by_class}
                                onChange={e => {
                                  const checked = e.currentTarget.checked
                                  setCompleteForm(st => ({
                                    ...st,
                                    [ds.id]: {
                                      ...(st[ds.id] ?? { method: 'fill', by_class: false }),
                                      by_class: checked,
                                    },
                                  }))
                                }}
                              />
                              <Text size="sm">within decision class only</Text>
                            </Group>
                          )}
                          <Button
                            variant="light" color="gold" size="compact-xs"
                            onClick={() => submitComplete(ds.id)}
                            loading={completeLoadingId === ds.id}
                          >
                            {cf.method === 'fill' ? 'Impute missing' : 'Delete incomplete'}
                          </Button>
                        </>
                      )}
                    </Group>
                    <Table
                      withTableBorder
                      withColumnBorders
                      fz="xs"
                      w="auto"
                      mt={8}
                      styles={{
                        table: {
                          backgroundColor: 'var(--mantine-color-teal-0)',
                        },
                        th: {
                          backgroundColor: 'var(--mantine-color-teal-2)',
                          color: 'var(--mantine-color-teal-9)',
                        },
                        td: {
                          backgroundColor: 'rgba(244, 251, 251, 0.96)',
                        },
                      }}
                    >
                      <Table.Thead>
                        <Table.Tr>
                          <Table.Th>Decision</Table.Th><Table.Th>α</Table.Th><Table.Th>ρ</Table.Th>
                          <Table.Th>L</Table.Th><Table.Th>U</Table.Th><Table.Th>Boundary</Table.Th>
                        </Table.Tr>
                      </Table.Thead>
                      <Table.Tbody>
                        {s.per_decision.map(r => (
                          <Table.Tr key={String(r.decision)}>
                            <Table.Td>{r.decision}</Table.Td>
                            <Table.Td>{r.alpha.toFixed(3)}</Table.Td>
                            <Table.Td>{r.rho.toFixed(3)}</Table.Td>
                            <Table.Td>{r.L}</Table.Td>
                            <Table.Td>{r.U}</Table.Td>
                            <Table.Td>{r.boundary}</Table.Td>
                          </Table.Tr>
                        ))}
                      </Table.Tbody>
                    </Table>
                  </Paper>
                  )
                })()}
                {s === 'error' && <Text span c="red" size="sm" ml={8}>Failed</Text>}
              </Box>
            )
          })}
        </Stack>
      )}

      <Modal
        opened={pendingUpload != null}
        onClose={() => setPendingUpload(null)}
        title="Upload dataset"
        centered
      >
        {pendingUpload && (
          <Stack gap="sm">
            <Text size="sm">
              Choose the decision column for <Text span fw={600}>{pendingUpload.file.name}</Text>.
            </Text>
            <Select
              label="Decision column"
              data={pendingUpload.columns.map(col => ({ value: col, label: col }))}
              value={pendingUpload.decisionCol}
              onChange={value => setPendingUpload(cur => cur ? { ...cur, decisionCol: value } : cur)}
              allowDeselect={false}
            />
            <Group justify="flex-end">
              <Button variant="default" onClick={() => setPendingUpload(null)}>Cancel</Button>
              <Button
                onClick={() => {
                  if (!pendingUpload.decisionCol) {
                    setError('Choose the decision column first.')
                    return
                  }
                  void handleUpload(pendingUpload.file, pendingUpload.decisionCol)
                }}
                loading={uploadLoading}
              >
                Upload
              </Button>
            </Group>
          </Stack>
        )}
      </Modal>

      <Modal
        opened={previewId != null}
        onClose={() => setPreviewId(null)}
        title={previewId != null ? datasets.find(ds => ds.id === previewId)?.name ?? 'Dataset preview' : 'Dataset preview'}
        size="xl"
        centered
      >
        {previewId != null && (() => {
          const p = preview[previewId]
          if (!p || p === 'loading') return <Text size="sm">Loading…</Text>
          if (p === 'error') return <Text size="sm" c="red">Failed to load preview.</Text>
          return (
            <Box style={{ overflowX: 'auto' }}>
              <Text size="xs" c="dimmed" mb={4}>
                Showing {p.offset + 1}–{Math.min(p.offset + p.limit, p.total)} of {p.total} rows
              </Text>
              <Table withTableBorder withColumnBorders fz="xs" w="auto" style={{ whiteSpace: 'nowrap' }}>
                <Table.Thead>
                  <Table.Tr>{p.columns.map(col => <Table.Th key={col}>{col}</Table.Th>)}</Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {p.rows.map((row, i) => (
                    <Table.Tr key={i}>{row.map((cell, j) => <Table.Td key={j}>{cell}</Table.Td>)}</Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
              <Group gap="xs" mt={6}>
                {p.offset > 0 && (
                  <Button variant="default" onClick={() => loadPreview(previewId, p.offset - p.limit)}>← Prev</Button>
                )}
                {p.offset + p.limit < p.total && (
                  <Button variant="default" onClick={() => loadPreview(previewId, p.offset + p.limit)}>Next →</Button>
                )}
              </Group>
            </Box>
          )
        })()}
      </Modal>
    </Container>
  )
}
