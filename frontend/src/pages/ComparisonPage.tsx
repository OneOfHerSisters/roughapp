import { useState, Fragment, useMemo, useEffect, useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Container, Title, Text, Anchor, Table, Loader, Box, Button, Paper, Select, Group, Modal, TextInput, Badge, Stack,
} from '@mantine/core'
import {
  api,
  buildTreeFromSource,
  deleteTreeRun,
  fetchTreeAlgorithms,
  fetchTreeSources,
  fetchTreeText,
  type TreeOrigin,
  type TreeSourceRow,
} from '../api/client'
import { useAuth } from '../auth'
import TreeTextPanel from '../components/TreeTextPanel'

interface TreeRunRow {
  tree_run_id: number
  created_at: string
  algorithm: string
  label: string
  table_name: string
  origin: TreeOrigin
  source: {
    kind: 'dataset' | 'result'
    dataset_id: number | null
    dataset_name: string
    strategy: string | null
    experiment_id: number | null
  }
  rows: number | null
  gamma: number | null
  depth: number | null
  avg_depth: number | null
  nodes: number | null
}

function fmt(v: number | null | undefined, digits = 2) {
  if (v == null) return '—'
  return Number.isInteger(v) ? String(v) : v.toFixed(digits)
}

function displayTableName(row: { table_name: string; origin: TreeOrigin }) {
  if (row.origin.type === 'experiment_result') {
    return row.origin.input_dataset_name
  }
  return row.table_name
}

function groupName(origin: TreeOrigin): string {
  return origin.group_name ?? origin.input_dataset_name
}

function clusterId(row: { dataset_id?: number | null; origin: TreeOrigin; source?: { dataset_id: number | null } }): number {
  if (row.origin.cluster_dataset_id != null) return row.origin.cluster_dataset_id
  if (row.origin.input_dataset_id != null && row.origin.type === 'experiment_result') {
    return row.origin.input_dataset_id
  }
  return row.dataset_id ?? row.source?.dataset_id ?? 0
}

function lineageRole(type: TreeOrigin['type']): number {
  // Keep upload/derived tables above experiment outputs that hang off them.
  if (type === 'experiment_result') return 1
  return 0
}

function sortSources(a: TreeSourceRow, b: TreeSourceRow) {
  const ga = groupName(a.origin)
  const gb = groupName(b.origin)
  if (ga !== gb) return ga.localeCompare(gb)
  const ca = clusterId(a)
  const cb = clusterId(b)
  if (ca !== cb) return ca - cb
  const ra = lineageRole(a.origin.type)
  const rb = lineageRole(b.origin.type)
  if (ra !== rb) return ra - rb
  return a.origin.label.localeCompare(b.origin.label)
}

function sourceSearchText(s: TreeSourceRow): string {
  return [
    s.table_name,
    s.origin.label,
    s.origin.input_dataset_name,
    s.origin.group_name ?? '',
    s.origin.type,
    s.strategy ?? '',
    ...s.built_algorithms,
  ].join(' ').toLowerCase()
}

function OriginCell({ origin }: { origin: TreeOrigin }) {
  if (origin.type === 'experiment_result' && origin.experiment_id != null) {
    return (
      <Anchor component={Link} to={`/experiments/${origin.experiment_id}`} size="sm">
        {origin.label}
      </Anchor>
    )
  }
  return <Text size="sm">{origin.label}</Text>
}

function Metric({ value, best, digits = 2 }: { value: number | null; best: number | null; digits?: number }) {
  if (value == null) return <Table.Td ta="right" c="gray.4">—</Table.Td>
  const isBest = best != null && value === best
  return (
    <Table.Td ta="right" fw={isBest ? 700 : 400} c={isBest ? 'green.8' : undefined}>
      {fmt(value, digits)}
    </Table.Td>
  )
}

export default function ComparisonPage() {
  const { loading: authLoading } = useAuth()
  const [searchParams, setSearchParams] = useSearchParams()
  const filterExpId = searchParams.get('experiment_id')
    ? Number(searchParams.get('experiment_id'))
    : null

  const [algorithms, setAlgorithms] = useState([{ name: 'id3', description: 'ID3' }])
  const [sources, setSources] = useState<TreeSourceRow[]>([])
  const [sourcesLoading, setSourcesLoading] = useState(false)
  const [sourcesError, setSourcesError] = useState(false)
  const [runs, setRuns] = useState<TreeRunRow[]>([])
  const [runsLoading, setRunsLoading] = useState(true)
  const [runsError, setRunsError] = useState(false)

  const [buildOpen, setBuildOpen] = useState(false)
  const [sourceSearch, setSourceSearch] = useState('')
  const [pendingBuild, setPendingBuild] = useState<string | null>(null)
  const [pendingAlgorithm, setPendingAlgorithm] = useState('id3')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [treeTexts, setTreeTexts] = useState<Record<number, string>>({})
  const [loadingText, setLoadingText] = useState<number | null>(null)
  const [building, setBuilding] = useState<string | null>(null)
  const [buildError, setBuildError] = useState('')
  const [compareError, setCompareError] = useState('')
  const [compareInput, setCompareInput] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<number | null>(null)

  const algorithmNames = useMemo(() => algorithms.map(a => a.name), [algorithms])

  const loadAlgorithms = useCallback(async () => {
    try {
      setAlgorithms(await fetchTreeAlgorithms())
    } catch {
      setAlgorithms([{ name: 'id3', description: 'ID3' }])
    }
  }, [])

  const loadSources = useCallback(async () => {
    setSourcesLoading(true)
    setSourcesError(false)
    try {
      setSources(await fetchTreeSources())
    } catch {
      setSourcesError(true)
    } finally {
      setSourcesLoading(false)
    }
  }, [])

  const loadRuns = useCallback(async () => {
    setRunsLoading(true)
    setRunsError(false)
    try {
      const { data } = await api.get('/trees')
      setRuns(data)
    } catch {
      setRunsError(true)
    } finally {
      setRunsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadAlgorithms()
    void loadRuns()
  }, [loadAlgorithms, loadRuns])

  useEffect(() => {
    if (buildOpen) void loadSources()
  }, [buildOpen, loadSources])

  useEffect(() => {
    if (filterExpId != null && !Number.isNaN(filterExpId)) {
      setBuildOpen(true)
    }
  }, [filterExpId])

  const visibleSources = useMemo(() => {
    const q = sourceSearch.trim().toLowerCase()
    const list = sources.filter(s => {
      if (filterExpId != null && !Number.isNaN(filterExpId)) {
        // every source is a dataset now; an experiment's output tables are
        // matched by their experiment_id, not by a "result" kind.
        if (s.experiment_id !== filterExpId) return false
      }
      if (q && !sourceSearchText(s).includes(q)) return false
      return true
    })
    return [...list].sort(sortSources)
  }, [sources, filterExpId, sourceSearch])

  const inputDatasetNames = useMemo(
    () => [...new Set(runs.map(r => groupName(r.origin)))].sort(),
    [runs],
  )

  const compareRuns = useMemo(() => {
    const list = compareInput
      ? runs.filter(r => groupName(r.origin) === compareInput)
      : runs
    return [...list].sort((a, b) => {
      const ga = groupName(a.origin)
      const gb = groupName(b.origin)
      if (ga !== gb) return ga.localeCompare(gb)
      const ca = clusterId({ origin: a.origin, source: a.source })
      const cb = clusterId({ origin: b.origin, source: b.source })
      if (ca !== cb) return ca - cb
      const ra = lineageRole(a.origin.type)
      const rb = lineageRole(b.origin.type)
      if (ra !== rb) return ra - rb
      return a.origin.label.localeCompare(b.origin.label)
    })
  }, [runs, compareInput])

  const best = {
    depth: Math.min(...compareRuns.map(r => r.depth ?? Infinity)),
    avg_depth: Math.min(...compareRuns.map(r => r.avg_depth ?? Infinity)),
    nodes: Math.min(...compareRuns.map(r => r.nodes ?? Infinity)),
  }

  async function loadTreeText(treeRunId: number) {
    if (treeTexts[treeRunId]) return
    setLoadingText(treeRunId)
    try {
      const data = await fetchTreeText(treeRunId)
      setTreeTexts(t => ({ ...t, [treeRunId]: data.text }))
    } catch {
      setTreeTexts(t => ({ ...t, [treeRunId]: 'Could not load tree text.' }))
    } finally {
      setLoadingText(null)
    }
  }

  async function toggleTreeText(treeRunId: number) {
    if (expanded === treeRunId) {
      setExpanded(null)
      return
    }
    setExpanded(treeRunId)
    await loadTreeText(treeRunId)
  }

  async function buildSource(source: TreeSourceRow, algorithm: string) {
    setBuilding(source.source_key)
    setBuildError('')
    try {
      await buildTreeFromSource({
        kind: 'dataset',
        dataset_id: source.dataset_id!,
        algorithm,
      })
      setCompareInput(groupName(source.origin))
      setPendingBuild(null)
      setBuildOpen(false)
      await Promise.all([loadSources(), loadRuns()])
    } catch {
      setBuildError('Tree build failed.')
    } finally {
      setBuilding(null)
    }
  }

  function startBuild(source: TreeSourceRow) {
    setBuildError('')
    setPendingBuild(source.source_key)
    const available = algorithmNames.find(a => !source.built_algorithms.includes(a))
    setPendingAlgorithm(available ?? algorithmNames[0] ?? 'id3')
  }

  function cancelBuild() {
    setPendingBuild(null)
    setBuildError('')
  }

  async function removeTree(treeRunId: number) {
    setDeleting(treeRunId)
    setCompareError('')
    try {
      await deleteTreeRun(treeRunId)
      if (expanded === treeRunId) setExpanded(null)
      await Promise.all([loadSources(), loadRuns()])
    } catch {
      setCompareError('Could not delete the tree.')
    } finally {
      setDeleting(null)
    }
  }

  function closeBuildModal() {
    setBuildOpen(false)
    setBuildError('')
    setSourceSearch('')
    setPendingBuild(null)
    if (filterExpId != null) {
      searchParams.delete('experiment_id')
      setSearchParams(searchParams, { replace: true })
    }
  }

  if (authLoading) {
    return <Container size="xl" py="lg"><Loader size="sm" /></Container>
  }

  return (
    <Container size="xl" py="lg">
      <Group justify="space-between" align="flex-start" mb="md" wrap="wrap">
        <Title order={2}>Build decision tree</Title>
        <Button variant="light" color="teal" onClick={() => setBuildOpen(true)}>
          Build tree
        </Button>
      </Group>

      {compareError && <Text size="sm" c="red" mb="sm">{compareError}</Text>}

      <Paper withBorder p="md">
        <Group justify="space-between" align="flex-end" mb="sm" wrap="wrap">
          <Title order={3}>Compare</Title>
          {inputDatasetNames.length > 1 && (
            <Select
              size="xs"
              w={220}
              clearable
              placeholder="All input datasets"
              data={inputDatasetNames}
              value={compareInput}
              onChange={setCompareInput}
            />
          )}
        </Group>

        {runsLoading && <Loader size="sm" />}
        {runsError && <Text c="red" size="sm">Failed to load built trees.</Text>}

        {!runsLoading && !runsError && compareRuns.length === 0 && (
          <Text size="sm" c="dimmed">
            No trees yet.{' '}
            <Anchor component="button" type="button" size="sm" onClick={() => setBuildOpen(true)}>
              Build one
            </Anchor>
          </Text>
        )}

        {!runsLoading && !runsError && compareRuns.length > 0 && (
          <Box style={{ overflowX: 'auto' }}>
            <Table withTableBorder withColumnBorders fz="sm">
              <Table.Thead>
                <Table.Tr bg="gray.0">
                  <Table.Th>Table</Table.Th>
                  <Table.Th>Origin</Table.Th>
                  <Table.Th ta="right">Rows</Table.Th>
                  <Table.Th ta="right">γ</Table.Th>
                  <Table.Th ta="right">Depth</Table.Th>
                  <Table.Th ta="right">Avg depth</Table.Th>
                  <Table.Th ta="right">Nodes</Table.Th>
                  <Table.Th w={60} />
                  <Table.Th w={60} />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {compareRuns.map(r => (
                  <Fragment key={r.tree_run_id}>
                    <Table.Tr>
                      <Table.Td>
                        {r.origin.type === 'derived' ? (
                          <>
                            <Text size="sm" fw={600}>{r.table_name}</Text>
                            <Text size="xs" c="dimmed">input: {r.origin.input_dataset_name}</Text>
                          </>
                        ) : (
                          displayTableName(r)
                        )}
                      </Table.Td>
                      <Table.Td><OriginCell origin={r.origin} /></Table.Td>
                      <Table.Td ta="right">{fmt(r.rows, 0)}</Table.Td>
                      <Table.Td ta="right">{r.gamma == null ? '—' : r.gamma.toFixed(3)}</Table.Td>
                      <Metric value={r.depth} best={best.depth} digits={0} />
                      <Metric value={r.avg_depth} best={best.avg_depth} />
                      <Metric value={r.nodes} best={best.nodes} digits={0} />
                      <Table.Td>
                        <Button
                          size="compact-xs"
                          variant="subtle"
                          loading={loadingText === r.tree_run_id}
                          onClick={() => void toggleTreeText(r.tree_run_id)}
                        >
                          {expanded === r.tree_run_id ? 'Hide' : 'Text'}
                        </Button>
                      </Table.Td>
                      <Table.Td>
                        <Button
                          size="compact-xs"
                          variant="subtle"
                          color="red"
                          loading={deleting === r.tree_run_id}
                          onClick={() => void removeTree(r.tree_run_id)}
                        >
                          Delete
                        </Button>
                      </Table.Td>
                    </Table.Tr>
                    {expanded === r.tree_run_id && treeTexts[r.tree_run_id] && (
                      <Table.Tr>
                        <Table.Td colSpan={9} bg="gray.0" p="sm">
                          <TreeTextPanel text={treeTexts[r.tree_run_id]} />
                        </Table.Td>
                      </Table.Tr>
                    )}
                  </Fragment>
                ))}
              </Table.Tbody>
            </Table>
          </Box>
        )}
      </Paper>

      <Modal
        opened={buildOpen}
        onClose={closeBuildModal}
        title="Build tree"
        size="xl"
      >
        {filterExpId != null && !Number.isNaN(filterExpId) && (
          <Text size="sm" mb="sm">
            Tables from{' '}
            <Anchor component={Link} to={`/experiments/${filterExpId}`}>experiment #{filterExpId}</Anchor>
          </Text>
        )}

        <Group align="flex-end" mb="md" wrap="wrap">
          <TextInput
            label="Search"
            placeholder="Table, origin, input dataset…"
            size="sm"
            w={320}
            value={sourceSearch}
            onChange={e => setSourceSearch(e.currentTarget.value)}
          />
          {sourceSearch && (
            <Text size="xs" c="dimmed" pb={8}>
              {visibleSources.length} of {sources.length}
            </Text>
          )}
        </Group>

        {buildError && <Text size="sm" c="red" mb="sm">{buildError}</Text>}

        {sourcesLoading && <Loader size="sm" />}
        {sourcesError && <Text c="red" size="sm">Failed to load tables.</Text>}

        {!sourcesLoading && !sourcesError && visibleSources.length === 0 && sources.length > 0 && (
          <Text size="sm" c="dimmed">No tables match your search.</Text>
        )}

        {!sourcesLoading && !sourcesError && sources.length === 0 && (
          <Text size="sm" c="dimmed">
            Nothing to build on — upload a dataset or run an experiment first.
          </Text>
        )}

        {!sourcesLoading && !sourcesError && visibleSources.length > 0 && (
          <Box style={{ overflowX: 'auto' }}>
            <Table withTableBorder withColumnBorders fz="sm">
              <Table.Thead>
                <Table.Tr bg="gray.0">
                  <Table.Th>Table</Table.Th>
                  <Table.Th>Origin</Table.Th>
                  <Table.Th ta="right">Rows</Table.Th>
                  <Table.Th ta="right">γ</Table.Th>
                  <Table.Th>Built trees</Table.Th>
                  <Table.Th w={180} />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {visibleSources.map(s => {
                  const isPending = pendingBuild === s.source_key
                  const alreadyBuilt = s.built_algorithms.includes(pendingAlgorithm)
                  const allBuilt = algorithmNames.length > 0
                    && algorithmNames.every(a => s.built_algorithms.includes(a))
                  const algorithmOptions = algorithms.map(a => ({
                    value: a.name,
                    label: a.name.toUpperCase(),
                    disabled: s.built_algorithms.includes(a.name),
                  }))

                  return (
                  <Table.Tr key={s.source_key}>
                    <Table.Td>
                      {s.origin.type === 'derived' ? (
                        <>
                          <Text size="sm" fw={600}>{s.table_name}</Text>
                          <Text size="xs" c="dimmed">input: {s.origin.input_dataset_name}</Text>
                        </>
                      ) : (
                        s.table_name
                      )}
                    </Table.Td>
                    <Table.Td><OriginCell origin={s.origin} /></Table.Td>
                    <Table.Td ta="right">{fmt(s.rows, 0)}</Table.Td>
                    <Table.Td ta="right">{s.gamma == null ? '—' : s.gamma.toFixed(3)}</Table.Td>
                    <Table.Td>
                      {s.built_algorithms.length === 0 ? (
                        <Text size="sm" c="dimmed">—</Text>
                      ) : (
                        <Group gap={4}>
                          {s.built_algorithms.map(a => (
                            <Badge key={a} variant="light" color="gray" size="xs">{a.toUpperCase()}</Badge>
                          ))}
                        </Group>
                      )}
                    </Table.Td>
                    <Table.Td>
                      {isPending ? (
                        <Stack gap={4}>
                          <Group gap={6} wrap="nowrap">
                            <Select
                              size="xs"
                              w={100}
                              data={algorithmOptions}
                              value={pendingAlgorithm}
                              onChange={v => v && setPendingAlgorithm(v)}
                              allowDeselect={false}
                            />
                            <Button
                              size="compact-xs"
                              variant="light"
                              color="teal"
                              loading={building === s.source_key}
                              disabled={alreadyBuilt}
                              onClick={() => void buildSource(s, pendingAlgorithm)}
                            >
                              Run
                            </Button>
                            <Button
                              size="compact-xs"
                              variant="subtle"
                              disabled={building === s.source_key}
                              onClick={cancelBuild}
                            >
                              Cancel
                            </Button>
                          </Group>
                          {alreadyBuilt && (
                            <Text size="xs" c="dimmed">Tree already built for this algorithm.</Text>
                          )}
                          {allBuilt && !alreadyBuilt && (
                            <Text size="xs" c="dimmed">Pick an algorithm without a tree yet.</Text>
                          )}
                        </Stack>
                      ) : (
                        <Button
                          size="compact-xs"
                          variant="light"
                          color="teal"
                          onClick={() => startBuild(s)}
                        >
                          Build
                        </Button>
                      )}
                    </Table.Td>
                  </Table.Tr>
                  )
                })}
              </Table.Tbody>
            </Table>
          </Box>
        )}
      </Modal>
    </Container>
  )
}
