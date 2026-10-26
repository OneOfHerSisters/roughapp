import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  Container, Grid, Paper, Title, Text, Table, Button,
  Select, NumberInput, Checkbox, Stack, Loader, Group,
} from '@mantine/core'
import { api, fetchDatasets, fetchExperiments, runExperiment } from '../api/client'
import type { Dataset, Experiment, Strategy } from '../types/api'
import { formatExperimentParams, experimentUsesRandomSeed } from '../utils/experimentParams'

const STRATEGY_OPTIONS: { value: Strategy; label: string }[] = [
  { value: 'mcd',                 label: 'MCD — Most Common Decision' },
  { value: 'rough-approximation', label: 'Rough Approximation' },
  { value: 'gd',                  label: 'GD — Generalized Decision' },
]

const MCD_TIE_OPTIONS = [
  { value: 'min', label: 'min — smallest decision value' },
  { value: 'keep', label: 'keep — leave class unresolved' },
  { value: 'first', label: 'first — first candidate' },
  { value: 'random', label: 'random' },
  { value: 'raise', label: 'raise — fail on tie' },
]

const QR_TIE_OPTIONS = [
  { value: 'keep', label: 'keep — leave class unresolved' },
  { value: 'random', label: 'random — pick one decision' },
  { value: 'raise', label: 'raise — fail on tie' },
]

export default function ExperimentsPage() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [experimentsLoading, setExperimentsLoading] = useState(true)
  const [experimentsError, setExperimentsError] = useState(false)
  const [runLoading, setRunLoading] = useState(false)
  const [deleteLoadingId, setDeleteLoadingId] = useState<number | null>(null)

  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [strategies, setStrategies] = useState<Strategy[]>(['mcd', 'rough-approximation', 'gd'])
  const [randomState, setRandomState] = useState<string | number>(42)
  const [tiePolicy, setTiePolicy] = useState('min')
  const [qrTiePolicy, setQrTiePolicy] = useState('keep')
  const [error, setError] = useState('')

  const loadData = useCallback(async () => {
    setExperimentsLoading(true)
    setExperimentsError(false)
    try {
      const [ds, exps] = await Promise.all([fetchDatasets(), fetchExperiments()])
      setDatasets(ds)
      setExperiments(exps)
    } catch {
      setExperimentsError(true)
    } finally {
      setExperimentsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadData()
  }, [loadData])

  // Preselect a dataset when navigated to with ?dataset_id=N (e.g. the
  // "Remove inconsistency" button on the Datasets page). Consume the
  // param so a later manual change to the Dataset select isn't overridden.
  useEffect(() => {
    const raw = searchParams.get('dataset_id')
    if (raw == null) return
    const id = Number(raw)
    if (!Number.isNaN(id)) setDatasetId(id)
    searchParams.delete('dataset_id')
    setSearchParams(searchParams, { replace: true })
  }, [searchParams, setSearchParams])

  async function handleDeleteExperiment(id: number) {
    setDeleteLoadingId(id)
    try {
      await api.delete(`/experiments/${id}`)
      await loadData()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      alert(typeof detail === 'string' ? detail : 'Failed to delete the experiment.')
    } finally {
      setDeleteLoadingId(null)
    }
  }

  async function handleRun() {
    if (!datasetId) { setError('Select a dataset first.'); return }
    if (strategies.length === 0) { setError('Select at least one strategy.'); return }
    setError('')
    setRunLoading(true)
    try {
      const data = await runExperiment({
        dataset_id: datasetId,
        strategies,
        random_state: experimentUsesRandomSeed(strategies, tiePolicy, qrTiePolicy)
          ? (randomState !== '' ? Number(randomState) : undefined)
          : undefined,
        tie_policy: tiePolicy,
        qr_tie_policy: qrTiePolicy,
      })
      await loadData()
      navigate(`/experiments/${data.id}`)
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(typeof detail === 'string' ? detail : 'Experiment failed — check the server logs.')
    } finally {
      setRunLoading(false)
    }
  }

  function toggleStrategy(s: Strategy) {
    setStrategies(prev => prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s])
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    void handleRun()
  }

  const selectable = datasets.filter(d => d.provenance?.kind !== 'experiment_output')
  const originals = selectable.filter(d => !d.provenance || d.provenance.kind === 'upload')
  const introduced = selectable.filter(d => d.provenance?.kind === 'derived')

  const datasetOptions = [
    {
      group: 'Original',
      items: originals.map(ds => ({ value: String(ds.id), label: `${ds.name} (${ds.rows} rows)` })),
    },
    {
      group: 'Derived inconsistent tables',
      items: introduced.map(ds => ({
        value: String(ds.id),
        label: `${ds.name} (${ds.rows} rows${ds.gamma_actual != null ? `, γ=${ds.gamma_actual}` : ''})`,
      })),
    },
  ].filter(g => g.items.length > 0)

  return (
    <Container size="lg" py="lg">
      <Grid gap="xl">
        <Grid.Col span={{ base: 12, md: 4 }}>
          <Title order={2} mb="md">Configure strategies</Title>
          <Paper withBorder p="md">
            <form onSubmit={handleSubmit}>
              <Stack gap="sm">
                <Select
                  label="Dataset"
                  placeholder="Choose…"
                  data={datasetOptions}
                  value={datasetId != null ? String(datasetId) : null}
                  onChange={v => setDatasetId(v != null ? Number(v) : null)}
                  searchable
                />

                <div>
                  <Text size="sm" fw={500} mb={4}>Strategies</Text>
                  <Stack gap={6}>
                    {STRATEGY_OPTIONS.map(({ value, label }) => (
                      <Checkbox
                        key={value} size="sm"
                        label={label}
                        checked={strategies.includes(value)}
                        onChange={() => toggleStrategy(value)}
                      />
                    ))}
                  </Stack>
                </div>

                {experimentUsesRandomSeed(strategies, tiePolicy, qrTiePolicy) && (
                  <NumberInput
                    label="Random state"
                    description="Random tie-breaking"
                    value={randomState}
                    onChange={setRandomState}
                    w={140}
                  />
                )}

                {strategies.includes('mcd') && (
                  <Select
                    label="MCD tie-breaking"
                    data={MCD_TIE_OPTIONS}
                    value={tiePolicy}
                    onChange={v => v && setTiePolicy(v)}
                    allowDeselect={false}
                  />
                )}

                {strategies.includes('rough-approximation') && (
                  <Select
                    label="Rough approximation tie-breaking"
                    data={QR_TIE_OPTIONS}
                    value={qrTiePolicy}
                    onChange={v => v && setQrTiePolicy(v)}
                    allowDeselect={false}
                  />
                )}

                {error && <Text c="red" size="sm">{error}</Text>}

                <Button type="submit" loading={runLoading} size="sm">
                  Run {strategies.length} strateg{strategies.length === 1 ? 'y' : 'ies'}
                </Button>
              </Stack>
            </form>
          </Paper>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 8 }}>
          <Title order={2} mb="md">History</Title>
          {experimentsLoading ? (
            <Loader size="sm" />
          ) : experimentsError ? (
            <Text c="red" size="sm">Could not load experiments.</Text>
          ) : experiments.length === 0 ? (
            <Text c="dimmed" size="sm">No experiments yet.</Text>
          ) : (
            <Table withTableBorder highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>#</Table.Th>
                  <Table.Th>Dataset</Table.Th>
                  <Table.Th>γ before</Table.Th>
                  <Table.Th>Strategies</Table.Th>
                  <Table.Th>Parameters</Table.Th>
                  <Table.Th>Date</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {experiments.map(exp => (
                  <Table.Tr key={exp.id}>
                    <Table.Td c="dimmed">{exp.id}</Table.Td>
                    <Table.Td>
                      <Text size="sm">{exp.dataset_name}</Text>
                    </Table.Td>
                    <Table.Td>{exp.gamma_before != null ? exp.gamma_before.toFixed(3) : '—'}</Table.Td>
                    <Table.Td c="dimmed">{exp.results.map(r => r.strategy).join(', ')}</Table.Td>
                    <Table.Td c="dimmed" fz="xs">{formatExperimentParams(exp)}</Table.Td>
                    <Table.Td c="dimmed" fz="xs">{new Date(exp.created_at).toLocaleString()}</Table.Td>
                    <Table.Td>
                      <Group gap={4} wrap="nowrap" justify="flex-end">
                        <Button
                          variant="light" size="compact-xs"
                          onClick={() => navigate(`/experiments/${exp.id}`)}
                        >
                          View statistics
                        </Button>
                        <Button
                          variant="subtle" color="red" size="compact-xs"
                          loading={deleteLoadingId === exp.id}
                          onClick={() => {
                            if (confirm(`Delete experiment #${exp.id}? Its result tables are kept.`)) {
                              void handleDeleteExperiment(exp.id)
                            }
                          }}
                        >
                          Delete
                        </Button>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          )}
        </Grid.Col>
      </Grid>
    </Container>
  )
}
