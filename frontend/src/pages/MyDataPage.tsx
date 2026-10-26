import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Container, Title, Text, Table, Button, Anchor, Stack } from '@mantine/core'
import { fetchDatasets, fetchExperiments } from '../api/client'
import { api } from '../api/client'
import { useAuth } from '../auth'
import type { Dataset, Experiment } from '../types/api'
import { formatExperimentParams } from '../utils/experimentParams'

export default function MyDataPage() {
  const { user } = useAuth()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [experiments, setExperiments] = useState<Experiment[]>([])

  const loadData = useCallback(async () => {
    try {
      const [ds, exps] = await Promise.all([fetchDatasets(), fetchExperiments()])
      setDatasets(ds)
      setExperiments(exps)
    } catch {
      setDatasets([])
      setExperiments([])
    }
  }, [])

  useEffect(() => {
    void loadData()
  }, [loadData])

  async function handleDeleteDataset(id: number, cascade = false) {
    try {
      await api.delete(`/datasets/${id}${cascade ? '?cascade=true' : ''}`)
      await loadData()
    } catch (err: unknown) {
      const data = (err as {
        response?: { data?: { detail?: string; cascade_required?: boolean; manifest?: { experiments?: { id: number }[] } } }
      })?.response?.data
      if (data?.cascade_required) {
        const exps = data.manifest?.experiments ?? []
        const list = exps.map(e => `#${e.id}`).join(', ')
        const ok = confirm(
          `This dataset is the input of ${exps.length} experiment${exps.length === 1 ? '' : 's'} (${list}).\n` +
          'Deleting it will also delete those experiment records; their result tables are kept.\n\nDelete anyway?'
        )
        if (ok) await handleDeleteDataset(id, true)
        return
      }
      alert(typeof data?.detail === 'string' ? data.detail : 'Failed to delete the dataset.')
    }
  }

  async function handleDeleteExperiment(id: number) {
    try {
      await api.delete(`/experiments/${id}`)
      await loadData()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      alert(typeof detail === 'string' ? detail : 'Failed to delete the experiment.')
    }
  }

  const myDatasets = datasets.filter(ds => {
    const kind = ds.provenance?.kind
    return kind === 'derived' || kind === 'experiment_output' || ds.storage_path?.includes('uploads/')
  })

  function originLabel(ds: Dataset) {
    const kind = ds.provenance?.kind
    if (kind === 'experiment_output') return 'experiment result'
    if (kind === 'derived' && ds.provenance?.source_dataset_id != null) {
      return `derived from #${ds.provenance.source_dataset_id}`
    }
    return 'upload'
  }

  return (
    <Container size="lg" py="lg">
      <Stack gap="lg">
        <div>
          <Title order={2}>My data</Title>
          <Text size="sm" c="dimmed">
            {user ? user.email : 'Guest session — this data is kept in your browser until it expires.'}
          </Text>
        </div>

        <div>
          <Title order={3} mb="xs">My datasets</Title>
          {myDatasets.length === 0 ? (
            <Text size="sm" c="dimmed">
              No datasets yet. <Anchor component={Link} to="/" size="sm">Upload one</Anchor> or{' '}
              <Anchor component={Link} to="/" size="sm">derive an inconsistent decision table</Anchor> on a built-in dataset.
            </Text>
          ) : (
            <Table withTableBorder>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Origin</Table.Th>
                  <Table.Th>Rows</Table.Th>
                  <Table.Th>Cols</Table.Th>
                  <Table.Th>γ</Table.Th>
                  <Table.Th>Uploaded</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {myDatasets.map(ds => (
                  <Table.Tr key={ds.id}>
                    <Table.Td>{ds.name}</Table.Td>
                    <Table.Td c="dimmed" fz="sm">{originLabel(ds)}</Table.Td>
                    <Table.Td>{ds.rows}</Table.Td>
                    <Table.Td>{ds.cols}</Table.Td>
                    <Table.Td c="dimmed">{ds.gamma_actual != null ? ds.gamma_actual : '—'}</Table.Td>
                    <Table.Td c="dimmed">{new Date(ds.created_at).toLocaleDateString()}</Table.Td>
                    <Table.Td>
                      <Button
                        variant="subtle" color="red"
                        onClick={() => { if (confirm(`Delete "${ds.name}"?`)) void handleDeleteDataset(ds.id) }}
                      >
                        Delete
                      </Button>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          )}
        </div>

        <div>
          <Title order={3} mb="xs">My experiments</Title>
          {experiments.length === 0 ? (
            <Text size="sm" c="dimmed">
              No experiments yet. <Anchor component={Link} to="/experiments" size="sm">Run one</Anchor>.
            </Text>
          ) : (
            <Table withTableBorder>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>#</Table.Th>
                  <Table.Th>Dataset</Table.Th>
                  <Table.Th>Strategies</Table.Th>
                  <Table.Th>Parameters</Table.Th>
                  <Table.Th>Date</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {experiments.map(exp => (
                  <Table.Tr key={exp.id}>
                    <Table.Td>
                      <Anchor component={Link} to={`/experiments/${exp.id}`} size="sm">{exp.id}</Anchor>
                    </Table.Td>
                    <Table.Td>{exp.dataset_name}</Table.Td>
                    <Table.Td c="dimmed">{exp.results.map(r => r.strategy).join(', ')}</Table.Td>
                    <Table.Td c="dimmed" fz="xs">{formatExperimentParams(exp)}</Table.Td>
                    <Table.Td c="dimmed">{new Date(exp.created_at).toLocaleDateString()}</Table.Td>
                    <Table.Td>
                      <Button
                        variant="subtle" color="red"
                        onClick={() => { if (confirm(`Delete experiment #${exp.id}?`)) void handleDeleteExperiment(exp.id) }}
                      >
                        Delete
                      </Button>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          )}
        </div>
      </Stack>
    </Container>
  )
}
