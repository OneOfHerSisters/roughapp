import { useState, Fragment, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  Container, Title, Text, Anchor, Table, Button,
  Group, Box, Loader, Modal, Paper,
} from '@mantine/core'
import { fetchExperiment, previewResult, downloadAuthenticated } from '../api/client'
import { formatExperimentParams } from '../utils/experimentParams'
import { downloadDetailsCsv } from '../utils/detailsCsv'
import type { Experiment, ExperimentResult } from '../types/api'

const LABELS: Record<string, string> = {
  mcd: 'MCD',
  'rough-approximation': 'Rough approximation',
  gd: 'GD',
}

function fmt(v: number | null | undefined) {
  return v == null ? '—' : v.toFixed(3)
}

interface Preview {
  columns: string[]
  rows: string[][]
  total: number
  offset: number
  limit: number
}

function ResultPreview({ expId, strategy }: { expId: number; strategy: string }) {
  const [data, setData] = useState<Preview | 'loading' | 'error' | null>(null)
  const [open, setOpen] = useState(false)

  async function load(offset = 0) {
    setOpen(true)
    setData('loading')
    try {
      const p = await previewResult(expId, strategy, offset)
      setData(p)
    } catch {
      setData('error')
    }
  }

  if (data === null) {
    return <Button variant="default" onClick={() => load(0)}>View table</Button>
  }

  return (
    <>
      <Button variant="default" onClick={() => setOpen(true)}>View table</Button>
      <Modal
        opened={open}
        onClose={() => setOpen(false)}
        title={`Result table — ${LABELS[strategy] ?? strategy}`}
        size="xl"
        centered
      >
        {data === 'loading' && <Loader size="sm" />}
        {data === 'error' && <Text size="sm" c="red">Failed to load preview.</Text>}
        {data !== 'loading' && data !== 'error' && (
          <Box>
            <Text size="xs" c="dimmed" mb={4}>
              Showing {data.offset + 1}–{Math.min(data.offset + data.limit, data.total)} of {data.total} rows
            </Text>
            <Box style={{ overflowX: 'auto' }}>
              <Table withTableBorder withColumnBorders fz="xs" w="auto" style={{ whiteSpace: 'nowrap' }}>
                <Table.Thead>
                  <Table.Tr>{data.columns.map(col => <Table.Th key={col}>{col}</Table.Th>)}</Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {data.rows.map((row, i) => (
                    <Table.Tr key={i}>{row.map((cell, j) => <Table.Td key={j}>{cell}</Table.Td>)}</Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Box>
            <Group gap="xs" mt={4}>
              {data.offset > 0 && (
                <Button variant="default" onClick={() => load(data.offset - data.limit)}>← Prev</Button>
              )}
              {data.offset + data.limit < data.total && (
                <Button variant="default" onClick={() => load(data.offset + data.limit)}>Next →</Button>
              )}
            </Group>
          </Box>
        )}
      </Modal>
    </>
  )
}

function DownloadButton({ path, filename, label }: { path: string; filename: string; label: string }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleClick() {
    setLoading(true)
    setError('')
    try {
      await downloadAuthenticated(path, filename)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Download failed'
      setError(msg === 'Download failed' ? 'Download failed — retry' : msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Anchor component="button" type="button" onClick={handleClick} size="xs" disabled={loading}>
      {loading ? 'Downloading…' : error || label}
    </Anchor>
  )
}

function DetailsDownloadButton({ result, expId }: { result: ExperimentResult; expId: number }) {
  if (!result.details) return null
  return (
    <Button
      variant="default"
      size="compact-xs"
      onClick={() => downloadDetailsCsv(result, expId)}
    >
      Download details CSV
    </Button>
  )
}

export default function ResultsPage() {
  const { id } = useParams()
  const [exp, setExp] = useState<Experiment | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState(false)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!id) {
      setExp(null)
      setLoading(false)
      setLoadError(true)
      return
    }
    let cancelled = false
    setLoading(true)
    setLoadError(false)
    fetchExperiment(Number(id))
      .then(data => { if (!cancelled) setExp(data) })
      .catch(() => { if (!cancelled) setLoadError(true) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [id])

  function toggle(strategy: string) {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(strategy)) next.delete(strategy)
      else next.add(strategy)
      return next
    })
  }

  if (loading) return <Container size="xl" py="lg"><Loader size="sm" /></Container>
  if (loadError || !exp) {
    return (
      <Container size="xl" py="lg">
        <Text c="red" size="sm">Experiment not found or you don&apos;t have access.</Text>
        <Anchor component={Link} to="/experiments" size="sm" mt="xs" display="inline-block">
          ← Back to experiments
        </Anchor>
      </Container>
    )
  }

  const results = exp.results ?? []
  const gammaBefore = exp.gamma_before ?? null
  const alphaBefore = exp.alpha_before ?? null
  const rowsBefore = exp.rows_before ?? null

  const expId = Number(id)
  const params = formatExperimentParams(exp)

  return (
    <Container size="xl" py="lg">
      <Button
        component={Link}
        to="/experiments"
        variant="light"
        color="teal"
        size="xs"
        mb="sm"
      >
        Back to experiments
      </Button>
      <Title order={2}>Experiment #{exp.id} — {exp.dataset_name}</Title>
      <Text size="sm" c="dimmed" mb="xs">
        {new Date(exp.created_at).toLocaleString()}
        {params !== '—' ? ` · ${params}` : ''}
        {gammaBefore != null ? ` · γ = ${gammaBefore.toFixed(4)}` : ''}
      </Text>

      <Paper withBorder p="md" mt="lg">
        <Table withTableBorder withColumnBorders fz="sm">
          <Table.Thead>
            <Table.Tr bg="gray.0">
              <Table.Th w={28} />
              <Table.Th>Strategy</Table.Th>
              <Table.Th>γ before</Table.Th><Table.Th>γ after</Table.Th>
              <Table.Th>α before</Table.Th><Table.Th>α after</Table.Th>
              <Table.Th>Rows before</Table.Th><Table.Th>Rows after</Table.Th>
              <Table.Th>ms</Table.Th><Table.Th>Result</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {results.map((r) => {
              const open = expanded.has(r.strategy)
              return (
                <Fragment key={r.strategy}>
                  <Table.Tr>
                    <Table.Td ta="center" style={{ cursor: 'pointer' }} onClick={() => toggle(r.strategy)}>
                      {open ? '▾' : '▸'}
                    </Table.Td>
                    <Table.Td>{LABELS[r.strategy] ?? r.strategy}</Table.Td>
                    <Table.Td>{fmt(gammaBefore)}</Table.Td><Table.Td>{fmt(r.gamma_after)}</Table.Td>
                    <Table.Td>{fmt(alphaBefore)}</Table.Td><Table.Td>{fmt(r.alpha_after)}</Table.Td>
                    <Table.Td>{rowsBefore ?? '—'}</Table.Td><Table.Td>{r.rows_after}</Table.Td>
                    <Table.Td>{r.time_ms}</Table.Td>
                    <Table.Td style={{ whiteSpace: 'nowrap' }}>
                      {r.has_result && (
                        <DownloadButton
                          path={`/experiments/${expId}/results/${r.strategy}/csv`}
                          filename={`exp${expId}_${r.strategy}.csv`}
                          label="Download CSV"
                        />
                      )}
                    </Table.Td>
                  </Table.Tr>
                  {open && (
                    <Table.Tr>
                      <Table.Td />
                      <Table.Td colSpan={9} bg="gray.0" p="sm">
                        <Group gap="sm">
                          <DetailsDownloadButton result={r} expId={expId} />
                          {r.has_result && (
                            <ResultPreview expId={expId} strategy={r.strategy} />
                          )}
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  )}
                </Fragment>
              )
            })}
          </Table.Tbody>
        </Table>
      </Paper>

    </Container>
  )
}
