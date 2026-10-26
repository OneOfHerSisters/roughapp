import type { ExperimentResult } from '../types/api'

function csvEscape(value: string | number | boolean): string {
  const s = String(value)
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

function toCsv(rows: Record<string, string | number | boolean>[], columns: string[]): string {
  const header = columns.join(',')
  const body = rows.map(row =>
    columns.map(col => csvEscape(row[col] ?? '')).join(','),
  )
  return [header, ...body].join('\n') + '\n'
}

export function buildDetailsCsv(result: ExperimentResult): string | null {
  const d = result.details
  if (!d) return null

  if (result.strategy === 'mcd') {
    const classes = (d.conflict_classes ?? []) as {
      key: Record<string, string>
      counts: Record<string, number>
      tied: boolean
      chosen: string[]
    }[]
    const rows = classes.map(cls => {
      const profile = Object.entries(cls.key).map(([k, v]) => `${k}=${v}`).join(', ')
      const decisions = Object.entries(cls.counts)
        .map(([dec, cnt]) => `dec ${dec}: ${cnt}×`)
        .join(', ')
      const chosen = cls.chosen.join(', ')
      const outcome = cls.tied
        ? `TIE — kept all (dec ${chosen})`
        : `chose ${chosen}`
      return { profile, decisions, outcome }
    })
    return toCsv(rows, ['profile', 'decisions', 'outcome'])
  }

  if (result.strategy === 'rough-approximation') {
    const classes = (d.conflict_classes ?? []) as {
      key: Record<string, string>
      decisions: { value: string; count: number; alpha: number }[]
      removed: string[]
      tie: boolean
    }[]
    const rows: Record<string, string | number | boolean>[] = []
    for (const cls of classes) {
      const profile = Object.entries(cls.key).map(([k, v]) => `${k}=${v}`).join(', ')
      for (const dec of cls.decisions) {
        const isRemoved = cls.removed.includes(dec.value)
        let status = 'kept'
        if (cls.tie) status = 'tie, kept'
        else if (isRemoved) status = 'removed (lowest α)'
        rows.push({
          profile,
          decision: dec.value,
          count: dec.count,
          alpha: dec.alpha,
          status,
          class_tie: cls.tie,
        })
      }
    }
    return toCsv(rows, ['profile', 'decision', 'count', 'alpha', 'status', 'class_tie'])
  }

  if (result.strategy === 'gd') {
    const mapping = (d.full_mapping ?? []) as {
      code: number
      decisions: number[]
      conflict: boolean
    }[]
    const rows = mapping.map(row => ({
      code: row.code,
      decisions: `{${row.decisions.join(', ')}}`,
      conflict: row.conflict,
    }))
    return toCsv(rows, ['code', 'decisions', 'conflict'])
  }

  return null
}

export function downloadDetailsCsv(result: ExperimentResult, expId: number) {
  const csv = buildDetailsCsv(result)
  if (!csv) return
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `exp${expId}_${result.strategy}_details.csv`
  a.click()
  URL.revokeObjectURL(url)
}
