import type { Experiment, ExperimentResult, Strategy } from '../types/api'

type ParamsInput = Pick<Experiment, 'random_state' | 'tie_policy' | 'qr_tie_policy'> & {
  results: Array<Pick<ExperimentResult, 'strategy'>>
}

function usesRandomSeed(input: ParamsInput): boolean {
  const strategies = new Set((input.results ?? []).map(r => r.strategy))
  return (
    (strategies.has('mcd') && input.tie_policy === 'random') ||
    (strategies.has('rough-approximation') && input.qr_tie_policy === 'random')
  )
}

export function formatExperimentParams(exp: Experiment): string {
  return formatParams(exp)
}

export function formatParams(input: ParamsInput): string {
  const parts: string[] = []
  if (usesRandomSeed(input) && input.random_state != null) {
    parts.push(`seed ${input.random_state}`)
  }
  if (input.tie_policy) parts.push(`MCD tie: ${input.tie_policy}`)
  if (input.qr_tie_policy) parts.push(`QR tie: ${input.qr_tie_policy}`)
  return parts.length ? parts.join(' · ') : '—'
}

export function experimentUsesRandomSeed(
  strategies: Strategy[] | string[],
  tiePolicy: string,
  qrTiePolicy: string,
): boolean {
  return (
    (strategies.includes('mcd') && tiePolicy === 'random') ||
    (strategies.includes('rough-approximation') && qrTiePolicy === 'random')
  )
}
