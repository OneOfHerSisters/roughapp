import { Box, Table, Text, Badge } from '@mantine/core'
import type { DecisionRule } from '../api/client'

export default function TreeRulesTable({ rules }: { rules: DecisionRule[] }) {
  if (rules.length === 0) {
    return <Text size="sm" c="dimmed">Нет правил.</Text>
  }

  return (
    <Box style={{ overflowX: 'auto' }}>
      <Table withTableBorder withColumnBorders fz="xs">
        <Table.Thead>
          <Table.Tr bg="gray.0">
            <Table.Th>IF</Table.Th>
            <Table.Th>THEN</Table.Th>
            <Table.Th ta="right" w={70}>Support</Table.Th>
            <Table.Th ta="right" w={90}>Confidence</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rules.map((rule, i) => (
            <Table.Tr key={i}>
              <Table.Td style={{ fontFamily: 'var(--mantine-font-family-monospace)' }}>
                {rule.conditions.length === 0
                  ? <Text size="xs" c="dimmed" span>—</Text>
                  : rule.conditions.map((c, j) => (
                    <span key={j}>
                      {j > 0 && <Text size="xs" c="dimmed" span> AND </Text>}
                      <Text size="xs" fw={500} span>{c.attribute}</Text>
                      <Text size="xs" c="dimmed" span> = </Text>
                      <Text size="xs" span>{c.value}</Text>
                    </span>
                  ))
                }
              </Table.Td>
              <Table.Td>{rule.conclusion}</Table.Td>
              <Table.Td ta="right">{rule.support}</Table.Td>
              <Table.Td ta="right">{(rule.confidence * 100).toFixed(1)}%</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Box>
  )
}
