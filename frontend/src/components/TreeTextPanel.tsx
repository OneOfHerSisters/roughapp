import { Box, Text } from '@mantine/core'

export default function TreeTextPanel({ text, label }: { text: string; label?: string }) {
  return (
    <Box mt="xs">
      {label && (
        <Text size="xs" c="dimmed" mb={4}>
          {label} — plain-text export from scikit-learn (not JSON)
        </Text>
      )}
      <Box
        component="pre"
        p="sm"
        fz="xs"
        style={{
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          maxHeight: 280,
          overflow: 'auto',
          background: 'var(--mantine-color-gray-0)',
          border: '1px solid var(--mantine-color-gray-3)',
          borderRadius: 'var(--mantine-radius-sm)',
          fontFamily: 'var(--mantine-font-family-monospace)',
        }}
      >
        {text}
      </Box>
    </Box>
  )
}
