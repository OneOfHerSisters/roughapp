import { createTheme, type MantineColorsTuple } from '@mantine/core'

const teal: MantineColorsTuple = [
  '#e4f6f6',
  '#d0ebec',
  '#a1d9db',
  '#6dc6ca',
  '#4cb6ba',
  '#32a0a5',
  '#1a7d81',
  '#113738',
  '#0b3032',
  '#022f31',
]

const gold: MantineColorsTuple = [
  '#f8f1e1',
  '#f1e3bf',
  '#e7d191',
  '#dcbf63',
  '#d1ad49',
  '#be9238',
  '#9b731f',
  '#7c5607',
  '#5d481b',
  '#513702',
]

const coral: MantineColorsTuple = [
  '#fae6e5',
  '#f3d0cf',
  '#e8aaa8',
  '#dd8280',
  '#d46663',
  '#be3b38',
  '#9b2422',
  '#7c0a07',
  '#5d1d1b',
  '#510402',
]

// Deliberately restrained: system fonts, small radius, no shadows by default.
export const theme = createTheme({
  colors: { teal, gold, coral },
  primaryColor: 'teal',
  primaryShade: 5,
  defaultRadius: 'sm',
  fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  headings: {
    fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
    sizes: {
      h1: { fontSize: '1.6rem' },
      h2: { fontSize: '1.3rem' },
      h3: { fontSize: '1.05rem' },
    },
  },
  components: {
    Button: {
      defaultProps: { size: 'xs', radius: 'md' },
    },
    Table: {
      defaultProps: { verticalSpacing: 4, horizontalSpacing: 'sm' },
    },
    Paper: {
      defaultProps: { radius: 'md' },
    },
  },
})
