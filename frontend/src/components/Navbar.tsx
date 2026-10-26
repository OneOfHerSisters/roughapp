import { NavLink as RouterNavLink, useLocation } from 'react-router-dom'
import { Box, Group, Text, Anchor, Button, Container } from '@mantine/core'
import { useAuth } from '../auth'

const links = [
  { to: '/', label: 'Datasets', end: true },
  { to: '/experiments', label: 'Remove inconsistency' },
  // { to: '/results', label: 'Build decision tree' },
]

export default function Navbar() {
  const { user, logout } = useAuth()
  const location = useLocation()

  return (
    <Box
      component="nav"
      style={{
        borderBottom: '1px solid #8c6b20',
        background: 'linear-gradient(180deg, #49A9AD, #3E9AA0)',
      }}
    >
      <Container size="lg" px={0}>
        <Group justify="space-between" align="center" py={8} wrap="nowrap">
          <Group gap="xl" align="center" wrap="nowrap" style={{ flex: 1 }}>
            <Text
              fw={800}
              size="sm"
              c="white"
              tt="lowercase"
              style={{
                letterSpacing: '0.06em',
                minWidth: 92,
                flexShrink: 0,
                textShadow: '0 1px 0 rgba(0,0,0,0.18)',
              }}
            >
              roughapp
            </Text>

            <Group gap={0} align="center">
              {links.map(l => {
                const isActive = l.end ? location.pathname === l.to : location.pathname.startsWith(l.to)
                return (
                  <Anchor
                    key={l.to}
                    component={RouterNavLink}
                    to={l.to}
                    end={l.end}
                    size="sm"
                    c={isActive ? '#F7E6A9' : '#3A2A08'}
                    underline="never"
                    fw={isActive ? 700 : 500}
                    style={{
                      background: isActive ? '#6D4C05' : '#C9AB61',
                      border: '1px solid #8c6b20',
                      borderRightWidth: l.to === links[links.length - 1].to ? '1px' : '0px',
                      borderRadius: 0,
                      padding: '7px 18px',
                      whiteSpace: 'nowrap',
                      boxShadow: isActive ? 'inset 0 1px 0 rgba(255,255,255,0.14)' : 'inset 0 1px 0 rgba(255,255,255,0.28)',
                    }}
                  >
                    {l.label}
                  </Anchor>
                )
              })}
            </Group>
          </Group>

          <Group gap="sm" align="center" wrap="nowrap" style={{ marginLeft: 'auto' }}>
            {user ? (
              <>
                <Anchor component={RouterNavLink} to="/my-data" size="sm" c="white" underline="hover">
                  {user.email}
                </Anchor>
                <Button variant="filled" color="coral" onClick={logout}>Logout</Button>
              </>
            ) : (
              <>
                <Anchor component={RouterNavLink} to="/my-data" size="sm" c="white" underline="hover">
                  My data
                </Anchor>
                <Anchor component={RouterNavLink} to="/login" size="sm" c="white">Login</Anchor>
                <Anchor component={RouterNavLink} to="/register" size="sm" c="white">Register</Anchor>
              </>
            )}
          </Group>
        </Group>
      </Container>
    </Box>
  )
}
