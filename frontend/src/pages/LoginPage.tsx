export default function LoginPage() {
  return null
}

/*
import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { Container, Paper, Title, TextInput, PasswordInput, Button, Text, Anchor, Stack } from '@mantine/core'
import { api } from '../api/client'
import { useAuth } from '../auth'

export default function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const from = (location.state as { from?: string } | null)?.from ?? '/experiments'
  const { setToken } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const { data } = await api.post('/auth/login', { email, password })
      setToken(data.access_token)
      navigate(from, { replace: true })
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container size={380} mt={80}>
      <Title order={2} mb="md">Login</Title>
      <Paper withBorder p="lg">
        <form onSubmit={handleSubmit}>
          <Stack gap="sm">
            <TextInput
              label="Email" type="email" value={email}
              onChange={e => setEmail(e.currentTarget.value)}
              required autoFocus
            />
            <PasswordInput
              label="Password" value={password}
              onChange={e => setPassword(e.currentTarget.value)}
              required
            />
            {error && <Text c="red" size="sm">{error}</Text>}
            <Button type="submit" loading={loading} fullWidth size="sm" mt="xs">
              Login
            </Button>
          </Stack>
        </form>
      </Paper>
      <Text size="sm" ta="center" mt="md" c="dimmed">
        No account? <Anchor component={Link} to="/register" size="sm">Register</Anchor>
      </Text>
    </Container>
  )
}
*/
