export default function RegisterPage() {
  return null
}

/*
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Container, Paper, Title, TextInput, PasswordInput, Button, Text, Anchor, Stack } from '@mantine/core'
import { api } from '../api/client'
import { useAuth } from '../auth'

export default function RegisterPage() {
  const navigate = useNavigate()
  const { setToken } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    if (password.length < 8) { setError('Password must be at least 8 characters'); return }
    setLoading(true)
    try {
      const { data } = await api.post('/auth/register', { email, password })
      setToken(data.access_token)
      navigate('/')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container size={380} mt={80}>
      <Title order={2} mb="md">Register</Title>
      <Paper withBorder p="lg">
        <form onSubmit={handleSubmit}>
          <Stack gap="sm">
            <TextInput
              label="Email" type="email" value={email}
              onChange={e => setEmail(e.currentTarget.value)}
              required autoFocus
            />
            <PasswordInput
              label="Password" description="Min 8 characters" value={password}
              onChange={e => setPassword(e.currentTarget.value)}
              required
            />
            {error && <Text c="red" size="sm">{error}</Text>}
            <Button type="submit" loading={loading} fullWidth size="sm" mt="xs">
              Create account
            </Button>
          </Stack>
        </form>
      </Paper>
      <Text size="sm" ta="center" mt="md" c="dimmed">
        Already have an account? <Anchor component={Link} to="/login" size="sm">Login</Anchor>
      </Text>
    </Container>
  )
}
*/
