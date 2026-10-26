import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'
import axios from 'axios'
import { api } from './api/client'

interface AuthState {
  token: string | null
  user: { id: number; email: string } | null
  loading: boolean
  setToken: (token: string | null) => void
  logout: () => void
}

const AuthContext = createContext<AuthState>({
  token: null, user: null, loading: false,
  setToken: () => {}, logout: () => {},
})

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(() => localStorage.getItem('token'))
  const [user, setUser] = useState<{ id: number; email: string } | null>(null)
  const [loading, setLoading] = useState<boolean>(() => !!localStorage.getItem('token'))

  function setToken(t: string | null) {
    setTokenState(t)
    if (t) {
      localStorage.setItem('token', t)
      api.defaults.headers.common['Authorization'] = `Bearer ${t}`
    } else {
      localStorage.removeItem('token')
      delete api.defaults.headers.common['Authorization']
    }
  }

  function logout() {
    setToken(null)
    setUser(null)
  }

  useEffect(() => {
    if (!token) {
      setUser(null)
      setLoading(false)
      return
    }

    let cancelled = false
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`

    api.get('/auth/me')
      .then(r => {
        if (!cancelled) {
          setUser(r.data)
          setLoading(false)
        }
      })
      .catch(err => {
        if (cancelled) return
        // Only clear session on explicit auth failure — not on timeouts or server busy.
        if (axios.isAxiosError(err) && err.response?.status === 401) {
          logout()
        }
        setLoading(false)
      })

    return () => { cancelled = true }
  }, [token])

  return (
    <AuthContext.Provider value={{ token, user, loading, setToken, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
