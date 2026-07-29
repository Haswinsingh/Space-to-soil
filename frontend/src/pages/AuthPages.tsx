import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { motion } from 'framer-motion'
import { Mail, Lock, User, Satellite, AlertCircle } from 'lucide-react'
import api from '../api/client'

export const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await api.post('/auth/login', { email, password });
      login(response.data.access_token, response.data.username);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || "Authentication failed. Check credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#040814] flex items-center justify-center p-6 relative font-sans">
      <div className="absolute top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-quantum-cyan/5 rounded-full filter blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] bg-quantum-violet/5 rounded-full filter blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md glass-panel p-8 rounded-2xl relative z-10"
      >
        <div className="flex flex-col items-center mb-8">
          <div className="bg-gradient-to-tr from-quantum-cyan to-quantum-blue p-2.5 rounded-xl text-[#040814] mb-3 shadow-quantum-glow">
            <Satellite size={26} />
          </div>
          <h2 className="text-2xl font-bold font-sans">Welcome Back</h2>
          <p className="text-xs text-dark-muted mt-1">Access the Quantum Remote Sensing platform</p>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-quantum-rose/10 border border-quantum-rose/25 text-quantum-rose text-xs flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Email Address</label>
            <div className="relative">
              <Mail size={16} className="absolute left-3 top-3.5 text-dark-muted" />
              <input
                type="email"
                required
                placeholder="operator@quantumcrop.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg pl-10 pr-4 py-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-quantum-cyan transition-colors"
              />
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-slate-300">Password</label>
              <Link to="/forgot-password" id="forgot-password-link" className="text-[10px] text-quantum-cyan hover:underline">
                Forgot password?
              </Link>
            </div>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-3.5 text-dark-muted" />
              <input
                type="password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg pl-10 pr-4 py-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-quantum-cyan transition-colors"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 mt-2 rounded-lg bg-gradient-to-r from-quantum-cyan to-quantum-blue text-[#040814] font-bold shadow-quantum-glow hover:opacity-95 transition-all text-sm flex justify-center items-center gap-2"
          >
            {loading ? (
              <span className="w-5 h-5 rounded-full border-2 border-[#040814] border-t-transparent animate-spin" />
            ) : (
              "Sign In to Operations"
            )}
          </button>
        </form>

        <p className="text-xs text-center text-dark-muted mt-8">
          Don't have an account?{" "}
          <Link to="/register" className="text-quantum-cyan hover:underline font-semibold">
            Create account
          </Link>
        </p>
      </motion.div>
    </div>
  )
}

export const RegisterPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await api.post('/auth/register', { username, email, password });
      login(response.data.access_token, response.data.username);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || "Registration failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#040814] flex items-center justify-center p-6 relative font-sans">
      <div className="absolute top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-quantum-cyan/5 rounded-full filter blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] bg-quantum-violet/5 rounded-full filter blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md glass-panel p-8 rounded-2xl relative z-10"
      >
        <div className="flex flex-col items-center mb-8">
          <div className="bg-gradient-to-tr from-quantum-cyan to-quantum-blue p-2.5 rounded-xl text-[#040814] mb-3 shadow-quantum-glow">
            <Satellite size={26} />
          </div>
          <h2 className="text-2xl font-bold font-sans">Operator Registry</h2>
          <p className="text-xs text-dark-muted mt-1">Configure your operator credentials</p>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-quantum-rose/10 border border-quantum-rose/25 text-quantum-rose text-xs flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Username</label>
            <div className="relative">
              <User size={16} className="absolute left-3 top-3.5 text-dark-muted" />
              <input
                type="text"
                required
                placeholder="Alex Mercer"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg pl-10 pr-4 py-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-quantum-cyan transition-colors"
              />
            </div>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Email Address</label>
            <div className="relative">
              <Mail size={16} className="absolute left-3 top-3.5 text-dark-muted" />
              <input
                type="email"
                required
                placeholder="operator@quantumcrop.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg pl-10 pr-4 py-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-quantum-cyan transition-colors"
              />
            </div>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Password</label>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-3.5 text-dark-muted" />
              <input
                type="password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg pl-10 pr-4 py-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-quantum-cyan transition-colors"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 mt-2 rounded-lg bg-gradient-to-r from-quantum-cyan to-quantum-blue text-[#040814] font-bold shadow-quantum-glow hover:opacity-95 transition-all text-sm flex justify-center items-center gap-2"
          >
            {loading ? (
              <span className="w-5 h-5 rounded-full border-2 border-[#040814] border-t-transparent animate-spin" />
            ) : (
              "Initialize Operator Account"
            )}
          </button>
        </form>

        <p className="text-xs text-center text-dark-muted mt-8">
          Already registered?{" "}
          <Link to="/login" className="text-quantum-cyan hover:underline font-semibold">
            Login
          </Link>
        </p>
      </motion.div>
    </div>
  )
}

export const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    setError(null);
    setLoading(true);

    try {
      const response = await api.post('/auth/forgot-password', { email });
      setMessage(response.data.message);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Forgot password process failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#040814] flex items-center justify-center p-6 relative font-sans">
      <div className="absolute top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-quantum-cyan/5 rounded-full filter blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] bg-quantum-violet/5 rounded-full filter blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md glass-panel p-8 rounded-2xl relative z-10"
      >
        <div className="flex flex-col items-center mb-8">
          <div className="bg-gradient-to-tr from-quantum-cyan to-quantum-blue p-2.5 rounded-xl text-[#040814] mb-3 shadow-quantum-glow">
            <Satellite size={26} />
          </div>
          <h2 className="text-2xl font-bold font-sans">Recover Access</h2>
          <p className="text-xs text-dark-muted mt-1">Get back into the cockpit</p>
        </div>

        {message && (
          <div className="mb-6 p-4 rounded-lg bg-quantum-emerald/10 border border-quantum-emerald/25 text-quantum-emerald text-xs">
            <span>{message}</span>
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-quantum-rose/10 border border-quantum-rose/25 text-quantum-rose text-xs flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Email Address</label>
            <div className="relative">
              <Mail size={16} className="absolute left-3 top-3.5 text-dark-muted" />
              <input
                type="email"
                required
                placeholder="operator@quantumcrop.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg pl-10 pr-4 py-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-quantum-cyan transition-colors"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 mt-2 rounded-lg bg-gradient-to-r from-quantum-cyan to-quantum-blue text-[#040814] font-bold shadow-quantum-glow hover:opacity-95 transition-all text-sm flex justify-center items-center gap-2"
          >
            {loading ? (
              <span className="w-5 h-5 rounded-full border-2 border-[#040814] border-t-transparent animate-spin" />
            ) : (
              "Request Recovery Code"
            )}
          </button>
        </form>

        <p className="text-xs text-center text-dark-muted mt-8">
          Back to{" "}
          <Link to="/login" className="text-quantum-cyan hover:underline font-semibold">
            Login
          </Link>
        </p>
      </motion.div>
    </div>
  )
}
