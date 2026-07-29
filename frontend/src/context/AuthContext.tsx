import React, { createContext, useContext, useState, useEffect } from 'react'

export interface AuthContextType {
  token: string | null;
  username: string | null;
  login: (token: string, username: string) => void;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
};

// Decodes JWT payload to check if it's expired
export const isTokenExpired = (token: string | null): boolean => {
  if (!token) return true;
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return true;
    const payload = JSON.parse(atob(parts[1]));
    if (!payload.exp) return false;
    const currentTime = Date.now() / 1000;
    return payload.exp < currentTime;
  } catch (e) {
    return true;
  }
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(localStorage.getItem('qc_token'));
  const [username, setUsername] = useState<string | null>(localStorage.getItem('qc_username'));

  const login = (newToken: string, newUsername: string) => {
    localStorage.setItem('qc_token', newToken);
    localStorage.setItem('qc_username', newUsername);
    setToken(newToken);
    setUsername(newUsername);
  };

  const logout = () => {
    localStorage.removeItem('qc_token');
    localStorage.removeItem('qc_username');
    setToken(null);
    setUsername(null);
  };

  // Check token validity on mount and periodically
  useEffect(() => {
    const checkAuth = () => {
      const storedToken = localStorage.getItem('qc_token');
      if (storedToken && isTokenExpired(storedToken)) {
        logout();
      }
    };
    checkAuth();
    const interval = setInterval(checkAuth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  const isAuthenticated = !!token && !isTokenExpired(token);

  return (
    <AuthContext.Provider value={{ token, username, login, logout, isAuthenticated }}>
      {children}
    </AuthContext.Provider>
  );
};
