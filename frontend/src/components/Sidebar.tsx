import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { 
  LayoutDashboard, 
  BarChart3, 
  FileText, 
  Settings as SettingsIcon, 
  LogOut, 
  Cpu, 
  Satellite,
  Upload as UploadIcon,
  BrainCircuit
} from 'lucide-react'

const Sidebar: React.FC = () => {
  const { logout, username } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { name: 'Dashboard Overview', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Upload Telemetry', path: '/dashboard/upload', icon: UploadIcon },
    { name: 'Inference Metrics', path: '/dashboard/predictions', icon: BrainCircuit },
    { name: 'ML Benchmarking', path: '/dashboard/ml-benchmark', icon: BarChart3 },
    { name: 'PDF Reports', path: '/dashboard/reports', icon: FileText },
    { name: 'Settings', path: '/dashboard/settings', icon: SettingsIcon },
  ];

  return (
    <div className="w-64 h-full glass-panel flex flex-col justify-between border-r border-dark-border z-30">
      <div>
        {/* Logo/Header */}
        <div className="p-6 border-b border-dark-border flex items-center gap-3">
          <div className="bg-gradient-to-tr from-quantum-cyan to-quantum-blue p-2 rounded-lg text-[#040814] shadow-quantum-glow animate-pulse">
            <Satellite size={22} />
          </div>
          <div>
            <h1 className="font-bold font-sans text-transparent bg-clip-text bg-gradient-to-r from-quantum-cyan to-quantum-blue tracking-wider text-base">
              QUANTUMCROP
            </h1>
            <p className="text-[10px] text-quantum-emerald font-semibold uppercase tracking-widest -mt-1">
              Eyes in the Sky
            </p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/dashboard'}
              className={({ isActive }) => `
                flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-300
                ${isActive 
                  ? 'bg-gradient-to-r from-quantum-cyan/15 to-quantum-blue/5 text-quantum-cyan border-l-2 border-quantum-cyan shadow-[inset_1px_0_0_rgba(0,242,254,0.15)]' 
                  : 'text-dark-muted hover:text-white hover:bg-white/5'
                }
              `}
            >
              <item.icon size={18} />
              <span>{item.name}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      {/* User profile & logout */}
      <div className="p-4 border-t border-dark-border">
        <div className="flex items-center gap-3 mb-4 px-3 py-2 bg-slate-900/30 rounded-lg border border-dark-border">
          <div className="w-8 h-8 rounded-full bg-quantum-violet flex items-center justify-center text-white font-bold text-sm shadow-[0_0_10px_rgba(139,92,246,0.3)]">
            {username?.charAt(0).toUpperCase()}
          </div>
          <div className="overflow-hidden">
            <p className="text-xs font-semibold text-slate-200 truncate">{username}</p>
            <div className="flex items-center gap-1">
              <Cpu size={10} className="text-quantum-emerald" />
              <p className="text-[9px] text-quantum-emerald uppercase font-bold tracking-wider">
                Quantum Operator
              </p>
            </div>
          </div>
        </div>
        
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium text-quantum-rose hover:bg-quantum-rose/10 transition-colors border border-transparent hover:border-quantum-rose/25"
        >
          <LogOut size={16} />
          <span>Sign Out</span>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
