import React, { useState } from 'react'
import { 
  Settings as SettingsIcon, 
  Cpu, 
  Key, 
  User, 
  Lock, 
  Globe, 
  Save, 
  CheckCircle2,
  FileCheck
} from 'lucide-react'

const Settings: React.FC = () => {
  const [success, setSuccess] = useState(false);
  const [ibmKey, setIbmKey] = useState('');
  const [simulator, setSimulator] = useState('qasm');
  const [language, setLanguage] = useState('en');

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSuccess(true);
    setTimeout(() => setSuccess(false), 3000);
  };

  return (
    <div className="space-y-8 max-w-3xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-extrabold font-sans text-slate-100">System Settings</h2>
        <p className="text-dark-muted text-xs">Configure QML simulators, credentials, and user preferences</p>
      </div>

      {success && (
        <div className="p-4 rounded-lg bg-quantum-emerald/10 border border-quantum-emerald/25 text-quantum-emerald text-xs flex items-center gap-2">
          <CheckCircle2 size={16} />
          <span>System configuration updated successfully.</span>
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Quantum Simulator preferences */}
        <div className="glass-panel p-6 rounded-xl space-y-4">
          <div className="flex items-center gap-2 border-b border-dark-border pb-3 text-quantum-cyan">
            <Cpu size={18} />
            <h3 className="font-bold text-slate-200 text-sm">Quantum Simulator Interface</h3>
          </div>

          <div className="space-y-4">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Default QML Engine Backend</label>
              <select 
                value={simulator}
                onChange={(e) => setSimulator(e.target.value)}
                className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg px-4 py-3 text-xs text-slate-300 focus:outline-none focus:border-quantum-cyan"
              >
                <option value="qasm">IBM Qasm Simulator (Local CPU)</option>
                <option value="default_qubit">PennyLane Default Qubit (High Performance)</option>
                <option value="ibm_cloud">IBM Quantum Cloud (Requires Token)</option>
              </select>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">IBM Quantum Token</label>
              <div className="relative">
                <Key size={16} className="absolute left-3 top-3.5 text-dark-muted" />
                <input
                  type="password"
                  placeholder="Paste your IBM Quantum cloud access token here"
                  value={ibmKey}
                  onChange={(e) => setIbmKey(e.target.value)}
                  className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg pl-10 pr-4 py-3 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-quantum-cyan"
                />
              </div>
              <p className="text-[10px] text-dark-muted">Enables running the QML models on real quantum processors (e.g. ibmq_lima, ibmq_quito).</p>
            </div>
          </div>
        </div>

        {/* Localization settings */}
        <div className="glass-panel p-6 rounded-xl space-y-4">
          <div className="flex items-center gap-2 border-b border-dark-border pb-3 text-quantum-emerald">
            <Globe size={18} />
            <h3 className="font-bold text-slate-200 text-sm">System Localization</h3>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Primary Language / Idioma</label>
            <select 
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full bg-[#0a0f1d] border border-dark-border rounded-lg px-4 py-3 text-xs text-slate-300 focus:outline-none focus:border-quantum-cyan"
            >
              <option value="en">English (US)</option>
              <option value="es">Español (ES)</option>
              <option value="fr">Français (FR)</option>
              <option value="de">Deutsch (DE)</option>
            </select>
          </div>
        </div>

        {/* Submit */}
        <button
          type="submit"
          className="w-full py-3 rounded-lg bg-gradient-to-r from-quantum-cyan to-quantum-blue text-[#040814] font-bold shadow-quantum-glow hover:opacity-95 transition-all text-xs flex justify-center items-center gap-2"
        >
          <Save size={14} />
          <span>Save Configurations</span>
        </button>
      </form>
    </div>
  )
}

export default Settings;
