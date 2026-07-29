import React from 'react'

interface QuantumCircuitProps {
  qubits?: number;
  circuitName?: string;
  type?: 'qsvm' | 'vqc';
}

const QuantumCircuit: React.FC<QuantumCircuitProps> = ({ qubits = 4, circuitName = "ZZFeatureMap (QSVM)", type = 'qsvm' }) => {
  return (
    <div className="w-full bg-[#070b19] border border-dark-border rounded-xl p-6 relative overflow-hidden">
      {/* Background neon ambient light */}
      <div className="absolute top-0 right-0 w-48 h-48 bg-quantum-cyan/5 rounded-full filter blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-48 h-48 bg-quantum-violet/5 rounded-full filter blur-3xl pointer-events-none" />

      <div className="flex items-center justify-between mb-4 border-b border-dark-border pb-3">
        <div>
          <h4 className="text-sm font-semibold text-slate-200">{circuitName}</h4>
          <p className="text-[10px] text-dark-muted">Dynamic Circuit visualization showing quantum operations on 4 registers</p>
        </div>
        <span className="text-[10px] px-2 py-0.5 rounded bg-quantum-cyan/10 text-quantum-cyan font-bold border border-quantum-cyan/20 uppercase tracking-widest">
          {type === 'qsvm' ? 'Feature Map' : 'Ansatz Layer'}
        </span>
      </div>

      <div className="overflow-x-auto py-4">
        <svg width="600" height="200" className="mx-auto font-mono text-[10px] select-none text-slate-300">
          {/* Qubit line labels */}
          <g>
            <text x="20" y="35" fill="#94a3b8">q[0]</text>
            <text x="20" y="75" fill="#94a3b8">q[1]</text>
            <text x="20" y="115" fill="#94a3b8">q[2]</text>
            <text x="20" y="155" fill="#94a3b8">q[3]</text>
          </g>

          {/* Horizontal qubit lines */}
          <g stroke="#334155" strokeWidth="2">
            <line x1="50" y1="32" x2="580" y2="32" />
            <line x1="50" y1="72" x2="580" y2="72" />
            <line x1="50" y1="112" x2="580" y2="112" />
            <line x1="50" y1="152" x2="580" y2="152" />
          </g>

          {type === 'qsvm' ? (
            /* QSVM ZZFeatureMap circuit rendering */
            <g>
              {/* Hadamard layer */}
              <g transform="translate(60, 0)">
                <rect x="0" y="17" width="30" height="30" rx="4" fill="#3b82f6" stroke="#60a5fa" strokeWidth="1" />
                <text x="15" y="35" textAnchor="middle" fill="white" className="font-bold">H</text>
                
                <rect x="0" y="57" width="30" height="30" rx="4" fill="#3b82f6" stroke="#60a5fa" strokeWidth="1" />
                <text x="15" y="75" textAnchor="middle" fill="white" className="font-bold">H</text>
                
                <rect x="0" y="97" width="30" height="30" rx="4" fill="#3b82f6" stroke="#60a5fa" strokeWidth="1" />
                <text x="15" y="115" textAnchor="middle" fill="white" className="font-bold">H</text>
                
                <rect x="0" y="137" width="30" height="30" rx="4" fill="#3b82f6" stroke="#60a5fa" strokeWidth="1" />
                <text x="15" y="155" textAnchor="middle" fill="white" className="font-bold">H</text>
              </g>

              {/* Rz rotation mapping */}
              <g transform="translate(115, 0)">
                <rect x="0" y="17" width="45" height="30" rx="4" fill="#00f2fe" stroke="#22d3ee" strokeWidth="1" />
                <text x="22.5" y="35" textAnchor="middle" fill="#040814" className="font-bold text-[9px]">U_p(x₀)</text>
                
                <rect x="0" y="57" width="45" height="30" rx="4" fill="#00f2fe" stroke="#22d3ee" strokeWidth="1" />
                <text x="22.5" y="75" textAnchor="middle" fill="#040814" className="font-bold text-[9px]">U_p(x₁)</text>
                
                <rect x="0" y="97" width="45" height="30" rx="4" fill="#00f2fe" stroke="#22d3ee" strokeWidth="1" />
                <text x="22.5" y="115" textAnchor="middle" fill="#040814" className="font-bold text-[9px]">U_p(x₂)</text>
                
                <rect x="0" y="137" width="45" height="30" rx="4" fill="#00f2fe" stroke="#22d3ee" strokeWidth="1" />
                <text x="22.5" y="155" textAnchor="middle" fill="#040814" className="font-bold text-[9px]">U_p(x₃)</text>
              </g>

              {/* Entangler CNOTs */}
              <g stroke="#8b5cf6" strokeWidth="1.5">
                {/* CNOT q0 -> q1 */}
                <line x1="190" y1="32" x2="190" y2="72" />
                <circle cx="190" cy="32" r="3" fill="#8b5cf6" />
                <circle cx="190" cy="72" r="6" fill="transparent" stroke="#8b5cf6" strokeWidth="1.5" />
                <line x1="187" y1="72" x2="193" y2="72" />
                <line x1="190" y1="69" x2="190" y2="75" />

                {/* CNOT q1 -> q2 */}
                <line x1="220" y1="72" x2="220" y2="112" />
                <circle cx="220" cy="72" r="3" fill="#8b5cf6" />
                <circle cx="220" cy="112" r="6" fill="transparent" stroke="#8b5cf6" strokeWidth="1.5" />
                <line x1="217" y1="112" x2="223" y2="112" />
                <line x1="220" y1="109" x2="220" y2="115" />

                {/* CNOT q2 -> q3 */}
                <line x1="250" y1="112" x2="250" y2="152" />
                <circle cx="250" cy="112" r="3" fill="#8b5cf6" />
                <circle cx="250" cy="152" r="6" fill="transparent" stroke="#8b5cf6" strokeWidth="1.5" />
                <line x1="247" y1="152" x2="253" y2="152" />
                <line x1="250" y1="149" x2="250" y2="155" />
              </g>

              {/* Double-excitation ZZ rotations */}
              <g transform="translate(280, 0)">
                <rect x="0" y="17" width="55" height="30" rx="4" fill="#a78bfa" stroke="#c084fc" strokeWidth="1" />
                <text x="27.5" y="35" textAnchor="middle" fill="#040814" className="font-bold text-[8px]">Rzz(x₀,x₁)</text>
                
                <rect x="0" y="57" width="55" height="30" rx="4" fill="#a78bfa" stroke="#c084fc" strokeWidth="1" />
                <text x="27.5" y="75" textAnchor="middle" fill="#040814" className="font-bold text-[8px]">Rzz(x₁,x₂)</text>
                
                <rect x="0" y="97" width="55" height="30" rx="4" fill="#a78bfa" stroke="#c084fc" strokeWidth="1" />
                <text x="27.5" y="115" textAnchor="middle" fill="#040814" className="font-bold text-[8px]">Rzz(x₂,x₃)</text>
              </g>

              {/* Measurement symbols */}
              <g transform="translate(520, 0)">
                {[32, 72, 112, 152].map((y, idx) => (
                  <g key={idx} transform={`translate(0, ${y - 15})`}>
                    <rect x="0" y="0" width="30" height="30" rx="4" fill="#475569" stroke="#64748b" strokeWidth="1" />
                    <path d="M 5 22 A 12 12 0 0 1 25 22" fill="none" stroke="white" strokeWidth="1.5" />
                    <line x1="15" y1="20" x2="23" y2="10" stroke="white" strokeWidth="1.5" />
                  </g>
                ))}
              </g>
            </g>
          ) : (
            /* VQC circuit rendering */
            <g>
              {/* Angle embedding rotations */}
              <g transform="translate(60, 0)">
                {[0, 1, 2, 3].map((idx) => (
                  <rect key={idx} x="0" y={17 + idx * 40} width="60" height="30" rx="4" fill="#10b981" stroke="#34d399" strokeWidth="1" />
                ))}
                <text x="30" y="35" textAnchor="middle" fill="white" className="font-bold">Rx(θ₀)</text>
                <text x="30" y="75" textAnchor="middle" fill="white" className="font-bold">Rx(θ₁)</text>
                <text x="30" y="115" textAnchor="middle" fill="white" className="font-bold">Rx(θ₂)</text>
                <text x="30" y="155" textAnchor="middle" fill="white" className="font-bold">Rx(θ₃)</text>
              </g>

              {/* StronglyEntangling variational layers */}
              <g transform="translate(150, 0)">
                <rect x="0" y="15" width="220" height="155" rx="8" fill="rgba(139, 92, 246, 0.15)" stroke="#8b5cf6" strokeWidth="1.5" strokeDasharray="4 2" />
                <text x="110" y="100" textAnchor="middle" fill="#c084fc" className="font-bold text-xs uppercase tracking-widest">
                  Variational Ansatz Layers
                </text>
                
                {/* Entanglers inside the ansatz */}
                {[0, 1, 2].map((idx) => (
                  <g key={idx} stroke="#a78bfa" strokeWidth="1">
                    <line x1={170 + idx * 80} y1="32" x2={170 + idx * 80} y2="152" />
                    <circle cx={170 + idx * 80} cy="32" r="3" fill="#a78bfa" />
                    <circle cx={170 + idx * 80} cy="72" r="3" fill="#a78bfa" />
                    <circle cx={170 + idx * 80} cy="112" r="3" fill="#a78bfa" />
                    <circle cx={170 + idx * 80} cy="152" r="3" fill="#a78bfa" />
                  </g>
                ))}
              </g>

              {/* Measurement symbols */}
              <g transform="translate(520, 0)">
                {[32, 72, 112, 152].map((y, idx) => (
                  <g key={idx} transform={`translate(0, ${y - 15})`}>
                    <rect x="0" y="0" width="30" height="30" rx="4" fill="#475569" stroke="#64748b" strokeWidth="1" />
                    <path d="M 5 22 A 12 12 0 0 1 25 22" fill="none" stroke="white" strokeWidth="1.5" />
                    <line x1="15" y1="20" x2="23" y2="10" stroke="white" strokeWidth="1.5" />
                  </g>
                ))}
              </g>
            </g>
          )}
        </svg>
      </div>
    </div>
  )
}

export default QuantumCircuit
