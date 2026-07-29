import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Satellite, Cpu, Sprout, ShieldAlert, LineChart, Globe } from 'lucide-react'

const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#040814] text-slate-100 overflow-x-hidden relative font-sans">
      {/* Decorative Blur Spheres */}
      <div className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] bg-quantum-cyan/5 rounded-full filter blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50vw] h-[50vw] bg-quantum-violet/5 rounded-full filter blur-[120px] pointer-events-none" />

      {/* Navigation */}
      <header className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between relative z-10">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-tr from-quantum-cyan to-quantum-blue p-2 rounded-lg text-[#040814] shadow-quantum-glow">
            <Satellite size={24} />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-quantum-cyan to-quantum-blue">
              QUANTUMCROP
            </h1>
            <p className="text-[10px] text-quantum-emerald font-semibold uppercase tracking-widest -mt-1">
              Eyes in the Sky
            </p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <Link to="/login" className="text-sm font-semibold text-slate-300 hover:text-white transition-colors">
            Login
          </Link>
          <Link to="/register" className="text-sm font-semibold px-4 py-2 rounded-lg bg-gradient-to-r from-quantum-cyan to-quantum-blue text-[#040814] shadow-quantum-glow hover:opacity-90 transition-all">
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 pt-16 pb-24 relative z-10 flex flex-col lg:flex-row items-center justify-between gap-12">
        <div className="flex-1 space-y-6 text-left">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-quantum-cyan/10 border border-quantum-cyan/25 text-quantum-cyan text-xs font-semibold uppercase tracking-wider"
          >
            <Cpu size={12} className="animate-spin" />
            Next-Gen Agricultural Analytics
          </motion.div>
          
          <motion.h2
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-6xl font-bold font-sans tracking-tight leading-none"
          >
            Quantum Crop Health & <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-quantum-cyan via-quantum-emerald to-quantum-blue">
              Yield Prediction
            </span>
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-slate-400 text-base max-w-xl leading-relaxed"
          >
            Quantum Support Vector Machines (QSVM) and Variational Classifiers applied to multi-spectral Sentinel-2 satellite data. Analyze crop health and forecast yield with sub-hectare precision.
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-wrap gap-4"
          >
            <Link to="/register" className="px-6 py-3 rounded-lg bg-gradient-to-r from-quantum-cyan to-quantum-blue text-[#040814] font-bold shadow-quantum-glow hover:translate-y-[-2px] transition-transform duration-300">
              Start Analysis
            </Link>
            <a href="#features" className="px-6 py-3 rounded-lg bg-slate-900/60 border border-dark-border text-slate-300 font-semibold hover:bg-slate-900/80 transition-colors">
              Learn More
            </a>
          </motion.div>
        </div>

        {/* Visual Graphic Panel */}
        <div className="flex-1 flex justify-center relative w-full max-w-md lg:max-w-none">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            className="w-[320px] h-[320px] md:w-[400px] md:h-[400px] rounded-full border border-quantum-cyan/20 flex items-center justify-center relative animate-pulse-slow"
          >
            {/* Inner Ring */}
            <div className="w-[240px] h-[240px] md:w-[300px] md:h-[300px] rounded-full border border-quantum-blue/15 flex items-center justify-center relative">
              <div className="w-20 h-20 bg-gradient-to-tr from-quantum-cyan to-quantum-emerald rounded-full flex items-center justify-center shadow-quantum-glow-green">
                <Sprout size={36} className="text-[#040814]" />
              </div>
              
              {/* Satellite Node */}
              <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-slate-950 p-2.5 rounded-full border border-quantum-cyan shadow-quantum-glow">
                <Satellite size={18} className="text-quantum-cyan" />
              </div>
              
              {/* Quantum Node */}
              <div className="absolute bottom-4 right-4 bg-slate-950 p-2.5 rounded-full border border-quantum-violet shadow-[0_0_15px_rgba(139,92,246,0.5)]">
                <Cpu size={18} className="text-quantum-violet" />
              </div>
            </div>
            
            {/* Orbiting particles */}
            <div className="absolute inset-0 border border-transparent border-t-quantum-cyan rounded-full animate-spin [animation-duration:15s]" />
            <div className="absolute inset-4 border border-transparent border-b-quantum-violet rounded-full animate-spin [animation-duration:8s] [animation-direction:reverse]" />
          </motion.div>
        </div>
      </section>

      {/* Features Grid */}
      <section id="features" className="max-w-7xl mx-auto px-6 py-24 border-t border-dark-border relative z-10">
        <div className="text-center max-w-xl mx-auto mb-16 space-y-3">
          <span className="text-quantum-cyan font-semibold text-xs uppercase tracking-widest">
            Platform Capabilities
          </span>
          <h3 className="text-3xl font-bold font-sans">
            Modular Remote Sensing Infrastructure
          </h3>
          <p className="text-dark-muted text-sm leading-relaxed">
            Fully compatible with Sentinel-2, EuroSAT, and multispectral drone TIFF uploads.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Card 1 */}
          <div className="glass-panel p-6 rounded-xl space-y-4 hover:border-quantum-cyan/30 transition-colors group">
            <div className="w-10 h-10 rounded-lg bg-quantum-cyan/10 flex items-center justify-center text-quantum-cyan group-hover:scale-110 transition-transform">
              <Globe size={20} />
            </div>
            <h4 className="font-bold text-slate-100 text-lg">Vegetation Index Pipeline</h4>
            <p className="text-dark-muted text-sm leading-relaxed">
              Auto-calculate NDVI, SAVI, EVI, NDWI, and Chlorophyll Index bands. Renders high-contrast spatial stress maps.
            </p>
          </div>

          {/* Card 2 */}
          <div className="glass-panel p-6 rounded-xl space-y-4 hover:border-quantum-emerald/30 transition-colors group">
            <div className="w-10 h-10 rounded-lg bg-quantum-emerald/10 flex items-center justify-center text-quantum-emerald group-hover:scale-110 transition-transform">
              <Sprout size={20} />
            </div>
            <h4 className="font-bold text-slate-100 text-lg">Yield Estimator</h4>
            <p className="text-dark-muted text-sm leading-relaxed">
              Predict crop tonnage per hectare dynamically utilizing multispectral reflection indexes.
            </p>
          </div>

          {/* Card 3 */}
          <div className="glass-panel p-6 rounded-xl space-y-4 hover:border-quantum-violet/30 transition-colors group">
            <div className="w-10 h-10 rounded-lg bg-quantum-violet/10 flex items-center justify-center text-quantum-violet group-hover:scale-110 transition-transform">
              <Cpu size={20} />
            </div>
            <h4 className="font-bold text-slate-100 text-lg">Quantum SVM & VQC</h4>
            <p className="text-dark-muted text-sm leading-relaxed">
              Evaluate quantum advantage on small sample datasets using ZZFeatureMaps, angle embeddings, and quantum kernels.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-dark-border py-12 max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center text-dark-muted text-xs gap-4 relative z-10">
        <p>© 2026 QuantumCrop AI - Hackathon Production Release.</p>
        <p>Built for Remote Sensing & Quantum Computing agricultural applications.</p>
      </footer>
    </div>
  )
}

export default LandingPage;
