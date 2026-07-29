import React from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  BrainCircuit, 
  TrendingUp, 
  Zap, 
  ShieldAlert, 
  Sprout, 
  Download 
} from 'lucide-react'

const Predictions: React.FC = () => {
  const { uploadData, predictions } = useOutletContext<any>();

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <div>
        <h2 className="text-3xl font-extrabold font-sans text-slate-100">Quantum Inference Results</h2>
        <p className="text-dark-muted text-xs">Analyze crop health classification, estimated yield, and AI recommendations</p>
      </div>

      {predictions ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="grid lg:grid-cols-2 gap-8"
        >
          <div className="glass-panel p-6 rounded-xl space-y-6">
            <div className="border-b border-dark-border pb-3 flex items-center justify-between">
              <div>
                <h3 className="font-bold text-slate-200 text-sm">Classification Metrics</h3>
                <p className="text-[10px] text-dark-muted font-medium">Derived from QML + Classical models</p>
              </div>
              <BrainCircuit className="text-quantum-cyan" size={20} />
            </div>

            {/* Crop Health Badge */}
            <div className="p-4 rounded-xl bg-slate-900/50 border border-dark-border text-center space-y-1 relative overflow-hidden">
              <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-quantum-cyan to-quantum-emerald" />
              <p className="text-[10px] text-dark-muted uppercase font-bold tracking-widest">Crop Health Status</p>
              <h4 className="text-2xl font-extrabold font-sans text-quantum-emerald">
                {predictions?.prediction || predictions?.crop_health || "Unknown"}
              </h4>
              <p className="text-xs text-slate-400">
                Confidence: <span className="font-bold text-quantum-cyan">{((predictions?.confidence || 0) * 100).toFixed(1)}%</span>
              </p>
            </div>

            {/* Crop Yield Prediction */}
            <div className="p-4 rounded-xl bg-slate-900/50 border border-dark-border flex items-center gap-4">
              <div className="p-3 bg-quantum-emerald/10 text-quantum-emerald rounded-lg">
                <TrendingUp size={22} />
              </div>
              <div>
                <p className="text-[10px] text-dark-muted uppercase font-bold tracking-wider">Estimated Crop Yield</p>
                <p className="text-lg font-bold text-slate-100">{(predictions?.yield_t_ha || 0).toFixed(2)} Tons / Hectare</p>
                <p className="text-[9px] text-dark-muted">Accuracy margin error: ±0.34 Tons</p>
              </div>
            </div>

            {/* PDF Report Export button */}
            <div className="pt-4">
              <a 
                href={`/api/predictions/report/download/${uploadData?.file_id}`} 
                download 
                className="w-full py-2.5 rounded-lg bg-slate-900 hover:bg-slate-950 border border-dark-border text-slate-200 text-xs font-bold flex justify-center items-center gap-2 transition-colors"
              >
                <Download size={14} />
                <span>Download PDF Report</span>
              </a>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-xl space-y-6">
            {/* AI Recommendation Engine */}
            <div className="space-y-3">
              <div className="flex items-center gap-1.5 border-b border-dark-border pb-3">
                <Zap size={18} className="text-quantum-cyan" />
                <h4 className="text-sm font-bold text-slate-200 uppercase">AI Recommendation Engine</h4>
              </div>
              
              <div className="space-y-3 pt-2">
                {(predictions?.recommendations || []).map((rec: any, index: number) => {
                  let severity = "low";
                  let title = "Recommendation";
                  let message = "";
                  
                  if (typeof rec === 'object' && rec !== null) {
                    severity = rec.severity || "low";
                    title = rec.title || "Recommendation";
                    message = rec.message || "";
                  } else {
                    message = String(rec);
                    severity = !message.startsWith("Optimal") ? "high" : "low";
                  }

                  const isHigh = severity === "high";
                  const isMedium = severity === "medium";

                  return (
                    <div 
                      key={index}
                      className={`p-3 rounded-lg border text-xs leading-relaxed flex gap-2.5
                        ${isHigh 
                          ? 'bg-quantum-rose/5 border-quantum-rose/15 text-slate-300' 
                          : isMedium
                            ? 'bg-amber-500/5 border-amber-500/15 text-slate-300'
                            : 'bg-quantum-emerald/5 border-quantum-emerald/15 text-slate-300'
                        }
                      `}
                    >
                      {isHigh ? (
                        <ShieldAlert className="text-quantum-rose flex-shrink-0 mt-0.5" size={16} />
                      ) : isMedium ? (
                        <ShieldAlert className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
                      ) : (
                        <Sprout className="text-quantum-emerald flex-shrink-0 mt-0.5" size={16} />
                      )}
                      <div>
                        <p className="font-bold text-slate-200 mb-0.5">{title}</p>
                        <span>{message}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </motion.div>
      ) : (
        <div className="glass-panel p-8 rounded-xl flex flex-col items-center justify-center text-center text-dark-muted h-[400px]">
          <BrainCircuit size={40} className="mb-3 text-dark-muted/40 animate-pulse" />
          <h4 className="font-bold text-sm text-slate-300">Run Quantum Inference</h4>
          <p className="text-xs text-dark-muted max-w-xs mt-1.5">
            Upload a multispectral raster image first to execute classifications and yield calculations.
          </p>
          <Link to="/dashboard/upload" className="mt-4 px-4 py-2 bg-gradient-to-r from-quantum-cyan to-quantum-blue text-[#040814] font-bold rounded-lg hover:opacity-90 transition-all text-xs">
            Go to Upload
          </Link>
        </div>
      )}
    </div>
  );
};

export default Predictions;
