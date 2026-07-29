import React, { useState, useEffect } from 'react'
import api from '../api/client'
import { useAuth } from '../context/AuthContext'
import { 
  FileText, 
  Download, 
  Trash2, 
  Calendar, 
  AlertCircle,
  FileCheck,
  TrendingUp,
  Cpu
} from 'lucide-react'

const Reports: React.FC = () => {
  const { token } = useAuth();
  const [loading, setLoading] = useState(true);
  const [uploads, setUploads] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await api.get('/admin/stats', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        setUploads(res.data.recent_uploads || []);
      } catch (err) {
        setError("Could not retrieve telemetry logs.");
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [token]);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-extrabold font-sans text-slate-100">Telemetry Reports</h2>
        <p className="text-dark-muted text-xs">Inspect historical telemetry logs and download compiled PDF summaries</p>
      </div>

      {loading ? (
        <div className="glass-panel p-12 rounded-xl flex justify-center items-center h-64">
          <div className="w-10 h-10 rounded-full border-4 border-quantum-cyan border-t-transparent animate-spin" />
        </div>
      ) : error ? (
        <div className="p-4 rounded-lg bg-quantum-rose/10 border border-quantum-rose/25 text-quantum-rose text-xs flex items-center gap-2">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      ) : uploads.length > 0 ? (
        <div className="space-y-4">
          <div className="glass-panel rounded-xl overflow-hidden border border-dark-border">
            <table className="w-full text-left border-collapse text-xs md:text-sm">
              <thead>
                <tr className="bg-slate-900/60 border-b border-dark-border text-dark-muted font-semibold uppercase tracking-wider">
                  <th className="p-4">File Name</th>
                  <th className="p-4">Dimensions</th>
                  <th className="p-4">Health Classification</th>
                  <th className="p-4">Yield Forecast</th>
                  <th className="p-4 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border/40">
                {uploads.map((upload) => (
                  <tr key={upload.file_id} className="hover:bg-slate-900/25 transition-colors">
                    <td className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-quantum-blue/10 text-quantum-blue rounded">
                          <FileText size={16} />
                        </div>
                        <div>
                          <p className="font-semibold text-slate-200 truncate max-w-[180px]">{upload.filename}</p>
                          <p className="text-[10px] text-dark-muted">{upload.source_type}</p>
                        </div>
                      </div>
                    </td>
                    <td className="p-4">
                      <p className="text-slate-300 font-medium">{upload.resolution}</p>
                      <p className="text-[10px] text-dark-muted">{upload.bands_count} Bands • {upload.file_size_kb} KB</p>
                    </td>
                    <td className="p-4">
                      {upload.predictions ? (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] bg-quantum-emerald/10 border border-quantum-emerald/20 text-quantum-emerald font-bold uppercase tracking-wider">
                          {upload.predictions.crop_health}
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] bg-slate-950 text-slate-400 font-bold uppercase">
                          Incomplete
                        </span>
                      )}
                    </td>
                    <td className="p-4">
                      {upload.predictions ? (
                        <div className="flex items-center gap-1.5 text-quantum-cyan">
                          <TrendingUp size={14} />
                          <span className="font-bold">{upload.predictions.yield_t_ha.toFixed(2)} T/ha</span>
                        </div>
                      ) : (
                        <span className="text-dark-muted">N/A</span>
                      )}
                    </td>
                    <td className="p-4 text-center">
                      {upload.predictions ? (
                        <a 
                          href={`/api/predictions/report/download/${upload.file_id}`} 
                          download
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-slate-900 hover:bg-slate-950 border border-dark-border text-slate-300 transition-colors"
                        >
                          <Download size={12} />
                          <span>PDF Report</span>
                        </a>
                      ) : (
                        <span className="text-xs text-dark-muted font-bold">Unprocessed</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="glass-panel p-12 rounded-xl flex flex-col items-center justify-center text-center text-dark-muted h-64 border-dashed">
          <FileText size={40} className="mb-3 text-dark-muted/40" />
          <h4 className="font-bold text-sm text-slate-300">No Telemetry Logs Found</h4>
          <p className="text-xs text-dark-muted max-w-xs mt-1.5">
            Logs of uploaded images and crop health reports will appear here once analysis runs.
          </p>
        </div>
      )}
    </div>
  )
}

export default Reports;
