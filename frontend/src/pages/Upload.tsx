import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import { useNavigate, useOutletContext } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import api from '../api/client'
import { 
  Upload as UploadIcon, 
  AlertCircle,
  TrendingUp
} from 'lucide-react'

const Upload: React.FC = () => {
  const { token } = useAuth();
  const navigate = useNavigate();
  const { uploadData, setUploadData, setPredictions } = useOutletContext<any>();
  
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState<'ndvi' | 'evi' | 'savi' | 'ndwi' | 'ci'>('ndvi');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append('file', file);
    
    setError(null);
    setLoading(true);
    setPredictions(null);
    setUploadData(null);
    
    const stages = [
      'Reading Multispectral Bands (RGB + NIR)...',
      'Executing Bilateral Noise Filtering...',
      'Scaling Contrast & Radiometric Calibration...',
      'Computing Pixel-wise Vegetation Indices...',
      'Extracting Spectral Statistics...'
    ];
    
    let currentStage = 0;
    setStage(stages[0]);
    const stageTimer = setInterval(() => {
      if (currentStage < stages.length - 1) {
        currentStage++;
        setStage(stages[currentStage]);
      }
    }, 900);

    try {
      // 1. Upload & Predict in a single request
      const predictRes = await api.post('/predictions/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        }
      });
      clearInterval(stageTimer);
      
      const resData = predictRes.data;
      if (resData && resData.success === false) {
        setError(resData.message || "Failed to analyze crop imagery.");
        return;
      }
      
      const payload = resData?.data || resData;
      setUploadData(payload);
      setPredictions(payload);
      
      // Navigate to predictions page
      navigate('/dashboard/predictions');
    } catch (err: any) {
      setError(err.response?.data?.message || err.response?.data?.detail || err.message || "Error processing agricultural imagery.");
      clearInterval(stageTimer);
    } finally {
      setLoading(false);
      setStage('');
    }
  }, [token, setUploadData, setPredictions, navigate]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.geotiff']
    },
    multiple: false
  });

  const features = uploadData?.features || uploadData?.indices;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <div>
        <h2 className="text-3xl font-extrabold font-sans text-slate-100">Upload Telemetry</h2>
        <p className="text-dark-muted text-xs">Upload multispectral drone or satellite images to run crop diagnostics</p>
      </div>

      <div className="glass-panel p-6 rounded-xl space-y-4">
        <div>
          <h3 className="font-bold text-slate-200 text-sm">Upload Drone / Satellite Image</h3>
          <p className="text-[10px] text-dark-muted font-medium">Supports JPEG, PNG, TIFF and multispectral GeoTIFF files</p>
        </div>
        
        {error && (
          <div className="p-4 rounded-lg bg-quantum-rose/10 border border-quantum-rose/25 text-quantum-rose text-xs flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <div {...getRootProps()} className={`
          border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all duration-300
          ${isDragActive ? 'border-quantum-cyan bg-quantum-cyan/5 shadow-quantum-glow' : 'border-dark-border hover:border-slate-600'}
        `}>
          <input {...getInputProps()} />
          <div className="p-4 bg-slate-900/50 rounded-full border border-dark-border mb-4 text-quantum-cyan hover:scale-110 transition-transform">
            <UploadIcon size={32} />
          </div>
          <p className="text-xs font-semibold text-slate-300">Drag & drop files here, or click to browse</p>
          <p className="text-[10px] text-dark-muted mt-1">Accepts maximum file size up to 10MB</p>
        </div>
      </div>

      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-panel p-8 rounded-xl flex flex-col items-center justify-center space-y-6"
        >
          <div className="w-12 h-12 rounded-full border-4 border-quantum-cyan border-t-transparent animate-spin shadow-quantum-glow" />
          <div className="text-center space-y-1">
            <h4 className="font-bold text-sm text-slate-200 uppercase tracking-widest animate-pulse">Running Preprocessing Pipeline</h4>
            <p className="text-xs text-quantum-emerald font-semibold">{stage}</p>
          </div>
        </motion.div>
      )}

      {uploadData && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-6 rounded-xl space-y-6"
        >
          <div className="flex justify-between items-center border-b border-dark-border pb-3">
            <div>
              <h3 className="font-bold text-slate-200 text-sm">Multispectral Band Analysis</h3>
              <p className="text-[10px] text-dark-muted">Renders color-mapped index configurations and band channels</p>
            </div>
            <div className="flex gap-2">
              {(['ndvi', 'evi', 'savi', 'ndwi', 'ci'] as const).map((idxName) => (
                <button
                  key={idxName}
                  onClick={() => setActiveIndex(idxName)}
                  className={`
                    text-[10px] px-2.5 py-1 rounded font-bold uppercase tracking-wider border transition-all duration-300
                    ${activeIndex === idxName 
                      ? 'bg-quantum-cyan/15 text-quantum-cyan border-quantum-cyan/40 shadow-[0_0_10px_rgba(0,242,254,0.15)]' 
                      : 'bg-transparent text-dark-muted border-dark-border hover:text-slate-200'
                    }
                  `}
                >
                  {idxName}
                </button>
              ))}
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6 items-center">
            <div className="bg-[#050917] rounded-lg overflow-hidden border border-dark-border aspect-square relative group">
              <img 
                src={uploadData?.paths?.processed?.[activeIndex] || ""} 
                alt={activeIndex} 
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-3 left-3 bg-[#040814]/80 backdrop-blur-sm border border-dark-border px-2.5 py-1 rounded text-[9px] font-bold text-quantum-cyan uppercase tracking-widest">
                {activeIndex} Map View
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-slate-200 uppercase">
                  {activeIndex === 'ndvi' && 'NDVI (Normalized Difference Veg Index)'}
                  {activeIndex === 'evi' && 'EVI (Enhanced Vegetation Index)'}
                  {activeIndex === 'savi' && 'SAVI (Soil Adjusted Vegetation Index)'}
                  {activeIndex === 'ndwi' && 'NDWI (Normalized Difference Water Index)'}
                  {activeIndex === 'ci' && 'CI (Chlorophyll Index)'}
                </h4>
                <p className="text-[10px] text-dark-muted leading-relaxed">
                  {activeIndex === 'ndvi' && 'Measures greenness density to identify canopy health and photosynthesis activity.'}
                  {activeIndex === 'evi' && 'Optimized index designed to minimize atmospheric and soil background noise in high biomass areas.'}
                  {activeIndex === 'savi' && 'Accounts for soil brightness variables; excellent for sparse vegetation regions.'}
                  {activeIndex === 'ndwi' && 'Detects leaf water content and moisture stress levels in crops.'}
                  {activeIndex === 'ci' && 'Evaluates leaf chlorophyll concentrations directly correlating with nitrogen intake.'}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3 bg-slate-900/30 p-3 rounded-lg border border-dark-border">
                <div>
                  <p className="text-[9px] text-dark-muted font-bold uppercase">Mean Value</p>
                  <p className="text-sm font-bold text-quantum-cyan">
                    {features?.[`${activeIndex}_mean` as keyof typeof features]?.toFixed(4) || "0.0000"}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-dark-muted font-bold uppercase">Std Deviation</p>
                  <p className="text-sm font-bold text-slate-300">
                    {features?.[`${activeIndex}_std` as keyof typeof features]?.toFixed(4) || "0.0000"}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-dark-muted font-bold uppercase">Minimum</p>
                  <p className="text-sm font-bold text-slate-300">
                    {features?.[`${activeIndex}_min` as keyof typeof features]?.toFixed(4) || "0.0000"}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-dark-muted font-bold uppercase">Maximum</p>
                  <p className="text-sm font-bold text-slate-300">
                    {features?.[`${activeIndex}_max` as keyof typeof features]?.toFixed(4) || "0.0000"}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4 pt-4 border-t border-dark-border">
            {['red', 'green', 'blue', 'nir'].map((band) => (
              <div key={band} className="text-center">
                <div className="bg-[#050917] rounded-lg overflow-hidden border border-dark-border mb-1 aspect-square">
                  <img src={uploadData?.paths?.processed?.[band] || ""} alt={band} className="w-full h-full object-cover" />
                </div>
                <p className="text-[9px] font-bold text-dark-muted uppercase">{band} Band</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Upload;
