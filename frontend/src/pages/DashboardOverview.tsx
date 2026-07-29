import React, { useState, useEffect, useRef } from 'react'
import { 
  Satellite, 
  Cpu, 
  Layers, 
  Download, 
  Loader2, 
  Upload as UploadIcon, 
  MapPin, 
  TrendingUp, 
  Activity, 
  Sprout, 
  ShieldAlert,
  HelpCircle
} from 'lucide-react'
import api from '../api/client'
import { useAuth } from '../context/AuthContext'

// Pre-defined field coordinates centered around agricultural zones in Tamil Nadu
const tamilNaduFields = [
  {
    id: 1,
    name: "Coimbatore Wheat Valley",
    center: { lat: 11.0168, lng: 76.9558 },
    coords: [
      [76.950, 11.015],
      [76.950, 11.025],
      [76.965, 11.025],
      [76.965, 11.015],
      [76.950, 11.015]
    ],
    status: "Query GEE",
    yield: "--",
    color: "#10b981"
  },
  {
    id: 2,
    name: "Thanjavur Rice cadastre",
    center: { lat: 10.7870, lng: 79.1378 },
    coords: [
      [79.130, 10.780],
      [79.130, 10.795],
      [79.145, 10.795],
      [79.145, 10.780],
      [79.130, 10.780]
    ],
    status: "Query GEE",
    yield: "--",
    color: "#3b82f6"
  },
  {
    id: 3,
    name: "Madurai Millet Orchard",
    center: { lat: 9.9252, lng: 78.1198 },
    coords: [
      [78.110, 9.915],
      [78.110, 9.935],
      [78.125, 9.935],
      [78.125, 9.915],
      [78.110, 9.915]
    ],
    status: "Query GEE",
    yield: "--",
    color: "#eab308"
  }
];

const DashboardOverview: React.FC = () => {
  const { token } = useAuth();
  
  // Maps & UI State
  const [selectedField, setSelectedField] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [geeData, setGeeData] = useState<any>(null);
  const [activeLayer, setActiveLayer] = useState<string>('satellite'); // satellite, hybrid, ndvi, evi, savi, ndwi, ci, true_color, false_color
  const [drawnPolygons, setDrawnPolygons] = useState<any[]>([]);

  // Refs for Google Map
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const drawingManagerRef = useRef<any>(null);
  const polygonsInstancesRef = useRef<any[]>([]);
  const geeOverlayInstanceRef = useRef<any>(null);

  // Initialize Google Maps API script
  useEffect(() => {
    const loadScript = () => {
      if (window.google && window.google.maps) {
        initMap();
        return;
      }
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${import.meta.env.VITE_GOOGLE_MAPS_API_KEY || ''}&libraries=drawing,geometry`;
      script.async = true;
      script.defer = true;
      script.onload = initMap;
      document.head.appendChild(script);
    };
    
    loadScript();

    return () => {
      // Clean up map overlays on unmount
      if (polygonsInstancesRef.current) {
        polygonsInstancesRef.current.forEach(p => p.setMap(null));
      }
    };
  }, []);

  // Initialize Map
  const initMap = () => {
    if (!mapRef.current) return;

    // Centered on Tamil Nadu
    const mapOptions = {
      center: { lat: 11.1271, lng: 78.6569 },
      zoom: 7.5,
      mapTypeId: 'satellite',
      styles: [
        { featureType: 'all', elementType: 'labels.text.fill', stylers: [{ color: '#ffffff' }] }
      ],
      streetViewControl: false,
      mapTypeControl: false,
      fullscreenControl: true
    };

    const map = new window.google.maps.Map(mapRef.current, mapOptions);
    mapInstanceRef.current = map;

    // Set up GEE pre-defined polygons
    renderPredefinedPolygons();

    // Set up Google Maps Drawing Tools
    const drawingManager = new window.google.maps.drawing.DrawingManager({
      drawingMode: null,
      drawingControl: true,
      drawingControlOptions: {
        position: window.google.maps.ControlPosition.TOP_CENTER,
        drawingModes: [window.google.maps.drawing.OverlayType.POLYGON],
      },
      polygonOptions: {
        fillColor: '#00ffd5',
        fillOpacity: 0.2,
        strokeWeight: 2,
        strokeColor: '#00ffd5',
        clickable: true,
        editable: true,
        zIndex: 1
      },
    });
    
    drawingManager.setMap(map);
    drawingManagerRef.current = drawingManager;

    // Listen for drawn polygons
    window.google.maps.event.addListener(drawingManager, 'polygoncomplete', (polygon: any) => {
      // Extract coordinates
      const path = polygon.getPath();
      const coords: [number, number][] = [];
      const pathCoords: any[] = [];
      
      for (let i = 0; i < path.getLength(); i++) {
        const xy = path.getAt(i);
        coords.push([xy.lng(), xy.lat()]);
        pathCoords.push({ lat: xy.lat(), lng: xy.lng() });
      }
      // Close polygon path
      coords.push(coords[0]);

      // Calculate center of drawn polygon
      const bounds = new window.google.maps.LatLngBounds();
      pathCoords.forEach(c => bounds.extend(c));
      const center = { lat: bounds.getCenter().lat(), lng: bounds.getCenter().lng() };

      const newField = {
        id: Date.now(),
        name: `Drawn Boundary #${drawnPolygons.length + 1}`,
        center: center,
        coords: coords,
        color: '#00ffd5'
      };

      setDrawnPolygons(prev => [...prev, newField]);
      setSelectedField(newField);
      triggerGeeAnalysis(newField);

      // Listen for click on drawn polygon to select it again
      window.google.maps.event.addListener(polygon, 'click', () => {
        setSelectedField(newField);
        triggerGeeAnalysis(newField);
      });

      // Track instance to clear later
      polygonsInstancesRef.current.push(polygon);
    });
  };

  // Render pre-defined fields
  const renderPredefinedPolygons = () => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Clear existing polygon overlays if any
    polygonsInstancesRef.current.forEach(p => p.setMap(null));
    polygonsInstancesRef.current = [];

    tamilNaduFields.forEach(field => {
      // Map coordinates format to Google Maps LatLng
      const paths = field.coords.map(c => ({ lat: c[1], lng: c[0] }));

      const polygon = new window.google.maps.Polygon({
        paths: paths,
        strokeColor: field.color,
        strokeOpacity: 0.8,
        strokeWeight: 2,
        fillColor: field.color,
        fillOpacity: 0.2,
        map: map
      });

      // Listen for click
      window.google.maps.event.addListener(polygon, 'click', () => {
        // Center map slightly with smooth pan
        map.panTo(field.center);
        map.setZoom(13.5);
        setSelectedField(field);
        triggerGeeAnalysis(field);
      });

      polygonsInstancesRef.current.push(polygon);
    });
  };

  // Call /api/gee/analyze endpoint
  const triggerGeeAnalysis = async (field: any) => {
    setLoading(true);
    setError(null);
    setGeeData(null);
    // Reset GEE tile layer overlay
    if (mapInstanceRef.current && geeOverlayInstanceRef.current) {
      mapInstanceRef.current.overlayMapTypes.clear();
      geeOverlayInstanceRef.current = null;
    }

    try {
      const response = await api.post('/gee/analyze', {
        latitude: field.center.lat,
        longitude: field.center.lng,
        polygon: field.coords,
        use_landsat: false
      }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.data && response.data.success) {
        setGeeData(response.data);
        // Automatically default layer to NDVI once loaded
        applyGeeTileOverlay(response.data.tile_urls.ndvi, 'ndvi');
      } else {
        setError(response.data?.message || "Earth Engine returned an unsuccessful response.");
      }
    } catch (err: any) {
      console.error(err);
      setError(
        err.response?.data?.detail || 
        err.response?.data?.message || 
        "Failed to query Earth Engine. Ensure Google account authentication status is active."
      );
    } finally {
      setLoading(false);
    }
  };

  // Apply Earth Engine Tile Overlay to Google Maps Map Instance
  const applyGeeTileOverlay = (tileUrlTemplate: string, layerKey: string) => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Clear existing GEE tile overlays
    map.overlayMapTypes.clear();
    geeOverlayInstanceRef.current = null;
    setActiveLayer(layerKey);

    if (layerKey === 'satellite') {
      map.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);
      return;
    } else if (layerKey === 'hybrid') {
      map.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
      return;
    }

    // Set map to satellite background for overlays
    map.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);

    if (!tileUrlTemplate) return;

    // Define custom Google Maps ImageMapType
    const geeTileOverlay = new window.google.maps.ImageMapType({
      getTileUrl: (coord: any, zoom: number) => {
        let url = tileUrlTemplate;
        url = url.replace('{z}', zoom.toString());
        url = url.replace('{x}', coord.x.toString());
        url = url.replace('{y}', coord.y.toString());
        return url;
      },
      tileSize: new window.google.maps.Size(256, 256),
      name: layerKey.toUpperCase(),
      opacity: 0.8
    });

    map.overlayMapTypes.push(geeTileOverlay);
    geeOverlayInstanceRef.current = geeTileOverlay;
  };

  // Handle Layer change via controls
  const handleLayerChange = (layerKey: string) => {
    if (!geeData) {
      if (layerKey === 'satellite' && mapInstanceRef.current) {
        mapInstanceRef.current.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);
        setActiveLayer('satellite');
      } else if (layerKey === 'hybrid' && mapInstanceRef.current) {
        mapInstanceRef.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
        setActiveLayer('hybrid');
      }
      return;
    }

    const tileUrlTemplate = geeData.tile_urls[layerKey];
    applyGeeTileOverlay(tileUrlTemplate, layerKey);
  };

  // Handle GeoJSON upload
  const handleGeoJsonUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const geojson = JSON.parse(event.target?.result as string);
        
        // Extract polygon geometry from feature collection or single feature
        let geom = null;
        if (geojson.type === "FeatureCollection" && geojson.features.length > 0) {
          geom = geojson.features[0].geometry;
        } else if (geojson.type === "Feature") {
          geom = geojson.geometry;
        } else if (geojson.type === "Polygon") {
          geom = geojson;
        }

        if (!geom || geom.type !== "Polygon") {
          alert("Invalid GeoJSON. Only single Polygons are supported.");
          return;
        }

        // GEE format: list of coords [[lng, lat]]
        const coords = geom.coordinates[0];
        
        // Map to Google Maps coords format
        const pathCoords = coords.map((c: any) => ({ lat: c[1], lng: c[0] }));
        
        // Fit Map bounds
        const bounds = new window.google.maps.LatLngBounds();
        pathCoords.forEach((c: any) => bounds.extend(c));
        const center = { lat: bounds.getCenter().lat(), lng: bounds.getCenter().lng() };

        const map = mapInstanceRef.current;
        if (map) {
          map.fitBounds(bounds);
        }

        const newField = {
          id: Date.now(),
          name: file.name.replace(".geojson", ""),
          center: center,
          coords: coords,
          color: '#eab308'
        };

        // Render polygon on map
        const polygon = new window.google.maps.Polygon({
          paths: pathCoords,
          strokeColor: newField.color,
          strokeOpacity: 0.8,
          strokeWeight: 2,
          fillColor: newField.color,
          fillOpacity: 0.2,
          map: map
        });

        polygonsInstancesRef.current.push(polygon);

        setSelectedField(newField);
        triggerGeeAnalysis(newField);

        window.google.maps.event.addListener(polygon, 'click', () => {
          setSelectedField(newField);
          triggerGeeAnalysis(newField);
        });

      } catch (err) {
        alert("Failed to parse GeoJSON: " + err);
      }
    };
    reader.readAsText(file);
  };

  // Trigger PDF Report Download
  const downloadPdfReport = async () => {
    if (!geeData || !geeData.file_id) return;
    try {
      window.open(`${api.defaults.baseURL}/predictions/report/download/${geeData.file_id}`, '_blank');
    } catch (err) {
      console.error(err);
      alert("Failed to trigger PDF generation.");
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-3xl font-extrabold font-sans text-slate-100">Google Earth Engine Control Center</h2>
          <p className="text-dark-muted text-xs">Analyze crop indices using Sentinel-2 and Landsat-8 remote sensing data</p>
        </div>
        
        {/* GeoJSON Upload */}
        <label className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-dark-border bg-slate-900/50 hover:bg-slate-800 text-xs font-semibold text-quantum-cyan cursor-pointer transition-all duration-300">
          <UploadIcon size={14} />
          <span>Upload Boundary (GeoJSON)</span>
          <input type="file" accept=".geojson,.json" className="hidden" onChange={handleGeoJsonUpload} />
        </label>
      </div>

      {/* Main Map Block */}
      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass-panel p-6 rounded-xl space-y-4 relative flex flex-col justify-between">
          <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3">
            <div>
              <h3 className="font-bold text-slate-200 text-sm">Interactive Satellite Field Cadastre</h3>
              <p className="text-[10px] text-dark-muted">Select Tamil Nadu zones, draw boundaries, or layer indices dynamically</p>
            </div>
            
            {/* Layer Controls */}
            <div className="flex flex-wrap gap-1 bg-slate-950/60 p-1 rounded-lg border border-dark-border">
              {(['satellite', 'hybrid', 'ndvi', 'evi', 'savi', 'ndwi', 'ci', 'true_color', 'false_color'] as const).map((layer) => (
                <button
                  key={layer}
                  onClick={() => handleLayerChange(layer)}
                  className={`
                    text-[9px] px-2 py-0.5 rounded font-bold uppercase tracking-wider transition-all duration-200
                    ${activeLayer === layer 
                      ? 'bg-quantum-cyan/20 text-quantum-cyan border border-quantum-cyan/30' 
                      : 'bg-transparent text-dark-muted hover:text-slate-200 border border-transparent'
                    }
                  `}
                >
                  {layer.replace('_', ' ')}
                </button>
              ))}
            </div>
          </div>
          
          {/* Google Map Ref Container */}
          <div className="h-[380px] w-full rounded-lg overflow-hidden relative border border-dark-border bg-[#050814]">
            <div ref={mapRef} style={{ height: '100%', width: '100%' }} />

            {/* loading state */}
            {loading && (
              <div className="absolute inset-0 bg-[#040814]/75 backdrop-blur-sm flex flex-col items-center justify-center space-y-4 z-50">
                <Loader2 size={36} className="text-quantum-cyan animate-spin shadow-quantum-glow" />
                <div className="text-center">
                  <p className="text-xs font-bold text-slate-200 uppercase tracking-widest animate-pulse">Running GEE Pipeline</p>
                  <p className="text-[10px] text-quantum-emerald font-semibold">Querying cloud-free bands & resolving indices...</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Selected Field Cadastre Telemetry Panel */}
        <div className="glass-panel p-6 rounded-xl flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-slate-200 text-sm mb-4 border-b border-dark-border pb-2 flex justify-between items-center">
              <span>Field Cadastre Telemetry</span>
              {geeData && (
                <span className="text-[9px] text-quantum-cyan border border-quantum-cyan/20 bg-quantum-cyan/10 px-2 py-0.5 rounded font-mono uppercase">
                  Live Satellite Data
                </span>
              )}
            </h3>

            {error && (
              <div className="p-3 mb-4 rounded bg-quantum-rose/10 border border-quantum-rose/25 text-quantum-rose text-[11px] leading-relaxed">
                <ShieldAlert size={14} className="inline mr-1" />
                <span>{error}</span>
              </div>
            )}
            
            {selectedField ? (
              <div className="space-y-4">
                <div className="p-3 bg-slate-900/50 rounded-lg border border-dark-border">
                  <p className="text-[9px] text-dark-muted font-bold uppercase tracking-wider">Field Name</p>
                  <p className="text-xs font-extrabold text-slate-200">{selectedField.name}</p>
                </div>

                {geeData ? (
                  <div className="space-y-4">
                    {/* Acquisition info */}
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="p-2 bg-slate-950/40 rounded border border-dark-border">
                        <p className="text-[8px] text-dark-muted font-bold">AREA (ha)</p>
                        <p className="text-[11px] font-extrabold text-slate-100">{geeData.area_ha.toFixed(2)}</p>
                      </div>
                      <div className="p-2 bg-slate-950/40 rounded border border-dark-border">
                        <p className="text-[8px] text-dark-muted font-bold">CLOUD %</p>
                        <p className="text-[11px] font-extrabold text-slate-100">{geeData.cloud_cover.toFixed(1)}%</p>
                      </div>
                      <div className="p-2 bg-slate-950/40 rounded border border-dark-border">
                        <p className="text-[8px] text-dark-muted font-bold">DATE</p>
                        <p className="text-[11px] font-extrabold text-slate-100">{geeData.acquisition_date}</p>
                      </div>
                    </div>

                    {/* Vegetation Indices */}
                    <div className="space-y-2 border-t border-dark-border/60 pt-3">
                      <p className="text-[9px] text-dark-muted font-bold uppercase tracking-wider">Computed GEE Indices</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="flex justify-between items-center p-2 bg-[#050a17]/50 rounded border border-dark-border/40 text-[10px]">
                          <span className="text-slate-400 font-medium">NDVI</span>
                          <span className="font-extrabold text-quantum-cyan">{geeData.indices.ndvi_mean.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-[#050a17]/50 rounded border border-dark-border/40 text-[10px]">
                          <span className="text-slate-400 font-medium">EVI</span>
                          <span className="font-extrabold text-quantum-cyan">{geeData.indices.evi_mean.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-[#050a17]/50 rounded border border-dark-border/40 text-[10px]">
                          <span className="text-slate-400 font-medium">SAVI</span>
                          <span className="font-extrabold text-quantum-cyan">{geeData.indices.savi_mean.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-[#050a17]/50 rounded border border-dark-border/40 text-[10px]">
                          <span className="text-slate-400 font-medium">NDWI</span>
                          <span className="font-extrabold text-quantum-cyan">{geeData.indices.ndwi_mean.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-[#050a17]/50 rounded border border-dark-border/40 text-[10px]">
                          <span className="text-slate-400 font-medium">GNDVI</span>
                          <span className="font-extrabold text-slate-300">{geeData.extra_indices.gndvi_mean.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-[#050a17]/50 rounded border border-dark-border/40 text-[10px]">
                          <span className="text-slate-400 font-medium">MSAVI</span>
                          <span className="font-extrabold text-slate-300">{geeData.extra_indices.msavi_mean.toFixed(4)}</span>
                        </div>
                      </div>
                      <div className="flex justify-between items-center p-2 bg-[#050a17]/50 rounded border border-dark-border/40 text-[10px]">
                        <span className="text-slate-400 font-medium">Chlorophyll Index</span>
                        <span className="font-extrabold text-quantum-cyan">{geeData.indices.ci_mean.toFixed(4)}</span>
                      </div>
                    </div>

                    {/* Quantum Prediction Pipeline outputs */}
                    <div className="space-y-2 border-t border-dark-border/60 pt-3">
                      <p className="text-[9px] text-dark-muted font-bold uppercase tracking-wider">Crop Health & Yield Forecast</p>
                      
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 bg-slate-900/60 rounded border border-dark-border text-center">
                          <p className="text-[8px] text-dark-muted font-semibold">CROP HEALTH</p>
                          <p className="text-xs font-bold text-quantum-emerald">{geeData.crop_health}</p>
                        </div>
                        <div className="p-2 bg-slate-900/60 rounded border border-dark-border text-center">
                          <p className="text-[8px] text-dark-muted font-semibold">PREDICTED CROP</p>
                          <p className="text-xs font-bold text-quantum-cyan">{geeData.predicted_crop}</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-center">
                        <div className="p-2 bg-slate-950/40 rounded border border-dark-border">
                          <p className="text-[8px] text-dark-muted font-semibold">EST. YIELD</p>
                          <p className="text-xs font-extrabold text-slate-200">{geeData.yield_t_ha.toFixed(2)} t/ha</p>
                        </div>
                        <div className="p-2 bg-slate-950/40 rounded border border-dark-border">
                          <p className="text-[8px] text-dark-muted font-semibold">DISEASE RISK</p>
                          <p className={`text-xs font-extrabold ${(geeData.disease_probability > 0.4) ? 'text-quantum-rose' : 'text-slate-300'}`}>
                            {(geeData.disease_probability * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>

                      <div className="p-2.5 bg-slate-900/30 rounded border border-dark-border flex justify-between items-center text-[10px]">
                        <span className="text-dark-muted font-semibold">QUANTUM CONFIDENCE</span>
                        <span className="font-bold text-quantum-cyan">{(geeData.quantum_confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    {/* PDF Download Button */}
                    <button
                      onClick={downloadPdfReport}
                      className="w-full mt-4 flex items-center justify-center gap-2 py-2 rounded-lg bg-gradient-to-r from-quantum-cyan to-quantum-emerald hover:opacity-90 text-slate-950 font-bold text-xs shadow-quantum-glow transition-all duration-300"
                    >
                      <Download size={14} />
                      <span>Download GEE Crop Report (PDF)</span>
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-48 border border-dashed border-dark-border rounded-lg text-dark-muted">
                    <Loader2 size={24} className="animate-spin mb-2" />
                    <p className="text-[9px] text-center px-4">Contacting Google Earth Engine catalog... resolving coordinates.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 border border-dashed border-dark-border rounded-lg text-dark-muted">
                <Satellite size={32} className="mb-2 text-dark-muted/60" />
                <p className="text-[10px] text-center px-4">Click any agricultural cadastre polygon on the map, upload a boundary, or draw one using drawing tools to retrieve Earth Engine telemetry.</p>
              </div>
            )}
          </div>
          
          <div className="border-t border-dark-border pt-4 mt-4 text-[9px] text-dark-muted flex justify-between">
            <span>GEE Integration Active</span>
            <span>Synced at: {new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardOverview;
