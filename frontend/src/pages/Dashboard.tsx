import React, { useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from '../components/Sidebar'

const Dashboard: React.FC = () => {
  const [uploadData, setUploadData] = useState<any>(null);
  const [predictions, setPredictions] = useState<any>(null);

  return (
    <div className="flex h-screen bg-[#040814] text-slate-100 overflow-hidden font-sans gradient-dark-quantum">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-y-auto">
        <main className="flex-1 p-6 md:p-8">
          <Outlet context={{ uploadData, setUploadData, predictions, setPredictions }} />
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
