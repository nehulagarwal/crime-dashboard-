import React from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Overview from './pages/Overview';
import Trends from './pages/Trends';
import Cities from './pages/Cities';
import Predictions from './pages/Predictions';
import Models from './pages/Models';
import Fairness from './pages/Fairness';

function App() {
  return (
    <HashRouter>

      <Navbar />

      <div style={{ paddingTop: 56 }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/trends" element={<Trends />} />
          <Route path="/cities" element={<Cities />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/models" element={<Models />} />
          <Route path="/fairness" element={<Fairness />} />

        </Routes>
      </div>

    </HashRouter>
  );
}

export default App;