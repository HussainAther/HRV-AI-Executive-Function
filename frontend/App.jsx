import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import GamesPage from "./pages/Games";

export default function App() {
  return (
    <Router>
      <nav className="bg-gray-100 p-4 mb-4 shadow">
        <Link to="/" className="mr-4 font-semibold">Dashboard</Link>
        <Link to="/games" className="font-semibold">Games</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/games" element={<GamesPage />} />
      </Routes>
    </Router>
  );
}

